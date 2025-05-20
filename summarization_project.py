# -*- coding: utf-8 -*-
import sys
import math
import random
import yaml 
import argparse 
import logging 
import traceback 
import nltk 
from copy import deepcopy 
from functools import partial 
import os 

# Настройка переменных окружения для корректной работы библиотек
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Отключает параллелизм токенизаторов Hugging Face 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Порядок нумерации GPU в соответствии с PCI шиной
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import torch

# Подавление избыточных логов TensorFlow и ограничение использования GPU им 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
try:
    import tensorflow as tf
    if tf.config.list_physical_devices('GPU'): 
        tf.config.set_visible_devices([], 'GPU') 
except ImportError: 
    pass 
except Exception as e_tf: 
    logging.warning(f"Could not restrict TensorFlow GPU usage: {e_tf}")

# Основные импорты ML/DL библиотек
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

# Импорты Hugging Face
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, disable_caching
import evaluate 

from einops import rearrange

# Отключаем кеширование datasets
disable_caching()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging configured.")

# Проверка NLTK данных
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logging.warning("NLTK 'punkt' resource not found locally. Downloading 'punkt'.")
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt') 
        logging.info("NLTK 'punkt' resource downloaded successfully.")
    except Exception as e_nltk_download:
        logging.error(f"Failed to download NLTK 'punkt': {e_nltk_download}. nltk.sent_tokenize might fail.")

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ 

def load_config(config_path):
    """Загружает конфигурацию из YAML файла."""
    try:
        with open(config_path, 'r') as f: # Открываем файл конфигурации для чтения
            config = yaml.safe_load(f) 
        logging.info(f"Configuration loaded from {config_path}") 
        return config
    except FileNotFoundError: 
        logging.error(f"Configuration file not found at {config_path}")
        raise 
    except Exception as e: 
        logging.error(f"Error loading configuration: {e}")
        raise 

# Код, связанный с Baseline моделью (T5) 

def tokenize_function_baseline(examples, tokenizer, config_baseline):
    """Токенизирует входные документы и целевые саммари для baseline модели."""
    # Добавляем префикс к документам 
    inputs = [config_baseline["prefix"] + doc for doc in examples["document"]]
    # Токенизируем документы
    model_inputs = tokenizer(inputs,
                             max_length=config_baseline["max_doc_len"], # Максимальная длина документа
                             truncation=True, # Обрезать, если длиннее max_length
                             padding="max_length") # Дополнить до max_length, если короче
    # Токенизируем целевые саммари (метки)
    labels = tokenizer(text_target=examples["summary"],
                       max_length=config_baseline["max_summary_len"], # Максимальная длина саммари
                       truncation=True,
                       padding="max_length")
    model_inputs["labels"] = labels["input_ids"] # Сохраняем ID токенов саммари как метки
    return model_inputs

def compute_metrics_baseline(eval_pred, tokenizer, rouge_metric):
    """Вычисляет метрики (ROUGE, длина генерации) для baseline модели."""
    predictions, labels = eval_pred # Распаковываем предсказания и истинные метки
    # Заменяем -100 (игнорируемые токены) на pad_token_id для декодирования
    predictions = np.where(predictions == -100, tokenizer.pad_token_id, predictions)
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Декодируем ID токенов обратно в текст
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    try:
        # Для ROUGE часто требуется, чтобы каждое предложение было на новой строке
        decoded_preds_nltk = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_nltk = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    except Exception as e_nltk: # Если токенизация NLTK не удалась
        logging.error(f"NLTK sentence tokenization failed: {e_nltk}. Falling back to simple strip.")
        decoded_preds_nltk = [pred.strip() for pred in decoded_preds] # Используем просто очищенный текст
        decoded_labels_nltk = [label.strip() for label in decoded_labels]

    if not decoded_preds_nltk or not decoded_labels_nltk: # Если нет валидных предсказаний/меток
        logging.warning("Empty predictions or labels for ROUGE, returning empty results.")
        return {}

    try:
        # Вычисляем ROUGE метрики
        result = rouge_metric.compute(predictions=decoded_preds_nltk, references=decoded_labels_nltk, use_stemmer=True)
        result = {key: value * 100 for key, value in result.items()} 
    except Exception as e: # Если вычисление ROUGE не удалось
        logging.error(f"Error computing ROUGE: {e}")
        result = {}

    # Вычисляем среднюю длину сгенерированных саммари
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens) if prediction_lens else 0
    return {k: round(v, 4) for k, v in result.items()} # Округляем результаты

# Функции и Классы для Диффузионной Модели 

def tokenize_function_diffusion(examples, tokenizer, config_diffusion):
    """Токенизирует входные документы и целевые саммари для диффузионной модели."""
    # Токенизируем документы
    doc_outputs = tokenizer(
        examples["document"],
        max_length=config_diffusion["max_doc_len"],
        truncation=True,
        padding="max_length"
    )
    # Токенизируем целевые саммари
    summary_outputs = tokenizer(
        text_target=examples["summary"],
        max_length=config_diffusion["max_summary_len"],
        truncation=True,
        padding="max_length"
    )
    # Возвращаем словарь с ID токенов и масками внимания для документов и саммари
    return {
        "document_input_ids": doc_outputs["input_ids"],
        "document_attention_mask": doc_outputs["attention_mask"],
        "summary_input_ids": summary_outputs["input_ids"],
        "summary_attention_mask": summary_outputs["attention_mask"],
    }

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Генерирует линейное расписание для коэффициентов beta диффузионного процесса."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    """Генерирует косинусное расписание для коэффициентов beta (более плавное)."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) # Ограничиваем значения beta для стабильности

def get_diffusion_variables(schedule_type, timesteps, beta_start, beta_end, device):
    """Рассчитывает и возвращает все необходимые переменные для диффузионного процесса."""
    # Выбираем тип расписания для beta
    if schedule_type == "linear":
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == "cosine":
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")

    # Рассчитываем alpha, alpha_cumprod и другие связанные переменные
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0) # Кумулятивное произведение alpha
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # alpha_cumprod на предыдущем шаге

    # Переменные, используемые в прямом и обратном диффузионных процессах
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)

    # Переменные для обратного процесса (семплирования)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # Дисперсия апостериорного распределения
    posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20)) # Логарифм дисперсии (ограниченный)
    posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod) # Коэффициент для x_start в апостериорном среднем
    posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod) # Коэффициент для x_t в апостериорном среднем

    # Собираем все переменные в словарь
    diffusion_vars = {
        "betas": betas, "alphas_cumprod": alphas_cumprod, "alphas_cumprod_prev": alphas_cumprod_prev,
        "alphas": alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "log_one_minus_alphas_cumprod": log_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
        "posterior_log_variance_clipped": posterior_log_variance_clipped,
        "posterior_mean_coef1": posterior_mean_coef1,
        "posterior_mean_coef2": posterior_mean_coef2,
    }
    # Переносим все тензоры на указанное устройство (CPU/GPU) и приводим к типу float
    for key in diffusion_vars:
        diffusion_vars[key] = diffusion_vars[key].to(device).float()
    return diffusion_vars

def extract(a, t, x_shape):
    """Извлекает значения из тензора 'a' по индексам 't' и придает им форму, совместимую с 'x_shape'."""
    batch_size = t.shape[0]
    # Выбираем значения из 'a' по индексам 't' (для каждого элемента батча свой временной шаг)
    out = a.gather(0, t)
    # Изменяем форму так, чтобы можно было выполнять батчевые операции с 'x'
    # Например, если x_shape=(B, C, H, W), то out станет (B, 1, 1, 1)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def q_sample(x_start, t, diffusion_vars, noise=None):
    """Прямой диффузионный процесс: добавляет шум к x_start на шаге t."""
    if noise is None: # Если шум не предоставлен, генерируем случайный гауссовский шум
        noise = torch.randn_like(x_start)
    # Извлекаем нужные коэффициенты для шага t
    sqrt_alphas_cumprod_t = extract(diffusion_vars["sqrt_alphas_cumprod"], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_vars["sqrt_one_minus_alphas_cumprod"], t, x_start.shape)
    # Формула прямого процесса: x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class SinusoidalPositionEmbeddings(nn.Module):
    """Модуль для создания синусоидальных позиционных эмбеддингов для времени t."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim # Размерность эмбеддинга

    def forward(self, time):
        """Создает эмбеддинги для временных шагов 'time'."""
        device = time.device
        half_dim = self.dim // 2
        # Математическая формула для синусоидальных эмбеддингов
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] # (batch_size, 1) * (1, half_dim) -> (batch_size, half_dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # Конкатенируем sin и cos
        if self.dim % 2 == 1: # Если размерность нечетная, добавляем нулевой столбец
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

# Функция потерь для диффузионной модели (обычно MSE между предсказанным и истинным шумом)
loss_fn_diffusion = nn.MSELoss()

def embeddings_to_ids(embeddings, token_embedding_matrix):
    """Преобразует непрерывные эмбеддинги в дискретные ID токенов."""
    device = embeddings.device
    embeddings = embeddings.to(device).float() # Убеждаемся, что эмбеддинги на нужном устройстве и типа float
    token_embedding_matrix = token_embedding_matrix.to(device).float() # Матрица эмбеддингов словаря

    # Нормализуем эмбеддинги и матрицу для вычисления косинусного сходства
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    matrix_norm = F.normalize(token_embedding_matrix, p=2, dim=-1)

    # Вычисляем косинусное сходство между каждым эмбеддингом и всеми эмбеддингами в словаре
    cosine_sim = torch.matmul(embeddings_norm, matrix_norm.t()) # (B, SeqLen, EmbDim) @ (EmbDim, VocabSize) -> (B, SeqLen, VocabSize)
    # Выбираем ID токена с максимальным сходством
    output_ids = torch.argmax(cosine_sim, dim=-1)

    # Проверка на выход за пределы словаря 
    vocab_size = token_embedding_matrix.shape[0]
    if output_ids.max() >= vocab_size or output_ids.min() < 0 :
        logging.warning(
            f"embeddings_to_ids generated OOB index! Max ID: {output_ids.max().item()}, Min ID: {output_ids.min().item()}, Vocab size: {vocab_size}. Clipping IDs."
        )
        output_ids = torch.clamp(output_ids, 0, vocab_size - 1) # Ограничиваем ID границами словаря
    return output_ids

class ConditionalDiffusionSummarizer(nn.Module):
    """Условная диффузионная модель для суммаризации текста (архитектура Трансформер)."""
    def __init__(self, vocab_size, embed_dim, seq_len_doc, seq_len_summ,
                 encoder_layers, decoder_layers, num_heads, time_embed_dim=128, dropout=0.1,
                 pad_token_id=0):
        super().__init__()
        # Сохраняем основные параметры
        self.seq_len_doc = seq_len_doc 
        self.seq_len_summ = seq_len_summ 
        self.embed_dim = embed_dim 
        self.vocab_size = vocab_size 
        self.pad_token_id = pad_token_id 

        # Слои эмбеддингов
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id) 
        self.doc_pos_embedding = nn.Parameter(torch.randn(1, seq_len_doc, embed_dim) * 0.02) 
        self.summ_pos_embedding = nn.Parameter(torch.randn(1, seq_len_summ, embed_dim) * 0.02) 

        # MLP для преобразования временного эмбеддинга t
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim), # Синусоидальные эмбеддинги для времени
            nn.Linear(time_embed_dim, embed_dim * 4), 
            nn.Mish(), # Функция активации Mish
            nn.Linear(embed_dim * 4, embed_dim), 
        )

        # Энкодер Трансформера для обработки документа (контекста)
        encoder_layer_cfg = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, 
            dropout=dropout, batch_first=True, activation='gelu' 
        )
        encoder_norm = nn.LayerNorm(embed_dim) # Нормализация после энкодера
        self.encoder = nn.TransformerEncoder(encoder_layer_cfg, num_layers=encoder_layers, norm=encoder_norm)

        # Декодер Трансформера для обработки зашумленного саммари и предсказания шума
        decoder_layer_cfg = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        decoder_norm = nn.LayerNorm(embed_dim) # Нормализация после декодера
        self.decoder = nn.TransformerDecoder(decoder_layer_cfg, num_layers=decoder_layers, norm=decoder_norm)

        # Выходной слой для предсказания шума (размерность эмбеддинга)
        self.output_layer = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout) # Слой Dropout

        # Применяем инициализацию весов ко всем модулям
        self.apply(self._init_weights)

        # Специальная инициализация для выходного слоя (после общей)
        if hasattr(self, 'output_layer') and isinstance(self.output_layer, nn.Linear):
            target_std_for_weights = 1.0 / math.sqrt(self.output_layer.in_features) # Целевое ст.откл. для весов
            init.normal_(self.output_layer.weight, mean=0.0, std=target_std_for_weights) # Нормальная инициализация
            if self.output_layer.bias is not None:
                init.zeros_(self.output_layer.bias) # Нулевая инициализация смещений

    def _init_weights(self, module):
        """Инициализирует веса для различных типов слоев."""
        if isinstance(module, nn.Linear): # Для линейных слоев
            init.xavier_uniform_(module.weight) # Инициализация Ксавье (равномерная)
            if module.bias is not None:
                init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding): # Для слоев эмбеддингов
            module.weight.data.normal_(mean=0.0, std=0.02) 
            if module.padding_idx is not None: # Если есть паддинг-токен
                with torch.no_grad(): # Не отслеживаем градиенты для этой операции
                    module.weight[module.padding_idx].zero_() # Зануляем эмбеддинг паддинг-токена
        elif isinstance(module, nn.LayerNorm): # Для слоев нормализации
            if module.bias is not None:
                init.zeros_(module.bias) # Нулевые смещения (beta)
            if module.weight is not None:
                init.ones_(module.weight) # Единичные веса (gamma)

    def _create_transformer_padding_mask(self, input_ids):
        """Создает маску для паддинг-токенов (True, где паддинг)."""
        return (input_ids == self.pad_token_id)

    def encode_document(self, doc_ids, doc_attention_mask=None, embedding_scaling_factor_doc=1.0):
        """Кодирует входной документ в контекстное представление."""
        batch_size, seq_len = doc_ids.shape
        # Обрезаем документ, если он длиннее максимальной длины
        if seq_len > self.seq_len_doc:
            doc_ids = doc_ids[:, :self.seq_len_doc]
            if doc_attention_mask is not None:
                doc_attention_mask = doc_attention_mask[:, :self.seq_len_doc]
            seq_len = self.seq_len_doc

        doc_embeds_raw = self.token_embedding(doc_ids) 
        doc_embeds = doc_embeds_raw * embedding_scaling_factor_doc 
        doc_pos = self.doc_pos_embedding[:, :seq_len, :] 
        x = doc_embeds + doc_pos # Суммируем эмбеддинги токенов и позиций
        x = self.dropout(x) # Применяем Dropout

        # Создаем маску для паддинг-токенов для энкодера Трансформера
        src_key_padding_mask_bool = None
        if doc_attention_mask is not None: # Если есть маска внимания от токенизатора
            src_key_padding_mask_bool = (doc_attention_mask == 0) # 0 в attention_mask -> True в padding_mask
        elif doc_ids is not None: # Если нет маски, но есть ID, создаем на лету
             src_key_padding_mask_bool = self._create_transformer_padding_mask(doc_ids)

        # Пропускаем через энкодер
        encoder_output = self.encoder(x, src_key_padding_mask=src_key_padding_mask_bool)
        return encoder_output

    def decode_summary(self, noisy_summary_embeds, timesteps, context_embedding,
                       summary_input_ids=None, doc_attention_mask=None):
        """Декодирует зашумленные эмбеддинги саммари, используя контекст, для предсказания шума."""
        batch_size, seq_len, embed_dim_actual = noisy_summary_embeds.shape
        if embed_dim_actual != self.embed_dim: # Проверка размерности
            raise ValueError(f"Input embedding dimension ({embed_dim_actual}) != model embed_dim ({self.embed_dim})")
        if seq_len > self.seq_len_summ:
            noisy_summary_embeds = noisy_summary_embeds[:, :self.seq_len_summ, :]
            if summary_input_ids is not None:
                summary_input_ids = summary_input_ids[:, :self.seq_len_summ]
            seq_len = self.seq_len_summ

        summ_pos = self.summ_pos_embedding[:, :seq_len, :] 
        y = noisy_summary_embeds + summ_pos # Добавляем позиционные эмбеддинги
        time_embeds = self.time_mlp(timesteps).unsqueeze(1).repeat(1, seq_len, 1) # Эмбеддинги времени, растянутые по длине последовательности
        y = y + time_embeds # Добавляем временные эмбеддинги
        y = self.dropout(y) 

        # Создаем маску для последующих токенов (causal mask) для декодера Трансформера
        tgt_attn_mask_bool = None
        if seq_len > 0:
            _tgt_mask_float = nn.Transformer.generate_square_subsequent_mask(seq_len, device=y.device)
            tgt_attn_mask_bool = torch.isneginf(_tgt_mask_float) 

        # Создаем маску для паддинг-токенов в целевой последовательности (саммари)
        tgt_padding_mask_bool = None
        if summary_input_ids is not None: # Используется во время обучения
            if summary_input_ids.shape[1] != seq_len:
                 summary_input_ids = summary_input_ids[:, :seq_len]
            tgt_padding_mask_bool = self._create_transformer_padding_mask(summary_input_ids)

        # Создаем маску для паддинг-токенов в памяти (выход энкодера)
        memory_padding_mask_bool = None
        if doc_attention_mask is not None: # Используем маску внимания от документа
            memory_seq_len = context_embedding.shape[1]
            if doc_attention_mask.shape[1] > memory_seq_len:
                doc_attention_mask = doc_attention_mask[:, :memory_seq_len]
            elif doc_attention_mask.shape[1] < memory_seq_len:
                padding_needed = memory_seq_len - doc_attention_mask.shape[1]
                doc_attention_mask = F.pad(doc_attention_mask, (0, padding_needed), value=1)
            memory_padding_mask_bool = (doc_attention_mask == 0)

        # Пропускаем через декодер
        decoder_output = self.decoder(
            tgt=y, memory=context_embedding, # Целевая последовательность и память (контекст)
            tgt_mask=tgt_attn_mask_bool, 
            memory_mask=None, 
            tgt_key_padding_mask=tgt_padding_mask_bool, 
            memory_key_padding_mask=memory_padding_mask_bool 
        )
        predicted_noise = self.output_layer(decoder_output) # Предсказываем шум
        return predicted_noise

    def forward(self, doc_ids, summary_ids, timesteps, diffusion_vars, noise=None,
                doc_attention_mask=None, summary_attention_mask=None,
                embedding_scaling_factor=1.0,
                embedding_scaling_factor_doc=1.0):
        """Полный проход модели: кодирование документа, зашумление саммари, предсказание шума."""
        # 1. Кодируем документ для получения контекста
        context_embedding = self.encode_document(
            doc_ids, doc_attention_mask,
            embedding_scaling_factor_doc=embedding_scaling_factor_doc
        )

        # 2. Получаем "чистые" эмбеддинги саммари (x_start)
        y_start_raw = self.token_embedding(summary_ids)
        # Масштабируем эмбеддинги (важно для стабильности диффузии, std ~ 1)
        y_start_scaled = y_start_raw * embedding_scaling_factor

        # 3. Генерируем или используем предоставленный шум
        if noise is None:
            noise = torch.randn_like(y_start_scaled) # Гауссовский шум

        # 4. Применяем прямой диффузионный процесс (зашумляем y_start_scaled)
        y_noisy = q_sample(x_start=y_start_scaled, t=timesteps, diffusion_vars=diffusion_vars, noise=noise)

        # 5. Предсказываем добавленный шум с помощью декодера
        predicted_noise = self.decode_summary(
            noisy_summary_embeds=y_noisy, 
            timesteps=timesteps, 
            context_embedding=context_embedding, 
            summary_input_ids=summary_ids, 
            doc_attention_mask=doc_attention_mask 
        )
        return predicted_noise, noise # Возвращаем предсказанный шум и истинный (целевой) шум


@torch.no_grad() # Отключаем вычисление градиентов для этой функции 
def p_sample_ddpm(model, y_t, t_tensor, t_index, context_embedding, diffusion_vars,
                  summary_input_ids_for_mask=None, doc_attention_mask=None):
    """Один шаг обратного диффузионного процесса по схеме DDPM."""
    # Извлекаем необходимые коэффициенты для текущего временного шага t
    betas_t = extract(diffusion_vars["betas"], t_tensor, y_t.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(diffusion_vars["sqrt_one_minus_alphas_cumprod"], t_tensor, y_t.shape)
    # alphas_t = extract(diffusion_vars["alphas"], t_tensor, y_t.shape) # Не используется напрямую в этой формуле DDPM mean
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / diffusion_vars["alphas"]), t_tensor, y_t.shape) # 1/sqrt(alpha_t)

    # Предсказываем шум с помощью модели
    predicted_noise = model.decode_summary(
        y_t, t_tensor, context_embedding, summary_input_ids_for_mask, doc_attention_mask
    )

    # Рассчитываем среднее значение для x_{t-1} по формуле DDPM
    model_mean = sqrt_recip_alphas_t * (
        y_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0: # Если это последний шаг (t=0), не добавляем шум
        return model_mean
    else:
        # Добавляем стохастический шум для шагов t > 0
        posterior_variance_t = extract(diffusion_vars["posterior_variance"], t_tensor, y_t.shape) # Дисперсия
        noise_p = torch.randn_like(y_t) # Случайный гауссовский шум
        return model_mean + torch.sqrt(posterior_variance_t) * noise_p # x_{t-1} = mean + sqrt(variance) * noise

@torch.no_grad() # Отключаем градиенты
def p_sample_ddim(model, y_t, t_tensor, t_prev_tensor, context_embedding, diffusion_vars, config_diffusion_runtime, eta=0.0,
                  summary_input_ids_for_mask=None, doc_attention_mask=None):
    """Один шаг обратного диффузионного процесса по схеме DDIM."""
    # 1. Предсказываем шум epsilon_theta(x_t, t)
    pred_noise_cond = model.decode_summary(
        y_t, t_tensor, context_embedding, summary_input_ids_for_mask, doc_attention_mask
    )

    # 2. Извлекаем alpha_prod_t и alpha_prod_t_prev
    alpha_prod_t = extract(diffusion_vars["alphas_cumprod"], t_tensor, y_t.shape)
    if t_prev_tensor[0] < 0: # Если t_prev = -1 (конец семплирования)
        alpha_prod_t_prev = torch.ones_like(alpha_prod_t) # alpha_prod_t_prev = 1
    else:
        alpha_prod_t_prev = extract(diffusion_vars["alphas_cumprod"], t_prev_tensor, y_t.shape)

    # 3. Предсказываем x_0_hat (оценка "чистых" данных)
    # Формула из DDIM: x_0_hat = (x_t - sqrt(1-alpha_prod_t) * eps_theta) / sqrt(alpha_prod_t)
    pred_y_start = (y_t - torch.sqrt(1. - alpha_prod_t + 1e-8) * pred_noise_cond) / torch.sqrt(alpha_prod_t + 1e-8)

    # 4. (Опционально) Клиппинг предсказанного x_0_hat
    x0_clamp_value = config_diffusion_runtime.get("x0_clamp_value_inference", None)
    if x0_clamp_value is not None:
        pred_y_start = torch.clamp(pred_y_start, -x0_clamp_value, x0_clamp_value)

    # 5. Рассчитываем sigma_t (стандартное отклонение шума, зависит от eta)
    # sigma_t^2 = eta^2 * (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
    term_under_sqrt_sigma = (1. - alpha_prod_t_prev) / (1. - alpha_prod_t + 1e-8) * (1. - alpha_prod_t / (alpha_prod_t_prev + 1e-8))
    sigma = eta * torch.sqrt(torch.clamp(term_under_sqrt_sigma, min=0.0)) # Гарантируем неотрицательность перед sqrt

    # 6. Рассчитываем x_{t-1}
    # x_{t-1} = sqrt(alpha_prod_t_prev) * x_0_hat +
    #           sqrt(1 - alpha_prod_t_prev - sigma_t^2) * eps_theta +  <-- направление к x_0
    #           sigma_t * random_noise                             <-- стохастический шум
    direction_term_coeff_sq = 1. - alpha_prod_t_prev - sigma**2 # Коэффициент для направления к x_0
    direction_term_coeff = torch.sqrt(torch.clamp(direction_term_coeff_sq, min=0.0))

    dir_xt = direction_term_coeff * pred_noise_cond # Компонента, направляющая к x_0

    # Генерируем стохастический шум (если eta > 0 и t_prev > 0)
    noise_p_sample = torch.randn_like(y_t) if t_prev_tensor[0] > 0 and eta > 0 else torch.zeros_like(y_t)
    # Собираем x_{t-1}
    y_prev = torch.sqrt(alpha_prod_t_prev) * pred_y_start + dir_xt + sigma * noise_p_sample
    return y_prev


@torch.no_grad() # Отключаем градиенты для всего цикла семплирования
def p_sample_loop_diffusion(model, shape, context_embedding, config_diffusion, diffusion_vars, device,
                            doc_attention_mask=None):
    """Полный цикл обратного диффузионного процесса (семплирования) для генерации саммари."""
    batch_size = shape[0] # Обычно 1 при генерации одного примера
    sampling_method = config_diffusion["sampling_method"] # 'ddpm' или 'ddim'
    total_timesteps = config_diffusion["timesteps"] # Общее количество шагов диффузии (T)
    inference_steps = config_diffusion["generation_steps_inference"] # Количество шагов для DDIM семплирования (меньше T)
    eta = config_diffusion.get("ddim_eta", 0.0) 

    y_t = torch.randn(shape, device=device) # Начинаем с чистого гауссовского шума x_T
    times = None # Инициализируем последовательность временных шагов

    # Определяем последовательность временных шагов в зависимости от метода семплирования
    if sampling_method == "ddim":
        # Для DDIM: от T-1 до 0 за `inference_steps` шагов
        times = torch.linspace(total_timesteps - 1, 0, steps=inference_steps + 1, device=device).long()
    elif sampling_method == "ddpm":
        # Для DDPM: от T-1 до 0, все `total_timesteps` шагов
        times = torch.arange(total_timesteps - 1, -1, -1, device=device).long()
        if len(times) != total_timesteps: # Проверка, что используются все шаги
             logging.warning(f"DDPM sampling implies using all {total_timesteps} steps. generation_steps_inference config may be misleading for DDPM.")
    else: 
        logging.error(f"Unknown sampling method: {sampling_method}")
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    if times is None: # Дополнительная проверка
        logging.error("'times' was not defined, check sampling_method logic.")
        raise ValueError("'times' was not defined, check sampling_method logic.")

    summary_ids_for_mask = None # Во время генерации у нас нет summary_ids для маски декодера

    # Настраиваем прогресс-бар
    actual_steps_for_progress = inference_steps if sampling_method == 'ddim' else total_timesteps
    progress_bar_desc = f"Generating Summary ({sampling_method.upper()} - {actual_steps_for_progress} steps)"

    # Цикл семплирования
    if sampling_method == "ddim":
        time_pairs = list(zip(times[:-1], times[1:])) # Пары (t_current, t_previous)
        progress_bar = tqdm(time_pairs, desc=progress_bar_desc, total=len(time_pairs), leave=False)
        for t, t_prev in progress_bar: # Итерируемся по парам временных шагов
            t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            t_prev_tensor = torch.full((batch_size,), t_prev.item(), device=device, dtype=torch.long)
            # Выполняем один шаг DDIM
            y_t = p_sample_ddim(model, y_t, t_tensor, t_prev_tensor, context_embedding, diffusion_vars,
                                config_diffusion, eta,
                                summary_ids_for_mask, doc_attention_mask)

    elif sampling_method == "ddpm":
        progress_bar = tqdm(range(total_timesteps - 1, -1, -1), desc=progress_bar_desc, total=total_timesteps, leave=False)
        for t_idx in progress_bar: # Итерируемся от T-1 до 0
            t_tensor = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            # Выполняем один шаг DDPM
            y_t = p_sample_ddpm(model, y_t, t_tensor, t_idx, context_embedding, diffusion_vars,
                                summary_ids_for_mask, doc_attention_mask)

    embedding_scaling_factor_inference = config_diffusion.get("embedding_scaling_factor", 1.0)
    final_y_start_raw = y_t / (embedding_scaling_factor_inference + 1e-8) # +1e-8 для избежания деления на ноль
    return final_y_start_raw

def summarize_diffusion(model, document_text, tokenizer, config_diffusion, diffusion_vars, device):
    """Генерирует саммари для одного документа с использованием диффузионной модели."""
    model.eval() 
    # 1. Токенизируем входной документ
    inputs = tokenizer(document_text,
                       max_length=config_diffusion["max_doc_len"],
                       truncation=True,
                       padding="max_length", 
                       return_tensors="pt") # Возвращаем PyTorch тензоры
    doc_ids = inputs["input_ids"].to(device) 
    doc_attention_mask = inputs["attention_mask"].to(device) 

    # Масштабирующий коэффициент для эмбеддингов документа при инференсе
    embedding_scaling_factor_doc_inference = config_diffusion.get("embedding_scaling_factor_doc",
                                                                    config_diffusion.get("embedding_scaling_factor", 1.0))
    with torch.no_grad(): # Отключаем вычисление градиентов
        # 2. Кодируем документ для получения контекста
        context_embedding = model.encode_document(
            doc_ids,
            doc_attention_mask,
            embedding_scaling_factor_doc=embedding_scaling_factor_doc_inference
        )
        # 3. Определяем целевую форму для генерируемых эмбеддингов саммари
        target_shape = (1, config_diffusion["max_summary_len"], config_diffusion["embed_dim"]) # (batch_size=1, seq_len, embed_dim)

        # 4. Запускаем цикл семплирования для генерации эмбеддингов саммари
        generated_summary_embeddings_raw = p_sample_loop_diffusion(
            model,
            target_shape,
            context_embedding,
            config_diffusion,
            diffusion_vars,
            device,
            doc_attention_mask=doc_attention_mask # Передаем маску документа для использования в декодере
        )
        
        # 5. Получаем матрицу эмбеддингов токенов из модели
        token_embedding_matrix = model.token_embedding.weight

        # 6. Преобразуем сгенерированные эмбеддинги в ID токенов
        summary_ids = embeddings_to_ids(generated_summary_embeddings_raw, token_embedding_matrix)
        summary_ids = summary_ids.squeeze(0) # Удаляем измерение батча (т.к. batch_size=1)
        # 7. Декодируем ID токенов обратно в текст
        summary_text = tokenizer.decode(summary_ids, skip_special_tokens=True)
    return summary_text


def train_diffusion_epoch(model, dataloader, optimizer, scheduler, loss_fn, config_diffusion,
                          diffusion_vars, device, epoch, enable_wandb):
    """Одна эпоха обучения диффузионной модели."""
    model.train() # Переключаем модель в режим обучения
    total_train_loss = 0 # Суммарная ошибка за эпоху
    accum_steps = config_diffusion.get("gradient_accumulation_steps", 1) # Шаги накопления градиента
    num_update_steps_per_epoch = math.ceil(len(dataloader) / accum_steps) # Количество обновлений весов за эпоху

    # Масштабирующие коэффициенты для эмбеддингов
    embedding_scaling_factor_train = config_diffusion.get("embedding_scaling_factor", 1.0)
    embedding_scaling_factor_doc_train = config_diffusion.get("embedding_scaling_factor_doc", embedding_scaling_factor_train)

    amp_enabled = (device.type == "cuda" and config_diffusion.get("amp_enabled", False))

    # Проверка, что функция потерь - это MSE с усреднением
    if not (isinstance(loss_fn, torch.nn.MSELoss) and loss_fn.reduction == 'mean'):
        logging.warning("loss_fn is not nn.MSELoss with reduction='mean'. "
                        "The loss calculation might not be as expected if elements are not handled carefully.")

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config_diffusion['num_epochs']} [Diff Training]")
    optimizer.zero_grad() # Обнуляем градиенты перед началом эпохи

    correlation_value_for_log = float('nan') # Для логирования корреляции 

    for batch_idx, batch in enumerate(progress_bar): 
        # Переносим данные батча на устройство (CPU/GPU)
        doc_ids = batch["document_input_ids"].to(device)
        summary_ids = batch["summary_input_ids"].to(device)
        doc_mask = batch["document_attention_mask"].to(device)
        summary_mask = batch["summary_attention_mask"].to(device) # Маска для саммари (1 - токен, 0 - паддинг)

        batch_size = doc_ids.shape[0]
        # Генерируем случайные временные шаги t для каждого примера в батче
        t = torch.randint(0, config_diffusion["timesteps"], (batch_size,), device=device).long()

        # Используем AMP, если включено
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            # Прямой проход модели: предсказываем шум и получаем истинный шум
            predicted_noise, target_noise = model(
                doc_ids, summary_ids, t, diffusion_vars,
                doc_attention_mask=doc_mask, summary_attention_mask=summary_mask,
                embedding_scaling_factor=embedding_scaling_factor_train,
                embedding_scaling_factor_doc=embedding_scaling_factor_doc_train
            )

            # Вычисление корреляции для первого батча первой эпохи 
            if batch_idx == 0 and epoch == 0:
                try:
                    _loss_mask_expanded_for_corr = summary_mask.unsqueeze(-1).expand_as(predicted_noise).bool()
                    active_predicted_noise_corr = predicted_noise[_loss_mask_expanded_for_corr].float().detach()
                    active_target_noise_corr = target_noise[_loss_mask_expanded_for_corr].float().detach()
                    if active_predicted_noise_corr.numel() > 1 and active_target_noise_corr.numel() > 1 and \
                       active_predicted_noise_corr.std() > 1e-9 and active_target_noise_corr.std() > 1e-9 and \
                       not torch.isnan(active_predicted_noise_corr).any() and not torch.isinf(active_predicted_noise_corr).any() and \
                       not torch.isnan(active_target_noise_corr).any() and not torch.isinf(active_target_noise_corr).any():
                        
                        stacked_vars = torch.stack([active_predicted_noise_corr.flatten(), active_target_noise_corr.flatten()])
                        correlation_matrix = torch.corrcoef(stacked_vars)
                        correlation_value_for_log = correlation_matrix[0, 1].item()
                except Exception as e_corr:
                    logging.warning(f"  Could not compute initial batch correlation: {e_corr}")

            # Вычисляем ошибку только по не-паддинг токенам саммари
            loss_mask_expanded = summary_mask.unsqueeze(-1).expand_as(predicted_noise).bool() # Расширяем маску до размерности шума
            num_active_elements = loss_mask_expanded.sum() # Количество активных (не-паддинг) элементов

            # Убеждаемся, что типы данных совпадают перед вычислением ошибки
            if predicted_noise.dtype != target_noise.dtype:
                effective_target_noise = target_noise.to(predicted_noise.dtype)
            else:
                effective_target_noise = target_noise
            
            if num_active_elements > 0: # Если есть активные элементы
                active_predicted = predicted_noise[loss_mask_expanded] 
                active_target = effective_target_noise[loss_mask_expanded] 
                loss = loss_fn(active_predicted, active_target) # MSE Loss усреднит по количеству активных элементов
            else: # Если все токены - паддинг
                loss = torch.tensor(0.0, device=predicted_noise.device, dtype=predicted_noise.dtype)
        
        loss_unscaled = loss # Сохраняем не масштабированную ошибку для логирования
        loss = loss / accum_steps # Масштабируем ошибку для накопления градиента

        current_loss_value_unaccumulated = loss_unscaled.item() # Текущее значение ошибки 

        # Обратный проход и накопление градиентов (AMP обрабатывает масштабирование)
        loss.backward()
        total_train_loss += current_loss_value_unaccumulated # Суммируем ошибку

        # Обновление весов, если достигнуто нужное количество шагов накопления
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Обрезка градиентов для предотвращения взрыва градиентов
            if config_diffusion.get("grad_clip_value", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config_diffusion["grad_clip_value"])
            optimizer.step() # Шаг оптимизатора (обновление весов)
            scheduler.step() # Шаг планировщика скорости обучения
            optimizer.zero_grad() # Обнуляем градиенты для следующего шага накопления

        progress_bar.set_postfix(loss=current_loss_value_unaccumulated) # Обновляем информацию в прогресс-баре

        # Логирование в Weights & Biases
        if enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader): # После каждого шага оптимизатора
                current_optimizer_step = (epoch * num_update_steps_per_epoch) + ((batch_idx + 1) // accum_steps)
                log_freq_wandb = config_diffusion.get("wandb_log_freq", 50) # Частота логирования в W&B
                if current_optimizer_step % log_freq_wandb == 0 :
                    log_data_batch = {
                        "diffusion_train_batch_loss": current_loss_value_unaccumulated,
                        "diffusion_learning_rate": scheduler.get_last_lr()[0], 
                        "diffusion_optimizer_step": current_optimizer_step # Общий номер шага оптимизатора
                    }
                    if batch_idx == 0 and epoch == 0 and not math.isnan(correlation_value_for_log):
                        log_data_batch["diffusion_train_corr_noise_initial"] = correlation_value_for_log # Логируем начальную корреляцию
                    try:
                        sys.modules['wandb'].log(log_data_batch) 
                    except Exception as e_wandb_log_train:
                        logging.warning(f"Could not log batch data to W&B: {e_wandb_log_train}")
    
    # Средняя ошибка за эпоху (на один батч)
    avg_loss_per_batch_epoch = total_train_loss / len(dataloader) if len(dataloader) > 0 else float('nan')
    return avg_loss_per_batch_epoch


def evaluate_diffusion_loss(model, dataloader, loss_fn, config_diffusion,
                            diffusion_vars, device, epoch, enable_wandb):
    """Оценка диффузионной модели на валидационном наборе (вычисление MSE ошибки предсказания шума)."""
    model.eval() # Переключаем модель в режим оценки
    total_val_loss = 0 # Суммарная ошибка на валидации
    # Масштабирующие коэффициенты и AMP 
    embedding_scaling_factor_eval = config_diffusion.get("embedding_scaling_factor", 1.0)
    embedding_scaling_factor_doc_eval = config_diffusion.get("embedding_scaling_factor_doc", embedding_scaling_factor_eval)
    amp_enabled_eval = (device.type == "cuda" and config_diffusion.get("amp_enabled", False))

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config_diffusion['num_epochs']} [Diff Validation]")
    with torch.no_grad(): # Отключаем вычисление градиентов для оценки
        for batch in progress_bar: # Итерация по валидационным батчам
            doc_ids = batch["document_input_ids"].to(device)
            summary_ids = batch["summary_input_ids"].to(device)
            doc_mask = batch["document_attention_mask"].to(device)
            summary_mask = batch["summary_attention_mask"].to(device)

            batch_size = doc_ids.shape[0]
            t = torch.randint(0, config_diffusion["timesteps"], (batch_size,), device=device).long()

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled_eval):
                predicted_noise, target_noise = model(
                    doc_ids, summary_ids, t, diffusion_vars,
                    doc_attention_mask=doc_mask, summary_attention_mask=summary_mask,
                    embedding_scaling_factor=embedding_scaling_factor_eval,
                    embedding_scaling_factor_doc=embedding_scaling_factor_doc_eval
                )

                # Вычисление ошибки 
                loss_mask_expanded = summary_mask.unsqueeze(-1).expand_as(predicted_noise).bool()
                num_active_elements = loss_mask_expanded.sum()

                if predicted_noise.dtype != target_noise.dtype:
                    effective_target_noise = target_noise.to(predicted_noise.dtype)
                else:
                    effective_target_noise = target_noise

                if num_active_elements > 0:
                    active_predicted = predicted_noise[loss_mask_expanded]
                    active_target = effective_target_noise[loss_mask_expanded]
                    loss = loss_fn(active_predicted, active_target)
                else:
                    loss = torch.tensor(0.0, device=predicted_noise.device, dtype=predicted_noise.dtype)
            
            total_val_loss += loss.item() # Суммируем ошибку
            progress_bar.set_postfix(loss=loss.item()) 
    
    # Средняя ошибка на валидационном наборе
    return total_val_loss / len(dataloader) if len(dataloader) > 0 else float('nan')

def evaluate_diffusion_rouge(model, dataloader, tokenizer, config_diffusion, diffusion_vars,
                             device, rouge_metric, num_samples=100):
    """Оценка диффузионной модели на тестовом наборе с использованием ROUGE метрик."""
    model.eval() # Режим оценки
    predictions, references = [], [] # Списки для предсказанных и эталонных саммари
    count = 0 # Счетчик сгенерированных примеров
    
    dataset_length = 0 
    if hasattr(dataloader, 'dataset') and dataloader.dataset is not None:
        dataset_length = len(dataloader.dataset)
    
    if dataset_length == 0: # Если датасет пуст
        logging.warning("Cannot evaluate ROUGE: DataLoader's dataset is empty or missing.")
        return {"num_evaluated": 0} 

    # Ограничиваем количество примеров для оценки ROUGE 
    max_samples_to_eval = min(num_samples, dataset_length)
    
    total_batches_for_progress = None 
    if hasattr(dataloader, 'batch_size') and dataloader.batch_size and max_samples_to_eval > 0:
        total_batches_for_progress = math.ceil(max_samples_to_eval / dataloader.batch_size)

    progress_bar = tqdm(dataloader, desc="Evaluating Diffusion ROUGE", total=total_batches_for_progress, leave=False)

    for batch_idx, batch in enumerate(progress_bar): # Итерация по батчам тестового набора
        if count >= max_samples_to_eval: break # Если достигли лимита примеров

        # Декодируем документы и эталонные саммари из ID токенов батча
        doc_texts_batch = tokenizer.batch_decode(batch["document_input_ids"], skip_special_tokens=True)
        ref_texts_batch = tokenizer.batch_decode(batch["summary_input_ids"], skip_special_tokens=True)

        for i in range(len(doc_texts_batch)): # Итерация по примерам внутри батча
            if count >= max_samples_to_eval: break
            doc_text = doc_texts_batch[i] 
            ref_text = ref_texts_batch[i] 
            try:
                # Генерируем саммари с помощью диффузионной модели
                pred_text = summarize_diffusion(model, doc_text, tokenizer, config_diffusion, diffusion_vars, device)
                if pred_text.strip() and ref_text.strip(): # Убеждаемся, что оба текста не пустые
                    predictions.append(pred_text)
                    references.append(ref_text)
                else: # Пропускаем, если один из текстов пуст
                    logging.warning(f"Skipping empty pred/ref for ROUGE sample {count}. Pred: '{pred_text}', Ref: '{ref_text}'")
                count += 1
                progress_bar.set_postfix({"Generated ROUGE": count}) 
            except Exception as e: 
                logging.error(f"Error during diffusion generation for ROUGE sample {count}: {e}")
                count += 1 

    if not predictions or not references: # Если не удалось сгенерировать валидные пары
        logging.warning("No valid (non-empty) samples were generated for diffusion ROUGE evaluation.")
        return {"num_evaluated": 0}
    try:
        # Подготовка текстов для ROUGE 
        decoded_preds_nltk = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
        decoded_labels_nltk = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]
        # Вычисление ROUGE метрик
        results = rouge_metric.compute(predictions=decoded_preds_nltk, references=decoded_labels_nltk, use_stemmer=True)
        results = {key: value * 100 for key, value in results.items()} # В проценты
        results["num_evaluated"] = len(predictions) # Количество оцененных примеров
        return results
    except Exception as e: 
        logging.error(f"Error during ROUGE calculation: {e}")
        return {"num_evaluated": len(predictions)} 

def generate_example(model, tokenizer, config_model_specific, test_ds, device, model_type,
                     enable_wandb, wandb_log_key,
                     diffusion_vars=None,
                     baseline_comparison_config=None,
                     baseline_model_path_for_comp=None):
    """Генерирует и выводит пример саммари для одного документа из тестового набора."""
    logging.info(f"\nGenerating {model_type} example summary...")
    if not test_ds or len(test_ds) == 0: # test_ds - это HF Dataset
        logging.warning("Test dataset is empty or not provided, cannot generate example.")
        return

    example_data = test_ds[0] # Берем первый пример из датасета
    document_text = example_data["document"] # Исходный документ
    reference_summary = example_data["summary"] # Эталонное саммари
    generated_summary = "N/A" # Инициализируем сгенерированное саммари

    # Выводим информацию о примере
    print("-" * 30)
    print(f"Original Document (truncated for {model_type}):")
    print(document_text[:500] + "...") 
    print("-" * 30)
    print(f"Reference Summary for {model_type}:")
    print(reference_summary)
    print("-" * 30)

    # Если генерируем для диффузионной модели и есть baseline для сравнения
    if model_type == "Diffusion" and baseline_model_path_for_comp and os.path.exists(baseline_model_path_for_comp) and baseline_comparison_config:
        try:
            logging.info(f"Loading baseline model from {baseline_model_path_for_comp} for comparison...")
            # Загружаем токенизатор и модель baseline
            baseline_tokenizer_comp = AutoTokenizer.from_pretrained(baseline_model_path_for_comp)
            baseline_model_comp = AutoModelForSeq2SeqLM.from_pretrained(baseline_model_path_for_comp).to(device)
            baseline_model_comp.eval() # Режим оценки

            # Готовим вход для baseline модели
            input_text_comp = baseline_comparison_config.get("prefix","summarize: ") + document_text
            inputs_comp = baseline_tokenizer_comp(
                input_text_comp, return_tensors="pt",
                max_length=baseline_comparison_config.get("max_doc_len", 512),
                truncation=True
            ).to(device)

            with torch.no_grad(): # Генерируем саммари baseline моделью
                outputs_comp = baseline_model_comp.generate(
                    **inputs_comp,
                    max_length=baseline_comparison_config.get("max_summary_len", 64) + 5,
                    num_beams=baseline_comparison_config.get("generation_num_beams", 4),
                    early_stopping=True
                )
            generated_summary_baseline_comp = baseline_tokenizer_comp.decode(outputs_comp[0], skip_special_tokens=True)
            print("BASELINE Generated Summary (for comparison):")
            print(generated_summary_baseline_comp)
            print("-" * 30)
            del baseline_model_comp, baseline_tokenizer_comp # Освобождаем память
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e_bl_comp: # Если не удалось загрузить/сгенерировать baseline
            logging.warning(f"Could not load/generate baseline for comparison from {baseline_model_path_for_comp}: {e_bl_comp}")

    # Генерация основной моделью (baseline или diffusion)
    try:
        model.eval() # Убеждаемся, что модель в режиме оценки
        if model_type == "Baseline T5": # Если это baseline модель
            input_text = config_model_specific.get("prefix", "summarize: ") + document_text
            inputs = tokenizer(
                input_text, return_tensors="pt",
                max_length=config_model_specific.get("max_doc_len", 512),
                truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=config_model_specific.get("max_summary_len", 64) + 5,
                    num_beams=config_model_specific.get("generation_num_beams", 4),
                    early_stopping=True
                )
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        elif model_type == "Diffusion": # Если это диффузионная модель
            if diffusion_vars is None: # Необходимы переменные диффузии
                raise ValueError("diffusion_vars must be provided for Diffusion model generation")
            generated_summary = summarize_diffusion(model, document_text, tokenizer, config_model_specific, diffusion_vars, device)
        else: # Неизвестный тип модели
            raise ValueError(f"Unknown model_type: {model_type}")

        print(f"{model_type.upper()} Generated Summary:")
        print(generated_summary)
        print("-" * 30)

        # Логирование примера в W&B
        if enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
            try:
                wandb_table_instance = sys.modules['wandb'].Table(columns=["Document", "Reference Summary", f"{model_type} Generated Summary"])
                wandb_table_instance.add_data(document_text, reference_summary, generated_summary)
                sys.modules['wandb'].log({wandb_log_key: wandb_table_instance})
            except Exception as e_wandb_genex:
                logging.warning(f"Could not log generation example to W&B: {e_wandb_genex}")

    except Exception as e_genex: 
        logging.error(f"Error during {model_type} example generation: {e_genex}")
        traceback.print_exc()


def run_baseline(config_baseline, train_ds, val_ds, test_ds, device, enable_wandb):
    """Запускает полный цикл обучения и оценки для baseline модели."""
    logging.info("\n--- Setting up Baseline Model (Loading Locally) ---")
    # Путь к локально сохраненным компонентам baseline модели
    baseline_local_path = os.path.join("offline_hf_components", config_baseline["model_name"])
    logging.info(f"Attempting to load baseline model and tokenizer from local path: {baseline_local_path}")

    if not os.path.isdir(baseline_local_path): 
        logging.error(f"Local path for baseline model not found: {baseline_local_path}. Cannot proceed with baseline.")
        return None, {} 

    baseline_tokenizer = None
    baseline_model = None
    try: # Загрузка токенизатора и модели
        logging.info(f"Loading baseline tokenizer from {baseline_local_path}...")
        baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_local_path)
        logging.info("Baseline tokenizer loaded successfully.")

        logging.info(f"Loading baseline model from {baseline_local_path}...")
        baseline_model_cpu = AutoModelForSeq2SeqLM.from_pretrained(baseline_local_path) # Сначала на CPU
        logging.info("Baseline model loaded to CPU. Moving to device...")
        baseline_model = baseline_model_cpu.to(device) # Затем на целевое устройство
        del baseline_model_cpu # Освобождаем память CPU
        if str(device) == "cuda" and torch.cuda.is_available(): torch.cuda.empty_cache() 
        logging.info(f"Baseline '{config_baseline['model_name']}' fully loaded and on {device}.")
        logging.info(f"Baseline model parameters: {sum(p.numel() for p in baseline_model.parameters() if p.requires_grad):,}") # Количество обучаемых параметров
    except Exception as e: # Если ошибка при загрузке
        logging.error(f"Failed during baseline model/tokenizer loading or moving to device: {e}")
        traceback.print_exc()
        return None, {}

    logging.info("Tokenizing datasets for Baseline...")
    try: # Токенизация датасетов
        tokenize_fn_baseline_partial = partial(tokenize_function_baseline, tokenizer=baseline_tokenizer, config_baseline=config_baseline)
        num_proc_map_baseline = config_baseline.get("num_proc_map", os.cpu_count() // 2 or 1) # Количество процессов для .map
        logging.info(f"Using {num_proc_map_baseline} processes for baseline dataset mapping.")
        
        cols_to_remove_baseline = [col for col in ["document", "summary", "id"] if col in train_ds.column_names] # Колонки для удаления

        train_tokenized_ds_baseline = train_ds.map(tokenize_fn_baseline_partial, batched=True, remove_columns=cols_to_remove_baseline, num_proc=num_proc_map_baseline)
        val_tokenized_ds_baseline = val_ds.map(tokenize_fn_baseline_partial, batched=True, remove_columns=cols_to_remove_baseline, num_proc=num_proc_map_baseline)
        test_tokenized_ds_baseline = test_ds.map(tokenize_fn_baseline_partial, batched=True, remove_columns=cols_to_remove_baseline, num_proc=num_proc_map_baseline)
        logging.info("Baseline tokenization complete.")
    except Exception as e: 
        logging.error(f"Failed during baseline tokenization: {e}")
        traceback.print_exc()
        return baseline_model, {} 

    # Коллатор данных для Seq2Seq задач
    data_collator_baseline = DataCollatorForSeq2Seq(
        tokenizer=baseline_tokenizer, model=baseline_model, padding="max_length",
        max_length=config_baseline["max_summary_len"],
        pad_to_multiple_of=8 if device.type == 'cuda' else None # Выравнивание для эффективности на GPU
    )
    logging.info("Baseline data collator created.")

    # Загрузка метрики ROUGE
    rouge_metric_baseline = None
    compute_metrics_fn_for_trainer = None
    try:
        rouge_script_path = config_baseline.get("rouge_script_path", "rouge") 
        trust_remote_rouge = config_baseline.get("rouge_trust_remote", False) 
        rouge_metric_baseline = evaluate.load(rouge_script_path, trust_remote_code=trust_remote_rouge)
        logging.info(f"ROUGE metric script loaded from '{rouge_script_path}' for baseline.")
        # Функция для вычисления метрик, передаваемая в Trainer
        compute_metrics_fn_for_trainer = partial(compute_metrics_baseline, tokenizer=baseline_tokenizer, rouge_metric=rouge_metric_baseline)
        logging.info("Compute metrics function for Trainer prepared.")
    except Exception as e:
        logging.error(f"Failed to load ROUGE metric script for baseline: {e}")
        logging.warning("Proceeding without ROUGE metric calculation for baseline.")

    # Создание директорий для сохранения результатов и модели
    os.makedirs(config_baseline["output_dir"], exist_ok=True)
    save_dir_for_model = os.path.dirname(config_baseline["save_path"])
    if not save_dir_for_model: save_dir_for_model = "."
    os.makedirs(save_dir_for_model, exist_ok=True)

    # Настройка скорости обучения
    baseline_lr = 1e-5 
    try:
        baseline_lr = float(config_baseline["learning_rate"])
        logging.info(f"Baseline Learning Rate set to: {baseline_lr}")
    except (TypeError, ValueError): # Если LR невалидный
        logging.error(f"Invalid baseline learning rate in config: {config_baseline.get('learning_rate', 'N/A')}. Must be a float.")
        logging.warning(f"Using default baseline learning rate: {baseline_lr}")

    # Расчет количества шагов на эпоху
    train_batch_size = config_baseline["batch_size"]
    grad_accum = config_baseline["grad_accum_steps"]
    if len(train_tokenized_ds_baseline) == 0 or train_batch_size == 0 or grad_accum == 0:
        steps_per_epoch = 1 # Избегаем деления на ноль
        logging.warning("Could not calculate steps_per_epoch for baseline due to zero train_size/batch_size/grad_accum. Defaulting to 1.")
    else:
        steps_per_epoch = math.ceil(len(train_tokenized_ds_baseline) / (train_batch_size * grad_accum))
    steps_per_epoch = max(1, steps_per_epoch) # Минимум 1 шаг
    logging.info(f"Baseline - Effective steps per epoch: ~{steps_per_epoch}.")

    # Аргументы для обучения Seq2SeqTrainer
    training_args_baseline = Seq2SeqTrainingArguments(
        output_dir=config_baseline["output_dir"], 
        eval_strategy="epoch", # Стратегия оценки 
        save_strategy="epoch", # Стратегия сохранения 
        learning_rate=baseline_lr, 
        per_device_train_batch_size=config_baseline["batch_size"], # Размер батча для обучения
        per_device_eval_batch_size=config_baseline.get("eval_batch_size", config_baseline["batch_size"] * 2), # Размер батча для оценки
        gradient_accumulation_steps=config_baseline["grad_accum_steps"], # Шаги накопления градиента
        weight_decay=config_baseline["weight_decay"], # Коэффициент затухания весов
        num_train_epochs=config_baseline["num_epochs"],
        predict_with_generate=True, # Использовать model.generate() для предсказаний
        generation_max_length=config_baseline["max_summary_len"], 
        generation_num_beams=config_baseline.get("generation_num_beams", 4), # Количество лучей для beam search
        fp16=torch.cuda.is_available() and device.type == 'cuda', # Использовать смешанную точность (fp16) на GPU
        load_best_model_at_end=True, 
        metric_for_best_model="eval_loss", # Метрика для определения лучшей модели
        greater_is_better=False, 
        push_to_hub=False, 
        report_to="wandb" if enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None else "none", 
        logging_steps=max(1, steps_per_epoch // 10 if steps_per_epoch > 10 else 1), # Частота логирования
        save_total_limit=2, # Максимальное количество сохраняемых чекпоинтов
        save_safetensors=True, 
        dataloader_num_workers=config_baseline.get("dataloader_num_workers", 0), # Количество воркеров для загрузки данных
    )

    if train_tokenized_ds_baseline is None or val_tokenized_ds_baseline is None: # Если датасеты не готовы
        logging.error("Tokenized datasets for baseline are missing, cannot create Trainer.")
        return baseline_model, {}

    # Создание объекта Trainer
    trainer_baseline = Seq2SeqTrainer(
        model=baseline_model,
        args=training_args_baseline,
        train_dataset=train_tokenized_ds_baseline,
        eval_dataset=val_tokenized_ds_baseline,
        tokenizer=baseline_tokenizer,
        data_collator=data_collator_baseline,
        compute_metrics=compute_metrics_fn_for_trainer, # Функция для вычисления метрик
    )

    logging.info("Starting Baseline Training...")
    baseline_model_after_train = None # Модель после обучения
    baseline_rouge_results_on_test = {} # Результаты ROUGE на тесте
    try: # Запуск обучения
        train_result = trainer_baseline.train()
        logging.info(f"Baseline training finished. Metrics: {train_result.metrics}")
        trainer_baseline.save_model(config_baseline["save_path"]) # Сохраняем лучшую модель
        logging.info(f"Best baseline model saved to {config_baseline['save_path']}")
        baseline_model_after_train = trainer_baseline.model 

        # Логирование артефакта модели в W&B, 
        if enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
            try:
                run_id = sys.modules['wandb'].run.id if sys.modules['wandb'].run else "unknown_run"
                baseline_artifact = sys.modules['wandb'].Artifact(f"baseline-model-{run_id}", type="model")
                baseline_artifact.add_dir(config_baseline["save_path"]) # Добавляем директорию с моделью в артефакт
                sys.modules['wandb'].log_artifact(baseline_artifact) 
            except Exception as e_art:
                logging.warning(f"Could not log baseline artifact to W&B: {e_art}")

        logging.info("Evaluating Baseline on Test Set...")
        if test_tokenized_ds_baseline is not None and compute_metrics_fn_for_trainer is not None: # Если есть тестовый набор и функция метрик
            logging.info("Proceeding with test set evaluation using ROUGE.")
            try:
                # Оценка на тестовом наборе
                baseline_eval_results = trainer_baseline.evaluate(test_tokenized_ds_baseline, metric_key_prefix="test")
                logging.info("\nBaseline ROUGE Results on Test Set:")
                # Фильтруем и форматируем результаты ROUGE
                baseline_rouge_results_on_test = {k.replace('test_', ''): v for k, v in baseline_eval_results.items() if k.startswith('test_') and ('rouge' in k or 'gen_len' in k)}
                for key, value in baseline_rouge_results_on_test.items():
                    if isinstance(value, (int, float)): logging.info(f"  {key}: {value:.4f}")
                    else: logging.info(f"  {key}: {value}")

                # Логирование результатов ROUGE в W&B
                if enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
                    try:
                        sys.modules['wandb'].log({f"baseline_test_{k}": v for k,v in baseline_rouge_results_on_test.items()})
                    except Exception as e_wandb_log:
                        logging.warning(f"Could not log baseline test results to W&B: {e_wandb_log}")
            except Exception as e_eval: # Если ошибка при оценке
                logging.error(f"ERROR during baseline test set evaluation: {e_eval}")
                traceback.print_exc()
        else: # Если нет данных для оценки
            logging.warning("Skipping ROUGE evaluation on test set for baseline (no data or no metric fn).")

    except Exception as e_train: # Если ошибка во время обучения
        logging.error(f"ERROR during Baseline Training: {e_train}")
        traceback.print_exc()
        baseline_model_after_train = trainer_baseline.model if hasattr(trainer_baseline, 'model') else None
    
    return baseline_model_after_train, baseline_rouge_results_on_test


def run_diffusion(config_diffusion, train_ds, val_ds, test_ds, device, enable_wandb,
                  baseline_model_path_for_comp=None,
                  baseline_comparison_config_for_generate_example=None):
    """Запускает полный цикл настройки, обучения и оценки для диффузионной модели."""

    logging.info("\n" + "="*50 + "\n=== Setting up Diffusion Model ===\n" + "="*50 + "\n")

    # Загрузка токенизатора для диффузионной модели 
    diffusion_tokenizer_local_path = os.path.join("offline_hf_components", config_diffusion["tokenizer_name"])
    logging.info(f"Attempting to load diffusion tokenizer from local path: {diffusion_tokenizer_local_path}")
    if not os.path.isdir(diffusion_tokenizer_local_path): 
        logging.error(f"Local path for diffusion tokenizer not found: {diffusion_tokenizer_local_path}. Cannot proceed with diffusion.")
        return
    
    diffusion_tokenizer = None
    try: # Загрузка токенизатора
        diffusion_tokenizer = AutoTokenizer.from_pretrained(diffusion_tokenizer_local_path)
        # Если у токенизатора нет pad_token, устанавливаем его равным eos_token или добавляем новый
        if diffusion_tokenizer.pad_token is None:
            if diffusion_tokenizer.eos_token is not None:
                diffusion_tokenizer.pad_token = diffusion_tokenizer.eos_token
                logging.info(f"Set diffusion tokenizer pad_token to eos_token: {diffusion_tokenizer.pad_token} (ID: {diffusion_tokenizer.pad_token_id})")
            else:
                new_pad_token = '[PAD]'
                num_added = diffusion_tokenizer.add_special_tokens({'pad_token': new_pad_token})
                logging.info(f"Added new pad_token: {new_pad_token} (ID: {diffusion_tokenizer.pad_token_id}). Tokens added: {num_added}.")
        logging.info(f"Diffusion tokenizer '{config_diffusion['tokenizer_name']}' loaded. Vocab size: {len(diffusion_tokenizer)}")
    except Exception as e: # Если ошибка при загрузке токенизатора
        logging.error(f"Failed to load diffusion tokenizer: {e}")
        traceback.print_exc()
        return

    # Токенизация датасетов для диффузионной модели 
    logging.info("Tokenizing datasets for Diffusion Model...")
    try:
        tokenize_fn_diff = partial(tokenize_function_diffusion, tokenizer=diffusion_tokenizer, config_diffusion=config_diffusion)
        num_proc_map_diffusion = config_diffusion.get("num_proc_map", os.cpu_count() // 2 or 1) # Количество процессов
        logging.info(f"Using {num_proc_map_diffusion} processes for diffusion dataset mapping.")
        
        cols_to_remove_diffusion = [col for col in ["document", "summary", "id"] if col in train_ds.column_names] # Колонки для удаления

        train_tokenized_ds_diffusion = train_ds.map(tokenize_fn_diff, batched=True, remove_columns=cols_to_remove_diffusion, num_proc=num_proc_map_diffusion)
        val_tokenized_ds_diffusion = val_ds.map(tokenize_fn_diff, batched=True, remove_columns=cols_to_remove_diffusion, num_proc=num_proc_map_diffusion)
        test_tokenized_ds_diffusion = test_ds.map(tokenize_fn_diff, batched=True, remove_columns=cols_to_remove_diffusion, num_proc=num_proc_map_diffusion)

        # Устанавливаем формат датасетов в PyTorch тензоры
        train_tokenized_ds_diffusion.set_format(type='torch')
        val_tokenized_ds_diffusion.set_format(type='torch')
        test_tokenized_ds_diffusion.set_format(type='torch')
        logging.info("Diffusion tokenization complete.")
    except Exception as e: 
        logging.error(f"Failed during diffusion tokenization: {e}")
        traceback.print_exc()
        return

    # Создание DataLoader'ов для диффузионной модели 
    num_workers_loader = config_diffusion.get("dataloader_num_workers", 0) 
    logging.info(f"Using {num_workers_loader} workers for Diffusion DataLoaders.")
    pin_memory_flag = (device.type == 'cuda' and num_workers_loader > 0) # Использовать pin_memory для ускорения на GPU

    train_dataloader_diffusion = DataLoader(train_tokenized_ds_diffusion, batch_size=config_diffusion["batch_size"], shuffle=True, num_workers=num_workers_loader, pin_memory=pin_memory_flag, drop_last=True) # drop_last=True для обучения
    eval_batch_size_diff = config_diffusion.get("eval_batch_size_diffusion", config_diffusion["batch_size"] * 2) # Размер батча для оценки
    val_dataloader_diffusion = DataLoader(val_tokenized_ds_diffusion, batch_size=eval_batch_size_diff, num_workers=num_workers_loader, pin_memory=pin_memory_flag)
    test_dataloader_diffusion = DataLoader(test_tokenized_ds_diffusion, batch_size=eval_batch_size_diff, num_workers=num_workers_loader, pin_memory=pin_memory_flag)
    logging.info(f"Diffusion DataLoaders created: Train {len(train_dataloader_diffusion)}, Val {len(val_dataloader_diffusion)}, Test {len(test_dataloader_diffusion)} batches")

    # Получение переменных диффузионного процесса 
    diffusion_variables_local = get_diffusion_variables(
        config_diffusion["noise_schedule"], config_diffusion["timesteps"],
        config_diffusion["beta_start"], config_diffusion["beta_end"], device
    )

    # Инициализация диффузионной модели 
    current_vocab_size = len(diffusion_tokenizer) # Актуальный размер словаря
    logging.info(f"Effective vocabulary size for diffusion model: {current_vocab_size}")

    diffusion_model = ConditionalDiffusionSummarizer(
        vocab_size=current_vocab_size,
        embed_dim=config_diffusion["embed_dim"],
        seq_len_doc=config_diffusion["max_doc_len"],
        seq_len_summ=config_diffusion["max_summary_len"],
        encoder_layers=config_diffusion["encoder_layers"],
        decoder_layers=config_diffusion["decoder_layers"],
        num_heads=config_diffusion["num_heads"],
        dropout=config_diffusion["dropout"],
        time_embed_dim=config_diffusion.get("time_embed_dim", 128),
        pad_token_id=diffusion_tokenizer.pad_token_id # Передаем ID паддинг-токена
    ).to(device)

    # Если размер словаря токенизатора изменился (например, добавили pad_token), изменяем размер слоя эмбеддингов модели
    if diffusion_model.token_embedding.num_embeddings != current_vocab_size:
        logging.warning(f"Resizing model token embeddings from {diffusion_model.token_embedding.num_embeddings} to {current_vocab_size} due to tokenizer vocab size change.")
        new_embedding_layer = nn.Embedding(current_vocab_size, config_diffusion["embed_dim"], padding_idx=diffusion_tokenizer.pad_token_id).to(device)
        diffusion_model._init_weights(new_embedding_layer) # Инициализируем новый слой
        
        num_to_copy = min(diffusion_model.token_embedding.num_embeddings, new_embedding_layer.num_embeddings) # Копируем старые веса
        with torch.no_grad():
            new_embedding_layer.weight.data[:num_to_copy, :] = diffusion_model.token_embedding.weight.data[:num_to_copy, :]
        
        diffusion_model.token_embedding = new_embedding_layer # Заменяем слой в модели
        diffusion_model.vocab_size = current_vocab_size # Обновляем размер словаря в модели
        logging.info(f"Model token embeddings resized. New vocab size: {current_vocab_size}")


    # Настройка оптимизатора и планировщика скорости обучения 
    lr_diffusion_str = str(config_diffusion.get("learning_rate", "1e-4"))
    lr_diffusion = 1e-4 
    try:
        lr_diffusion = float(lr_diffusion_str)
        logging.info(f"Diffusion Learning Rate set to: {lr_diffusion}")
    except (TypeError, ValueError): 
        logging.error(f"Invalid diffusion learning rate in config: '{lr_diffusion_str}'. Must be a float.")
        logging.warning(f"Using default diffusion learning rate: {lr_diffusion}")

    optimizer_name = config_diffusion.get("optimizer", "AdamW").lower() # Тип оптимизатора
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=lr_diffusion,
                                      weight_decay=config_diffusion.get("weight_decay_diffusion", 0.01),
                                      betas=config_diffusion.get("adam_betas", (0.9, 0.999)))
    else:
        logging.error(f"Unsupported optimizer: {optimizer_name}. Defaulting to AdamW.")
        optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=lr_diffusion,
                                      weight_decay=config_diffusion.get("weight_decay_diffusion", 0.01))

    # Расчет общего количества шагов обучения и шагов разогрева для планировщика
    grad_accum_steps = max(1, config_diffusion.get("gradient_accumulation_steps", 1))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_diffusion) / grad_accum_steps)
    total_steps = num_update_steps_per_epoch * config_diffusion["num_epochs"]

    scheduler_warmup_steps_val = config_diffusion.get("scheduler_warmup_steps", 0.1)
    if isinstance(scheduler_warmup_steps_val, float) and 0 < scheduler_warmup_steps_val <= 1.0: # Если доля от общих шагов
        actual_warmup_steps = int(total_steps * scheduler_warmup_steps_val)
    elif isinstance(scheduler_warmup_steps_val, int) and scheduler_warmup_steps_val >= 0: # Если абсолютное число шагов
        actual_warmup_steps = scheduler_warmup_steps_val
    else: # Невалидное значение
        logging.warning(f"Invalid scheduler_warmup_steps: {scheduler_warmup_steps_val}. Setting to 0.")
        actual_warmup_steps = 0
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=total_steps)
    
    logging.info(f"Diffusion model parameters: {sum(p.numel() for p in diffusion_model.parameters() if p.requires_grad):,}") # Количество обучаемых параметров
    logging.info(f"Total optimizer steps: {total_steps}, Grad Accum Steps: {grad_accum_steps}, Warmup steps: {actual_warmup_steps}")

    # Настройка W&B watch
    wandb_is_active = enable_wandb and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None
    if wandb_is_active:
        try:
            effective_watch_freq = config_diffusion.get("wandb_watch_freq", 200) # Частота логирования градиентов
            sys.modules['wandb'].watch(diffusion_model, log="gradients", log_freq=effective_watch_freq, log_graph=False)
        except Exception as e_watch:
            logging.warning(f"Could not setup wandb.watch: {e_watch}")

    # Переменные для отслеживания лучшей модели и пути сохранения
    best_val_loss_diffusion = float('inf') # Инициализируем лучшую ошибку как бесконечность
    diffusion_save_path = config_diffusion.get("save_path", "./diffusion_best_model.pth") # Путь для сохранения модели
    save_dir_diff_model = os.path.dirname(diffusion_save_path)
    if not save_dir_diff_model: save_dir_diff_model = "."
    os.makedirs(save_dir_diff_model, exist_ok=True) # Создаем директорию, если ее нет

    logging.info("Starting Diffusion Model Training...")
    training_successful = False # Флаг успешности обучения
    force_retrain = config_diffusion.get("force_retrain", False) # Принудительно переобучать, даже если есть чекпоинт

    # Если не указано принудительное переобучение и есть сохраненная модель, загружаем ее
    if not force_retrain and os.path.exists(diffusion_save_path):
        logging.info(f"Found existing model at {diffusion_save_path} and force_retrain is False. Loading model.")
        try:
            diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=True)) 
            logging.info(f"Successfully loaded model from {diffusion_save_path}.")
            training_successful = True # Считаем успешным, если загрузили
        except RuntimeError: 
            logging.warning(f"Failed to load model with weights_only=True from {diffusion_save_path}. Retrying with weights_only=False.")
            try:
                diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=False))
                logging.info(f"Successfully loaded model (weights_only=False) from {diffusion_save_path}.")
                training_successful = True
            except Exception as e_load_fallback: # Если и так не удалось
                logging.error(f"Failed to load existing model from {diffusion_save_path}: {e_load_fallback}. Proceeding with training.")
                force_retrain = True # Принудительно переобучаем
        except Exception as e_load: # Другие ошибки загрузки
            logging.error(f"Failed to load existing model from {diffusion_save_path}: {e_load}. Proceeding with training.")
            force_retrain = True
    
    # Запускаем цикл обучения, если нужно переобучать или не удалось загрузить модель
    if force_retrain or not training_successful :
        logging.info("Proceeding with diffusion model training loop.")
        try:
            for epoch in range(config_diffusion["num_epochs"]): # Итерация по эпохам
                # Шаг обучения
                avg_train_loss = train_diffusion_epoch(diffusion_model, train_dataloader_diffusion, optimizer, scheduler, loss_fn_diffusion, config_diffusion, diffusion_variables_local, device, epoch, wandb_is_active)
                logging.info(f"Epoch {epoch+1}/{config_diffusion['num_epochs']} Avg Diffusion Training Loss: {avg_train_loss:.6f}")

                # Шаг валидации
                avg_val_loss = evaluate_diffusion_loss(diffusion_model, val_dataloader_diffusion, loss_fn_diffusion, config_diffusion, diffusion_variables_local, device, epoch, wandb_is_active)
                logging.info(f"Epoch {epoch+1}/{config_diffusion['num_epochs']} Avg Diffusion Validation Loss: {avg_val_loss:.6f}")

                # Логирование результатов эпохи в W&B
                if wandb_is_active:
                    try:
                        sys.modules['wandb'].log({
                            "diffusion_epoch": epoch + 1, 
                            "diffusion_avg_train_loss": avg_train_loss, 
                            "diffusion_avg_val_loss": avg_val_loss,
                            "diffusion_current_lr": scheduler.get_last_lr()[0]
                        }, step=(epoch + 1) * num_update_steps_per_epoch) # Логируем по шагам оптимизатора
                    except Exception as e_wandb_log_epoch:
                        logging.warning(f"Could not log epoch results to W&B: {e_wandb_log_epoch}")

                # Сохраняем лучшую модель по валидационной ошибке
                if avg_val_loss < best_val_loss_diffusion:
                    best_val_loss_diffusion = avg_val_loss
                    torch.save(diffusion_model.state_dict(), diffusion_save_path)
                    logging.info(f"Saved best diffusion model to {diffusion_save_path} (Val Loss: {best_val_loss_diffusion:.6f})")
                    # Обновляем конфигурацию W&B информацией о лучшей модели
                    if wandb_is_active:
                        try:
                            sys.modules['wandb'].config.update({
                                "best_diffusion_checkpoint": diffusion_save_path, 
                                "best_diffusion_val_loss": best_val_loss_diffusion, 
                                "best_diffusion_epoch": epoch + 1 
                            }, allow_val_change=True)
                        except Exception as e_wandb_conf:
                            logging.warning(f"Could not update W&B config with best model info: {e_wandb_conf}")
            training_successful = True # Обучение завершено успешно
        except KeyboardInterrupt: # Если прервано пользователем
            logging.warning("Training interrupted by user (KeyboardInterrupt).")
        except Exception as e_train_diff: # Если ошибка во время обучения
            logging.error(f"ERROR during Diffusion Training: {e_train_diff}")
            traceback.print_exc()
            # Пытаемся загрузить лучший сохраненный чекпоинт, если обучение прервалось с ошибкой
            if os.path.exists(diffusion_save_path):
                logging.info(f"Attempting to load best diffusion model from {diffusion_save_path} due to training error.")
                try:
                    diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=True))
                    training_successful = True # Загрузили чекпоинт
                except RuntimeError:
                    logging.warning(f"Fallback: Failed to load with weights_only=True. Retrying with weights_only=False.")
                    try:
                        diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=False))
                        training_successful = True
                    except Exception as e_load_fb:
                        logging.error(f"CRITICAL FALLBACK: Failed to load model even with weights_only=False after error: {e_load_fb}")
                        training_successful = False
                except Exception as e_load_err:
                    logging.error(f"CRITICAL: Failed to load model checkpoint after error: {e_load_err}")
                    training_successful = False
            else: # Если чекпоинта нет
                logging.error("No best diffusion model checkpoint found after training error.")
                training_successful = False
    
    logging.info("Diffusion training process finished or was skipped/interrupted.")

    # Если обучение не было успешным, но есть чекпоинт, пытаемся его загрузить для оценки
    if not training_successful:
        if not os.path.exists(diffusion_save_path): # Если нет чекпоинта, выходим
            logging.error(f"Diffusion model training failed and no checkpoint at '{diffusion_save_path}'. Skipping further diffusion steps.")
            return
        else: # Если чекпоинт есть
            logging.warning(f"Training was not fully completed or marked successful, but a checkpoint exists at '{diffusion_save_path}'. Attempting to load it for evaluation.")
            try:
                diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=True))
                logging.info(f"Successfully loaded model from {diffusion_save_path} for evaluation.")
            except RuntimeError: # Пробуем без weights_only=True
                logging.warning(f"Eval Load: Failed weights_only=True. Retrying weights_only=False.")
                try:
                    diffusion_model.load_state_dict(torch.load(diffusion_save_path, map_location=device, weights_only=False))
                    logging.info(f"Successfully loaded model (weights_only=False) from {diffusion_save_path} for evaluation.")
                except Exception as e_load_eval_fb: # Если и так не удалось
                    logging.error(f"CRITICAL EVAL LOAD: Failed to load from '{diffusion_save_path}' for evaluation: {e_load_eval_fb}. Cannot proceed with diffusion eval.")
                    return
            except Exception as e_load_eval: 
                logging.error(f"CRITICAL EVAL LOAD: Failed to load from '{diffusion_save_path}' for evaluation: {e_load_eval}. Cannot proceed with diffusion eval.")
                return
    
    diffusion_model.eval() # Переключаем модель в режим оценки для последующих шагов

    # Загрузка метрики ROUGE для оценки диффузионной модели
    rouge_metric_diff = None
    try:
        rouge_script_path_diff = config_diffusion.get("rouge_script_path", "rouge") # Путь к скрипту ROUGE
        trust_remote_rouge_diff = config_diffusion.get("rouge_trust_remote", False)
        rouge_metric_diff = evaluate.load(rouge_script_path_diff, trust_remote_code=trust_remote_rouge_diff)
        logging.info(f"ROUGE metric script loaded from '{rouge_script_path_diff}' for diffusion.")
    except ModuleNotFoundError: # Если библиотека evaluate не найдена
        logging.error("evaluate library not found. Please install it. Skipping ROUGE for diffusion.")
    except Exception as e_load_rouge: # Другие ошибки загрузки ROUGE
        logging.error(f"Failed to load ROUGE metric script for diffusion: {e_load_rouge}. Skipping ROUGE.")

    # Оценка ROUGE на тестовом наборе, если метрика загружена и есть данные
    if rouge_metric_diff is not None and hasattr(test_dataloader_diffusion, 'dataset') and test_dataloader_diffusion.dataset is not None and len(test_dataloader_diffusion.dataset) > 0:
        logging.info("\nEvaluating Diffusion Model on Test Set (ROUGE)...")
        rouge_eval_samples_diff = config_diffusion.get("rouge_eval_samples_diffusion", 100) # Количество примеров для ROUGE
        num_samples_to_eval_rouge = min(rouge_eval_samples_diff, len(test_dataloader_diffusion.dataset)) # Не больше, чем в датасете
        logging.info(f"Calculating ROUGE on up to {num_samples_to_eval_rouge} test samples for diffusion...")
        
        diffusion_rouge_results = evaluate_diffusion_rouge(diffusion_model, test_dataloader_diffusion, diffusion_tokenizer, config_diffusion, diffusion_variables_local, device, rouge_metric_diff, num_samples=num_samples_to_eval_rouge)
        
        logging.info("\nDiffusion ROUGE Results on Test Set:")
        if diffusion_rouge_results and "num_evaluated" in diffusion_rouge_results and diffusion_rouge_results["num_evaluated"] > 0:
            for key, value in diffusion_rouge_results.items(): # Выводим результаты ROUGE
                if isinstance(value, (int, float)): logging.info(f"  {key}: {value:.4f}")
                else: logging.info(f"  {key}: {value}")
            
            # Логирование результатов ROUGE в W&B
            if wandb_is_active:
                try:
                    log_data_test_diff = {f"diffusion_test_{k.replace('rouge', 'rouge_')}": v 
                                          for k, v in diffusion_rouge_results.items() 
                                          if 'rouge' in k.lower() or 'num_evaluated' in k or 'gen_len'in k}
                    sys.modules['wandb'].log(log_data_test_diff)
                except Exception as e_wandb_log_diff_test:
                    logging.warning(f"Could not log diffusion ROUGE to W&B: {e_wandb_log_diff_test}")
        else:
            logging.warning("Diffusion ROUGE results could not be computed or evaluation yielded no samples.")
    else: 
        logging.warning("Skipping ROUGE evaluation for diffusion (no metric, no data, or dataset empty).")

    # Генерация примера саммари для диффузионной модели
    logging.info("\nGenerating Diffusion example summary...")
    generate_example(
        model=diffusion_model, tokenizer=diffusion_tokenizer, 
        config_model_specific=config_diffusion, test_ds=test_ds,
        device=device, model_type="Diffusion", 
        enable_wandb=wandb_is_active, wandb_log_key="diffusion_generation_example",
        diffusion_vars=diffusion_variables_local,
        baseline_comparison_config=baseline_comparison_config_for_generate_example, # Конфиг baseline для сравнения
        baseline_model_path_for_comp=baseline_model_path_for_comp # Путь к модели baseline для сравнения
    )

    # Логирование финальной модели как артефакта в W&B
    if (training_successful or os.path.exists(diffusion_save_path)) and wandb_is_active: # Если обучение успешно или есть чекпоинт
        if os.path.exists(diffusion_save_path): # Убеждаемся, что файл модели существует
            try:
                logging.info(f"Logging final diffusion model from {diffusion_save_path} as W&B artifact.")
                run_id_for_artifact = sys.modules['wandb'].run.id if sys.modules['wandb'].run else "unknown_run"
                desc_val_loss = f"Best val loss: {best_val_loss_diffusion:.4f}" if best_val_loss_diffusion != float('inf') else "Loaded model (val loss not from this run)"
                diffusion_artifact_final = sys.modules['wandb'].Artifact(f"diffusion-model-final-{run_id_for_artifact}", type="model", 
                                                                        description=f"Final diffusion model. {desc_val_loss}")
                diffusion_artifact_final.add_file(diffusion_save_path) # Добавляем файл модели в артефакт
                sys.modules['wandb'].log_artifact(diffusion_artifact_final) # Логируем артефакт
            except Exception as e_art_final:
                logging.warning(f"Could not log final diffusion model artifact to W&B: {e_art_final}")
        else: # Если файла модели нет
            logging.warning(f"Diffusion model artifact logging skipped: file {diffusion_save_path} not found.")
            
    logging.info("run_diffusion function finished.")


def main(config_path):
    """Основная функция, управляющая всем процессом."""
    logging.info("main() function called.")
    try: # Загрузка конфигурации
        config = load_config(config_path)
    except Exception as e_conf: # Если не удалось загрузить конфиг, завершаем программу
        logging.error(f"CRITICAL: Failed to load config from {config_path}. Error: {e_conf}")
        return

    # Настройка устройства (CPU/GPU) и зерна случайности (seed)
    logging.info("Setting up DEVICE and SEED...")
    cuda_is_actually_available = False 
    if torch.cuda.is_available(): # Проверяем, доступна ли CUDA
        try:
            device_count = torch.cuda.device_count() # Количество GPU
            if device_count > 0:
                _ = torch.tensor([1.0]).cuda() # Простая операция на CUDA для проверки работоспособности
                cuda_is_actually_available = True
        except Exception as e_cuda_init_main: 
            logging.warning(f"CUDA is reported as available but failed functional test in main(): {e_cuda_init_main}. Defaulting to CPU.")

    requested_device = config.get("device", "cuda") # Запрошенное устройство из конфига
    DEVICE = torch.device("cuda" if requested_device == "cuda" and cuda_is_actually_available else "cpu") 
    if requested_device == "cuda" and not cuda_is_actually_available: 
        logging.warning("CUDA was requested in config but is not available/functional, using CPU.")
    logging.info(f"Using device: {DEVICE}")

    SEED = config.get("seed", 42) # Зерно случайности из конфига
    random.seed(SEED) 
    np.random.seed(SEED) 
    torch.manual_seed(SEED) 
    if DEVICE.type == 'cuda': 
        torch.cuda.manual_seed_all(SEED) 
    logging.info(f"Seed set to {SEED}")

    # Настройка Weights & Biases для логирования экспериментов
    ENABLE_WANDB = config.get("enable_wandb", False) 
    if 'wandb' in sys.modules: # Очищаем предыдущее состояние W&B, если оно было
        sys.modules['wandb'].run = None

    if ENABLE_WANDB:
        try:
            import wandb # Импортируем wandb только если он нужен
            sys.modules['wandb'] = wandb # Делаем wandb доступным глобально

            if os.environ.get("WANDB_API_KEY"): 
                wandb.init( 
                    project=config.get("wandb_project", "diffusion_summarization"), 
                    name=config.get("wandb_run_name", f"run-{SEED}-{DEVICE.type}"), 
                    config=config # Логируем всю конфигурацию
                )
                logging.info(f"Weights & Biases initialized. Run ID: {wandb.run.id if wandb.run else 'N/A'}")
            else: # Если ключа нет, отключаем W&B
                logging.warning("WANDB_API_KEY not found in environment. Disabling W&B to avoid login prompts. Set WANDB_API_KEY or login manually.")
                ENABLE_WANDB = False
                sys.modules['wandb'].run = None
        except ImportError: 
            logging.warning("wandb library not found, but ENABLE_WANDB is True. Disabling W&B for this run.")
            ENABLE_WANDB = False
            if 'wandb' in sys.modules: sys.modules['wandb'].run = None
        except Exception as e_wandb: 
            logging.error(f"Could not initialize W&B: {e_wandb}. Disabling W&B for this run.")
            ENABLE_WANDB = False
            if 'wandb' in sys.modules: sys.modules['wandb'].run = None

    # Загрузка датасета
    logging.info("Loading dataset...")
    train_ds, val_ds, test_ds = None, None, None # Инициализируем датасеты
    try:
        dataset_path = config["dataset_name"] # Имя датасета из конфига
        logging.info(f"Loading dataset '{dataset_path}'...")
        trust_remote_flag = config.get("dataset_trust_remote_code", True) 
        dataset_config_name = config.get("dataset_config_name", None) 

        # Загружаем полные сплиты датасета
        full_train_ds = load_dataset(dataset_path, name=dataset_config_name, split='train', trust_remote_code=trust_remote_flag)
        full_val_ds = load_dataset(dataset_path, name=dataset_config_name, split='validation', trust_remote_code=trust_remote_flag)
        full_test_ds = load_dataset(dataset_path, name=dataset_config_name, split='test', trust_remote_code=trust_remote_flag)

        # Получаем размеры выборок из конфига 
        train_size = config.get("train_size", len(full_train_ds))
        val_size = config.get("val_size", len(full_val_ds))
        test_size = config.get("test_size", len(full_test_ds))

        # Выбираем подмножества датасетов нужного размера
        train_ds = full_train_ds.select(range(min(train_size, len(full_train_ds))))
        val_ds = full_val_ds.select(range(min(val_size, len(full_val_ds))))
        test_ds = full_test_ds.select(range(min(test_size, len(full_test_ds))))

        logging.info("Dataset loaded and sliced successfully.")
        logging.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")
    except Exception as e_dataset: # Если ошибка при загрузке датасета
        logging.error(f"Error loading dataset '{config.get('dataset_name', 'N/A')}': {e_dataset}")
        traceback.print_exc()
        if ENABLE_WANDB and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
            sys.modules['wandb'].finish(exit_code=1) # Завершаем W&B run с ошибкой
        sys.exit(1) # Завершаем программу

    # Проверка, что датасеты не пустые
    if not all([train_ds, val_ds, test_ds]) or not all(len(ds) > 0 for ds in [train_ds, val_ds, test_ds]):
        logging.error("One or more dataset splits are empty after slicing or loading. Cannot proceed.")
        if ENABLE_WANDB and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
            sys.modules['wandb'].finish(exit_code=1)
        sys.exit(1)

    # Запуск baseline модели, если указано в конфиге
    baseline_model_path_for_diffusion_comparison = None # Путь к baseline модели для сравнения
    if config.get("run_baseline_first", False) and "baseline" in config:
        logging.info("Running Baseline Model as per 'run_baseline_first' setting...")
        baseline_model, baseline_results = run_baseline(
            config_baseline=config["baseline"], train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            device=DEVICE, enable_wandb=ENABLE_WANDB
        )
        # Сохраняем путь к baseline модели, если она была успешно обучена/сохранена
        baseline_save_path_from_config = config.get("baseline", {}).get("save_path")
        if baseline_model and baseline_save_path_from_config and os.path.exists(baseline_save_path_from_config) and \
           os.path.exists(os.path.join(baseline_save_path_from_config, "config.json")): # Проверяем наличие файла конфига модели
            baseline_model_path_for_diffusion_comparison = baseline_save_path_from_config
            logging.info(f"Baseline model path for comparison with diffusion set to: {baseline_model_path_for_diffusion_comparison}")
        else:
            logging.warning(f"Baseline model training/saving might have failed or path '{baseline_save_path_from_config}' is invalid/incomplete. No baseline model available for diffusion comparison.")
    else: # Если baseline не запускается
        logging.info("Skipping baseline execution as 'run_baseline_first' is False or 'baseline' config is missing.")


    # Запуск диффузионной модели
    if "diffusion" in config:
        logging.info("Starting Diffusion Model process...")
        run_diffusion(
            config_diffusion=config["diffusion"],
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds, device=DEVICE,
            enable_wandb=ENABLE_WANDB,
            baseline_model_path_for_comp=baseline_model_path_for_diffusion_comparison, # Путь к baseline для сравнения
            baseline_comparison_config_for_generate_example=config.get("baseline",{}) # Конфиг baseline для генерации примера
        )
        logging.info("Diffusion Model process finished.")
    else: 
        logging.info("No 'diffusion' configuration found. Skipping diffusion model run.")

    # Завершаем W&B run, если он был активен
    if ENABLE_WANDB and 'wandb' in sys.modules and getattr(sys.modules['wandb'], 'run', None) is not None:
        sys.modules['wandb'].finish()
        logging.info("W&B run finished.")
    
    logging.info("Project finished successfully.")


if __name__ == "__main__": # Точка входа в скрипт
    logging.info("Script execution started.")
    # Парсинг аргументов командной строки (путь к файлу конфигурации)
    parser = argparse.ArgumentParser(description="Train and evaluate summarization models.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    
    main(args.config) # Запускаем основную функцию
    logging.info("Script execution completed.")