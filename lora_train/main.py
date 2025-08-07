# === НАСТРОЙКИ ===
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
DATASET_PATH = "train_dataset.jsonl"
PERMANENT_OUTPUT_DIR = "lora_out"
READY_LORA_DIR = "ready_lora"
LOG_DIR = "training_logs"
DATASET_CACHE_DIR = "dataset_cache"

# Доля данных для валидации и частота логов/оценок
VALIDATION_SPLIT = 0.03
VALIDATION_STEPS = 50
GENERATION_EVAL_STEPS = 50
NUM_GENERATION_SAMPLES = 5

# LoRA настройки
LORA_RANK = 32
LORA_ALPHA = 64
USE_DORA = False 

# Tokenization
MAX_SEQ_LENGTH = 328
USE_RAMDISK = False
RAMDISK_PATH = "/dev/shm"

# === ИМПОРТЫ ===
from unsloth import FastLanguageModel
import os, json, torch, tempfile, shutil, random, signal, atexit, sys, hashlib
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from datasets import Dataset
from rouge import Rouge
from transformers import TrainingArguments, pipeline, AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from trl import SFTTrainer

torch.manual_seed(42)

# === КЛАСС ДЛЯ ЛОГИРОВАНИЯ ВЫВОДА ===
class Tee:
    def __init__(self, *files):
        self.files = files
        
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
            
    def flush(self):
        for f in self.files:
            f.flush()

# === КАЛБЭК ДЛЯ ОЧИСТКИ ПАМЯТИ ===
class MemoryCleanerCallback(TrainerCallback):
    def __init__(self, clean_before_eval=True, clean_before_generation=True):
        self.clean_before_eval = clean_before_eval
        self.clean_before_generation = clean_before_generation
        
    def on_evaluate(self, args, state, control, **kwargs):
        if self.clean_before_eval:
            self._clean_memory("перед валидацией")
    
    def on_generation(self, args, state, control, **kwargs):
        if self.clean_before_generation:
            self._clean_memory("перед генерацией")
    
    def _clean_memory(self, context):
        print(f"\n🧹 Очистка памяти {context}...")
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            print("✅ CUDA-память очищена")
        except Exception as e:
            print(f"⚠️ Ошибка очистки памяти: {e}")

# === КАЛБЭК ДЛЯ ЛОГИРОВАНИЯ МЕТРИК ===
class MetricsCallback(TrainerCallback):
    def __init__(self, log_file):
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        logs["step"] = state.global_step
        
        if "eval_loss" in logs:
            logs["step_type"] = "validation"
        else:
            logs["step_type"] = "training"
            
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

# === КАЛБЭК ДЛЯ ОЦЕНКИ ГЕНЕРАЦИИ ===
class GenerationEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, eval_dataset, output_dir, model, gen_kwargs=None):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.rouge = Rouge()
        self.rouge_scores = defaultdict(list)
        os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "rouge_scores"), exist_ok=True)
        
        # Создаем pipeline для генерации
        self.pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        
        # Параметры генерации
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            #"length_penalty": 0.6,
            "eos_token_id": tokenizer.eos_token_id,
            "early_stopping": True,
        }
        
    def extract_response_from_output(self, output):
        """Извлекает ответ из вывода pipeline"""
        for i in range(len(output) - 1, -1, -1):
            if output[i]["role"] == "assistant":
                return output[i]["content"]
        return ""
        
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % GENERATION_EVAL_STEPS == 0:
            self.run_generation_eval(state.global_step)
    
    def run_generation_eval(self, step):
        print("\n🚀 Запуск оценки генерации...")
        
        sample_indices = random.sample(range(len(self.eval_dataset)), 
                                     min(NUM_GENERATION_SAMPLES, len(self.eval_dataset)))
        samples = [self.eval_dataset[i] for i in sample_indices]
        
        generations = []
        
        for i, sample in enumerate(samples):
            try:
                messages = sample["messages"]
                
                prompt_messages = []
                reference = ""
                for msg in messages:
                    if msg["role"] in ["system", "user"]:
                        prompt_messages.append(msg)
                    elif msg["role"] == "assistant":
                        reference = msg["content"]
                
                output = self.pipe(prompt_messages, **self.gen_kwargs)[0]["generated_text"]
                generated_response = self.extract_response_from_output(output)
                
                try:
                    if generated_response and reference:
                        scores = self.rouge.get_scores(generated_response, reference)[0]
                    else:
                        scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
                    
                    for metric, values in scores.items():
                        self.rouge_scores[metric].append(values['f'])
                except Exception as e:
                    print(f"⚠️ Ошибка расчета ROUGE: {e}")
                    scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
                
                generations.append({
                    "step": step,
                    "prompt": json.dumps(prompt_messages, ensure_ascii=False),
                    "reference": reference,
                    "generated": generated_response,
                    "rouge": scores
                })
            except Exception as e:
                print(f"⚠️ Ошибка обработки примера {i}: {e}")
                generations.append({"step": step, "error": str(e)})
        
        generation_file = os.path.join(self.output_dir, "generations", f"step-{step}.json")
        with open(generation_file, "w", encoding="utf-8") as f:
            json.dump(generations, f, indent=2, ensure_ascii=False)
        
        avg_rouge = {metric: sum(scores)/len(scores) 
                     for metric, scores in self.rouge_scores.items()} if self.rouge_scores else \
                    {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
        
        rouge_file = os.path.join(self.output_dir, "rouge_scores", f"step-{step}.json")
        with open(rouge_file, "w", encoding="utf-8") as f:
            json.dump({
                "step": step,
                "avg_rouge": avg_rouge,
                "rouge_scores": dict(self.rouge_scores)
            }, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Результаты оценки генерации (шаг {step}):")
        print(f"  ROUGE-1 F1: {avg_rouge.get('rouge-1', 0):.4f}")
        print(f"  ROUGE-2 F1: {avg_rouge.get('rouge-2', 0):.4f}")
        print(f"  ROUGE-L F1: {avg_rouge.get('rouge-l', 0):.4f}")

# === ОЧИСТКА ПАМЯТИ ===
def cleanup():
    print("\nОчистка ресурсов...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✅ CUDA-память очищена.")
    except Exception as e:
        print(f"⚠️ Ошибка при очистке: {e}")
    
    global TEMP_DIR, USE_RAMDISK
    if USE_RAMDISK and TEMP_DIR and os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"✅ Очищен RAM-диск: {TEMP_DIR}")
        except Exception as e:
            print(f"⚠️ Ошибка очистки RAM-диска: {e}")

# === ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ХЕША КОНФИГУРАЦИИ ===
def generate_config_hash(tokenizer_config, dataset_path):
    """Генерирует уникальный хеш для конфигурации токенизатора и датасета"""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(tokenizer_config, sort_keys=True).encode("utf-8"))
    with open(dataset_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

# === ОБНОВЛЕННАЯ ФУНКЦИЯ ДЛЯ ЗАГРУЗКИ ДАТАСЕТА С КЭШИРОВАНИЕМ ===
def load_and_format_dataset(tokenizer, dataset_path, cache_dir="dataset_cache", validation_split=0.0):
    """Загружает и форматирует датасет с использованием кэша и разделением"""
    cache_key = generate_config_hash({
        "chat_template": tokenizer.chat_template,
        "max_length": tokenizer.model_max_length,
        "validation_split": validation_split,
        "version": 3
    }, dataset_path)
    
    train_cache = os.path.join(cache_dir, f"{cache_key}_train")
    valid_cache = os.path.join(cache_dir, f"{cache_key}_valid")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Пытаемся загрузить из кэша
    if validation_split > 0 and os.path.exists(train_cache) and os.path.exists(valid_cache):
        print(f"✅ Загрузка форматированных датасетов из кэша")
        return {
            "train": Dataset.load_from_disk(train_cache),
            "validation": Dataset.load_from_disk(valid_cache)
        }
    elif os.path.exists(train_cache):
        print(f"✅ Загрузка тренировочного датасета из кэша")
        return {"train": Dataset.load_from_disk(train_cache)}
    
    print("🔍 Загрузка и форматирование датасета...")
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                if "messages" in data and isinstance(data["messages"], list) and len(data["messages"]) >= 2:
                    examples.append(data)
                else:
                    print(f"⚠️ Пропущена строка {i+1}: неверная структура данных")
            except json.JSONDecodeError as e:
                print(f"Ошибка JSON в строке {i+1}: {e}")
    
    if not examples:
        raise ValueError("Датасет пуст! Проверьте файл train_dataset.jsonl")
    
    print(f"✅ Загружено {len(examples)} примеров")
    
    # Сортировка по длине токенов
    print("🔁 Сортировка примеров по длине токенов...")
    def token_length(ex):
        return len(tokenizer.apply_chat_template(ex["messages"], tokenize=True))
    examples.sort(key=token_length)
    
    formatted_data = []
    error_count = 0
    
    for i, example in enumerate(tqdm(examples, desc="Форматирование")):
        try:
            messages = example["messages"]
            assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
            reference = assistant_messages[-1]["content"] if assistant_messages else ""
            
            formatted_data.append({
                "text": tokenizer.apply_chat_template(messages, tokenize=False),
                "reference": reference,
                "messages": messages,
            })
        except Exception as e:
            error_count += 1
            print(f"Ошибка форматирования в примере {i+1}: {e}")
    
    print(f"✅ Успешно отформатировано: {len(formatted_data)} примеров")
    print(f"⚠️ Ошибки форматирования: {error_count}")
    
    if not formatted_data:
        raise ValueError("Нет валидных примеров для обучения!")
    
    # Разделение на train/validation
    result = {}
    if validation_split > 0 and len(formatted_data) > 10:
        split_index = int(len(formatted_data) * (1 - validation_split))
        train_data = formatted_data[:split_index]
        valid_data = formatted_data[split_index:]
        
        print(f"📊 Разделение датасета: train={len(train_data)} validation={len(valid_data)}")
        
        train_dataset = Dataset.from_list(train_data)
        valid_dataset = Dataset.from_list(valid_data)
        
        train_dataset.save_to_disk(train_cache)
        valid_dataset.save_to_disk(valid_cache)
        result = {
            "train": train_dataset,
            "validation": valid_dataset
        }
    else:
        print("ℹ️ Валидационный набор не создан (мало данных или split=0)")
        train_dataset = Dataset.from_list(formatted_data)
        train_dataset.save_to_disk(train_cache)
        result = {"train": train_dataset}
    
    return result


# === ФУНКЦИЯ ДЛЯ ОЦЕНКИ ГЕНЕРАЦИИ ДО ОБУЧЕНИЯ ===
def run_initial_generation_eval(model, tokenizer, eval_dataset, output_dir, num_samples=10):
    """Запускает оценку генерации до начала обучения"""
    print("\n🚀 Запуск начальной оценки генерации (шаг 0)...")
    
    # Создаем pipeline для генерации
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    # Параметры генерации
    gen_kwargs = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
    }
    
    # Выбираем случайные примеры
    sample_indices = random.sample(range(len(eval_dataset)), 
                                 min(num_samples, len(eval_dataset)))
    samples = [eval_dataset[i] for i in sample_indices]
    
    generations = []
    rouge_scores = defaultdict(list)
    rouge = Rouge()
    
    for i, sample in enumerate(tqdm(samples, desc="Начальная генерация")):
        try:
            # Парсим JSON из текста
            messages = sample["messages"]
            
            # Извлекаем промпт (система + пользователь)
            prompt_messages = []
            reference = ""
            for msg in messages:
                if msg["role"] in ["system", "user"]:
                    prompt_messages.append(msg)
                elif msg["role"] == "assistant":
                    reference = msg["content"]
            
            # Генерация ответа с помощью pipeline
            output = pipe(
                prompt_messages,
                **gen_kwargs
            )[0]["generated_text"]
            
            # Извлекаем сгенерированный ответ
            generated_response = ""
            for j in range(len(output) - 1, -1, -1):
                if output[j]["role"] == "assistant":
                    generated_response = output[j]["content"]
                    break
            
            # Расчет метрик ROUGE
            try:
                if generated_response and reference:
                    scores = rouge.get_scores(generated_response, reference)[0]
                else:
                    scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
                
                for metric, values in scores.items():
                    rouge_scores[metric].append(values['f'])
            except Exception as e:
                print(f"⚠️ Ошибка расчета ROUGE: {e}")
                scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
            
            # Сохранение примера
            generations.append({
                "step": 0,
                "prompt": json.dumps(prompt_messages, ensure_ascii=False),
                "reference": reference,
                "generated": generated_response,
                "rouge": scores
            })
        except Exception as e:
            print(f"⚠️ Ошибка обработки примера {i}: {e}")
            generations.append({
                "step": 0,
                "error": str(e)
            })
    
    # Создаем директории, если их нет
    os.makedirs(os.path.join(output_dir, "generations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rouge_scores"), exist_ok=True)
    
    # Сохранение примеров генерации
    generation_file = os.path.join(output_dir, "generations", "step-0.json")
    with open(generation_file, "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)
    
    # Расчет и сохранение средних метрик ROUGE
    avg_rouge = {metric: sum(scores)/len(scores) 
                for metric, scores in rouge_scores.items()} if rouge_scores else {}
    
    rouge_file = os.path.join(output_dir, "rouge_scores", "step-0.json")
    with open(rouge_file, "w", encoding="utf-8") as f:
        json.dump({
            "step": 0,
            "avg_rouge": avg_rouge,
            "rouge_scores": dict(rouge_scores)
        }, f, indent=2, ensure_ascii=False)
    
    # Логирование результатов
    print(f"📊 Результаты начальной оценки генерации (шаг 0):")
    print(f"  ROUGE-1 F1: {avg_rouge.get('rouge-1', 0):.4f}")
    print(f"  ROUGE-2 F1: {avg_rouge.get('rouge-2', 0):.4f}")
    print(f"  ROUGE-L F1: {avg_rouge.get('rouge-l', 0):.4f}")
    print(f"💾 Результаты сохранены в:\n  {generation_file}\n  {rouge_file}")

# === ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ ===
def inspect_dataset_samples(dataset, tokenizer, num_samples=3):
    print("\n🧪 Визуальная проверка отформатированных данных (те, что реально идут в обучение):")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i in indices:
        try:
            sample_text = dataset[i]["text"]
            print(f"\n--- Пример {i} ---\n{sample_text}\n{'-'*60}")
        except Exception as e:
            print(f"⚠️ Ошибка отображения примера {i}: {e}")


# === ОСНОВНОЙ КОД ===
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = open(os.path.join(LOG_DIR, "console_output.log"), "w", encoding="utf-8")
    metrics_log = os.path.join(LOG_DIR, "training_metrics.jsonl")
    
    # Перенаправляем вывод
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    print(f"=== ЗАПУСК ОБУЧЕНИЯ ===")
    print(f"Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Модель: {MODEL_PATH}")
    print(f"Датасет: {DATASET_PATH}")
    print(f"Размер валидации: {VALIDATION_SPLIT*100}%")
    print(f"Шаг валидации: каждые {VALIDATION_STEPS} шагов")
    print(f"Шаг оценки генерации: каждые {GENERATION_EVAL_STEPS} шагов")

    # === RAM-ДИСК ===
    if USE_RAMDISK and os.path.exists(RAMDISK_PATH):
        ram_free = shutil.disk_usage(RAMDISK_PATH).free
        ds_size = os.path.getsize(DATASET_PATH)
        if ds_size < ram_free * 0.7:
            TEMP_DIR = tempfile.mkdtemp(dir=RAMDISK_PATH)
            ram_path = os.path.join(TEMP_DIR, os.path.basename(DATASET_PATH))
            shutil.copy2(DATASET_PATH, ram_path)
            DATASET_PATH = ram_path
            print(f"✅ Датасет скопирован в RAM-диск: {ram_path}")
        else:
            print("⚠️ Недостаточно места в RAM-диске")

    # === ЗАГРУЗКА МОДЕЛИ ===
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        token=None,
        attn_implementation="xformers"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === ДАННЫЕ ===
    dataset = load_and_format_dataset(tokenizer, DATASET_PATH, cache_dir=DATASET_CACHE_DIR, validation_split=VALIDATION_SPLIT)
    train_data = dataset["train"]
    valid_data = dataset.get("validation")

    inspect_dataset_samples(train_data, tokenizer, num_samples=5)

    # === LoRA ===
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        use_dora=USE_DORA,
        lora_dropout=0.05,
        use_gradient_checkpointing=True,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    # Проверка чекпоинтов
    checkpoint = None
    if os.path.exists(PERMANENT_OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(PERMANENT_OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(PERMANENT_OUTPUT_DIR, checkpoints[-1])
            print(f"✅ Найден чекпоинт: {checkpoint}")

    first_gen = True
    if os.path.exists(PERMANENT_OUTPUT_DIR + "/generations/step-0.json"):
        first_gen = False

    # === ЗАПУСК ОЦЕНКИ ГЕНЕРАЦИИ ДО ОБУЧЕНИЯ ===
    if valid_data and first_gen:
        run_initial_generation_eval(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=valid_data,
            output_dir=PERMANENT_OUTPUT_DIR,
            num_samples=NUM_GENERATION_SAMPLES
        )

    # === ARGS ===
    training_args = TrainingArguments(
        output_dir=PERMANENT_OUTPUT_DIR,
        per_device_train_batch_size=18, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=3, 
        lr_scheduler_type = "cosine_with_restarts", 
        warmup_ratio = 0.03,
        learning_rate=5e-5,
        max_grad_norm=0.3,
        num_train_epochs=3,
        optim="adamw_torch_fused",
        logging_steps=5,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps" if valid_data else "no",
        eval_steps=VALIDATION_STEPS if valid_data else None,
        logging_dir=f"{PERMANENT_OUTPUT_DIR}/logs",
        bf16=True,
        dataloader_pin_memory=True,
        torch_compile=True
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        eval_dataset=valid_data,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )

    # === CALLBACKS ===
    trainer.add_callback(MemoryCleanerCallback())
    trainer.add_callback(MetricsCallback(metrics_log))

    if valid_data:
        trainer.add_callback(GenerationEvalCallback(
            tokenizer=tokenizer,
            eval_dataset=valid_data,
            output_dir=PERMANENT_OUTPUT_DIR,
            model=model,
            gen_kwargs={
                "max_new_tokens": 512,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "early_stopping": True,
                "eos_token_id": tokenizer.eos_token_id,
            }
        ))

    # === ОБУЧЕНИЕ ===
    if checkpoint:
        print(f"⏩ Возобновление обучения с чекпоинта: {checkpoint}")
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    trainer.save_model(READY_LORA_DIR)
    print(f"✅ Обучение завершено. Модель сохранена в {READY_LORA_DIR}")

except Exception as e:
    print(f"\n‼️ Критическая ошибка: {e}")
    import traceback
    traceback.print_exc()
    cleanup()
    sys.exit(1)
finally:
    # Восстанавливаем стандартные потоки вывода
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    if 'log_file' in locals():
        log_file.close()
