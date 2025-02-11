import pandas as pd
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling 
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.preprocessing import LabelEncoder

model_checkpoint = "./Phi-3.5-mini-instruct"

# デバイスマップの設定

num_gpus = torch.cuda.device_count()
print(f"使用可能な GPU 数: {num_gpus}")
# 各 GPU の情報を表示
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

device_map = {
    "model.embed_tokens": "cuda:0",  # 最初のレイヤーを GPU 0 に割り当てる
}

for i in range(0, 10):
    device_map[f"model.layers.{i}"] = "cuda:0"
for i in range(10, 40):
    device_map[f"model.layers.{i}"] = "cuda:1"

device_map["model.norm.weight"] = "cuda:1"
device_map["lm_head.weight"] = "cuda:1"

# 結果を表示して確認
for layer, device in device_map.items():
    print(f"{layer}: {device}")

model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    load_in_8bit=True, 
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map=device_map,  # 手動でデバイスを指定
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# LoRA設定
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # ランク
    lora_alpha=32,
#    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    target_modules="all-linear",
    lora_dropout=0.1,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# トレーニング対象のパラメータを確認（ここで実行するのが推奨される位置）
print(f"Training target: ")
model.print_trainable_parameters()

# データセットの準備
df = pd.read_json('merged_sharegpt.json')
dataset = Dataset.from_pandas(df)

# ラベル列の一覧を取得
labels = dataset['label']
unique_labels = set(labels)   # 集合に変換して重複を除去
print(f"Unique labels in the dataset: {unique_labels}")

# すべてのラベルを含むリストを作成する
all_labels = list(unique_labels) # 集合からリストに変換

# LabelEncoder を初期化し、全てのラベルでフィットさせる
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)    # ラベルを数値に変換

def tokenize_function(examples):
    texts = []
    for conversation in examples["conversations"]:
        prompt = ""
        # 各ターンを走査して、話者ごとに明示的なラベルを付与
        for turn in conversation:
            sender = turn.get("from", "").strip().lower()
            message = turn.get("value", "").strip()
            if sender == "human":
                prompt += "<|user|>" + "\n" + message + "<|end|>\n"
            elif sender == "gpt":
                prompt += "<|assistant|>" + "\n" + message + "<|end|>\n"
            elif sender == "system":
                # システムメッセージは文脈の補足として扱う（必要に応じて）
                prompt += "<|system|>" + "\n" + message + "<|end|>\n"
            else:
                # 万一、想定外の送信者があった場合
                prompt += message + "<|end|>\n"
        print(f"prompt:\n{prompt}")
        texts.append(prompt)

    tokenized_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=512)

#    # ラベルの処理（分類タスクの場合）
#    if "label" in examples and examples["label"] is not None:
#        try:
#            labels = label_encoder.transform(examples["label"])
#        except ValueError:
#            labels = [0] * len(texts)  # ラベル変換ができなかった場合のダミー値
#    else:
#        labels = [0] * len(texts)
#
#    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# データセットの形式を設定
split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

#train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# TrainingArgumentsの設定
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # 各GPUでのバッチサイズ
    num_train_epochs=3,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    fp16=True,  # FP16を使用してメモリ使用量を減らす
)

# Trainerの初期化（tokenizerの代わりにprocessing_classを使用）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # 言語モデルの学習では MLM (Masked Language Modeling) を使わない
)

trainer = Trainer(
    model=model,    # LoRAが適用されたモデル
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # 追加
)

# モデルのデバイスを確認
for name, param in model.named_parameters():
    print(f"{name}: {param.device}")

# 訓練の実行
trainer.train()

# トレーニング後のモデルとLoRAパラメータを保存
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# LoRAパラメータをベースモデルにマージ
model = model.merge_and_unload()
model.save_pretrained("./fine_tuned_model_merged")
tokenizer.save_pretrained("./fine_tuned_model_merged")
