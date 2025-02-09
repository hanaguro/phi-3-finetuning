#!/usr/bin/env python
import glob
import os
import shutil
import torch.onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TUNED_DIR = "./fine_tuned_model_merged"
ONNX_PATH = "./fine_tuned_model.onnx"

# モデルとトークナイザのロード
loaded_model = AutoModelForSequenceClassification.from_pretrained(TUNED_DIR)
loaded_tokenizer = AutoTokenizer.from_pretrained(TUNED_DIR)

loaded_model.config.use_cache = False
loaded_model.config.return_dict = False  # （任意）出力をタプルにする

# サンプル入力の作成
dummy_input = loaded_tokenizer(["Hello, world!", "How are you?"], return_tensors="pt", padding=True, truncation=True)

# ONNX形式にエクスポート
torch.onnx.export(
    loaded_model,
    dummy_input["input_ids"],
    ONNX_PATH,
    input_names=["input_ids"],
    output_names=["output"],
    dynamic_axes={"input_ids": {0: "batch_size"}, "output": {0: "batch_size"}}
)

# トークナイザの保存
loaded_tokenizer.save_pretrained(TUNED_DIR)

TOKENIZER_DIR = TUNED_DIR + "/tokenizer"

os.makedirs(TOKENIZER_DIR, exist_ok=True)
shutil.move(ONNX_PATH, TUNED_DIR + "/model.onnx")

for item in os.listdir(TUNED_DIR):
    src_path = os.path.join(TUNED_DIR, item)
    
    # 現在のアイテムがディレクトリである場合に再帰的にコピー
    if os.path.isdir(src_path):
        if item == "tokenizer":
            continue
        shutil.copytree(src_path, os.path.join(TOKENIZER_DIR, item), dirs_exist_ok=True)
    else:
        # ファイルや既存のディレクトリ（tokenizer）をシンプルにコピー
        shutil.copy(src_path, TOKENIZER_DIR)

with open(TUNED_DIR + "/Modelfile", "w") as f:
    absolute_path = os.path.abspath(TUNED_DIR)
    text = f"""
FROM phi4
FROM {absolute_path}
# 創造性を高めるためtemperatureを1に設定
PARAMETER temperature 1
"""
    f.write(text)
