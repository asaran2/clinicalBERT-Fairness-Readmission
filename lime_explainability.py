import torch
import numpy as np
import pandas as pd
import glob
import os
from torch import nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from modeling_readmission import BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tune INFERENCE_BATCH_SIZE down if you hit OOM; 64 fits comfortably on A100 80GB.
INFERENCE_BATCH_SIZE = 64
# 500 LIME samples gives near-identical top features at half the cost.
LIME_NUM_SAMPLES = 500

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded")

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained('./model/discharge_readmission', 1)
model.to(device)
model.eval()

# FP16 halves VRAM usage and runs faster on tensor cores; safe for inference.
if device.type == 'cuda':
    model = model.half()

print("Model loaded and set to eval mode")

m = nn.Sigmoid()


def _encode(text):
    tokens = tokenizer.tokenize(text)[:510]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_len = 512 - len(input_ids)
    input_ids  += [0] * pad_len
    input_mask  = [1] * (512 - pad_len) + [0] * pad_len
    segment_ids = [0] * 512
    return input_ids, segment_ids, input_mask


def predictor_function(texts):
    """
    Batched BERT predictor for LIME.
    Input: list of strings (all LIME perturbations at once)
    Output: numpy array shape (n, 2) with [P(no readmit), P(readmit)]
    """
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
            batch = texts[i : i + INFERENCE_BATCH_SIZE]
            encoded = [_encode(t) for t in batch]

            ids  = torch.tensor([e[0] for e in encoded], dtype=torch.long).to(device)
            seg  = torch.tensor([e[1] for e in encoded], dtype=torch.long).to(device)
            mask = torch.tensor([e[2] for e in encoded], dtype=torch.long).to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                logits = model(ids, seg, mask)

            probs = m(logits.float()).squeeze(-1).cpu().numpy()
            for p1 in probs:
                all_probs.append([1.0 - float(p1), float(p1)])

    return np.array(all_probs)


explainer = LimeTextExplainer(class_names=['No Readmit', 'Readmit'])
print(predictor_function(["patient has CHF and requires dialysis"]))

# Smoke-test
explanation = explainer.explain_instance(
    "patient has CHF and requires dialysis",
    predictor_function,
    num_features=20,
    num_samples=LIME_NUM_SAMPLES,
)
print(explanation.as_list())

inputDir  = "/content/drive/MyDrive/NarrativeGuard/Final_Code_Dataset/dataSplit_1_1000_chunked"
outputDir = "/content/drive/MyDrive/NarrativeGuard/Final_Code_Dataset/lime_1_1000"
csv_files = glob.glob(os.path.join(inputDir, "*.csv"))

os.makedirs(outputDir, exist_ok=True)

for file_path in csv_files:
    chunks_df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path)
    output_path = os.path.join(outputDir, f"processed_{file_name}")
    results = []

    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc=f"LIME: {file_name}"):
        chunk_text = row['mimic_text']

        if pd.isna(chunk_text) or str(chunk_text).strip() == '':
            continue

        try:
            explanation = explainer.explain_instance(
                str(chunk_text),
                predictor_function,
                num_features=20,
                num_samples=LIME_NUM_SAMPLES,
            )
            p_readmit = explanation.predict_proba[1]

            results.append({
                'chunk_idx': idx,
                'hadm_id': row['hadm_id'],
                'true_label': row['true_label'],
                'bert_prob_readmit': round(p_readmit, 4),
                'bert_prediction': 1 if p_readmit >= 0.5 else 0,
                'weights': str(explanation.as_list()),
            })
        except Exception as e:
            print(f"Error on chunk {idx}: {e}")
            continue

        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Saved {len(results)} results to {output_path}")