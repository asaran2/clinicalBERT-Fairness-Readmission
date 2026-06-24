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

# VRAM guide (based on ~48MB/sample observed on A100):
#   T4  16GB (free tier) → 300
#   V100 16GB (Pro)      → 300
#   A100 40GB (Pro+)     → 700
#   A100 80GB (Pro+)     → 1000
LIME_NUM_SAMPLES = 1000
INFERENCE_BATCH_SIZE = 1000  # 300 fits T4 (16GB); raise to 1000 on A100 80GB
# BERT only reads ~350 words (512 subword tokens). Pre-truncate so LIME perturbs
# a shorter string — saves perturbation generation AND tokenization work.
MAX_WORDS_FOR_LIME = 350

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded")

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained('./model/exp1_discharge', 1)
model.to(device)
model.eval()

# FP16: halves VRAM, runs faster on tensor cores. Safe for inference.
if device.type == 'cuda':
    model = model.half()

print("Model loaded and set to eval mode")

m = nn.Sigmoid()


def _encode_with_cache(words, word_cache):
    """Build BERT input from a list of whitespace-split words using a subtoken cache."""
    subtokens = ['[CLS]']
    for word in words:
        subtokens.extend(word_cache[word])
        if len(subtokens) >= 511:
            subtokens = subtokens[:511]
            break
    subtokens.append('[SEP]')
    input_ids = tokenizer.convert_tokens_to_ids(subtokens)
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
    # All LIME perturbations are subsets of the same source text, so they share
    # the same ~350 unique words. Build a word→subtoken cache once instead of
    # calling tokenizer.tokenize() ~175,000 times (once per word per perturbation).
    word_cache = {}
    for text in texts:
        for word in text.split():
            if word not in word_cache:
                word_cache[word] = tokenizer.tokenize(word)

    all_probs = np.empty((len(texts), 2), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
            batch_words = [texts[j].split() for j in range(i, min(i + INFERENCE_BATCH_SIZE, len(texts)))]
            encoded = [_encode_with_cache(words, word_cache) for words in batch_words]

            ids  = torch.tensor([e[0] for e in encoded], dtype=torch.long).to(device, non_blocking=True)
            seg  = torch.tensor([e[1] for e in encoded], dtype=torch.long).to(device, non_blocking=True)
            mask = torch.tensor([e[2] for e in encoded], dtype=torch.long).to(device, non_blocking=True)

            logits = model(ids, seg, mask)
            probs = m(logits.float()).squeeze(-1).cpu().numpy()

            all_probs[i : i + len(probs), 1] = probs
            all_probs[i : i + len(probs), 0] = 1.0 - probs

    return all_probs


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

        # Truncate to MAX_WORDS_FOR_LIME words. BERT only sees ~350 words anyway,
        # so LIME perturbing text beyond that is wasted string + tokenization work.
        truncated_text = ' '.join(str(chunk_text).split()[:MAX_WORDS_FOR_LIME])

        try:
            explanation = explainer.explain_instance(
                truncated_text,
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