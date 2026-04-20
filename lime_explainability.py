import torch
import numpy as np
import pandas as pd
from torch import nn
from pytorch_pretrained_bert.tokenization import BertTokenizer
from modeling_readmission import BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded")

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained('./model/discharge_readmission', 1)
model.to(device)
model.eval()
print("Model loaded and set to eval mode")

m = nn.Sigmoid()


def predictor_function(texts):
    """
    Convert text between LIME and BERT
    Input: list of strings from LIME
    Output: numpy array shape (n, 2) with [P(no readmit), P(readmit)]
    """
    all_probs = []
    with torch.no_grad():
        for text in texts:
            # Tokenize (same as convert_examples_to_features lines 162-176)
            tokens = tokenizer.tokenize(text)[:510]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            
            # Pad to 512 (same as lines 220-223)
            pad_len = 512 - len(input_ids)
            input_ids += [0] * pad_len
            segment_ids += [0] * pad_len
            input_mask += [0] * pad_len
            
            # Convert to tensors (same as lines 633-636, but one example)
            ids = torch.tensor([input_ids]).to(device)
            seg = torch.tensor([segment_ids]).to(device)
            mask = torch.tensor([input_mask]).to(device)
            
            # Forward pass (same as line 656)
            logit = model(ids, seg, mask)
            
            # Sigmoid (same as line 658)
            p1 = m(logit).item()
            
            # LIME needs 2 columns: [P(class 0), P(class 1)]
            all_probs.append([1 - p1, p1])
    
    return np.array(all_probs)

explainer = LimeTextExplainer(class_names=['No Readmit', 'Readmit'])
print(predictor_function(["patient has CHF and requires dialysis"]))
# Load test chunks
explanation = explainer.explain_instance(
    "patient has CHF and requires dialysis",
    predictor_function,
    num_features=20,
    num_samples=1000
)
print(explanation.as_list())
chunks_df = pd.read_csv('./data/discharge/test.csv')

for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="LIME explanations"):
    chunk_text = row['TEXT']
    
    explanation = explainer.explain_instance(
        "chunk_text",
        predictor_function,
        num_features=20,
        num_samples=1000
    )
    
    weights = dict(explanation.as_list())