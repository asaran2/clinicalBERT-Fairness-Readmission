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
            tokens = tokenizer.tokenize(text)[:510]
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0] * len(input_ids)
            input_mask = [1] * len(input_ids)
            
            pad_len = 512 - len(input_ids)
            input_ids += [0] * pad_len
            segment_ids += [0] * pad_len
            input_mask += [0] * pad_len
            
            ids = torch.tensor([input_ids]).to(device)
            seg = torch.tensor([segment_ids]).to(device)
            mask = torch.tensor([input_mask]).to(device)
            
            logit = model(ids, seg, mask)
            
            p1 = m(logit).item()
            
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

inputDir = "/content/drive/MyDrive/NarrativeGuard/Final_Code_Dataset/dataSplit_1_1000_chunked"

outputDir = "/content/drive/MyDrive/NarrativeGuard/Final_Code_Dataset/lime_input_1_1000"
csv_files = glob.glob(os.path.join(inputDir, "*.csv"))

for file_path in csv_files:
    chunks_df = pd.read_csv(file_path)
    results = []
    for idx, row in tqdm(chunks_df.iterrows(), total=len(chunks_df), desc="LIME explanations"):
        chunk_text = row['TEXT']
        
        explanation = explainer.explain_instance(
            chunk_text,
            predictor_function,
            num_features=20,
            num_samples=1000
        )
        p_readmit = explanation.predict_proba[1]  # stored from LIME's initial call

        results.append({
            'chunk_idx': idx,
            'bert_prob_readmit': round(p_readmit, 4),
            'bert_prediction': 1 if p_readmit >= 0.5 else 0,
            'weights': str(explanation.as_list())
        })

    df_results = pd.DataFrame(results)
    file_name = os.path.basename(file_path)
    output_path = os.path.join(outputDir, f"processed_{file_name}")
    df_results.to_csv(output_path, index=False)