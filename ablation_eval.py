import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from model import HybridDetector
from train import DeepFakeDataset, BATCH_SIZE
import os
import json

CHECKPOINT_PATH = r"c:\Users\rheap\Desktop\DeepFake Multimedia Detection\checkpoints\best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_ablation(mode='full'):
    test_dataset = DeepFakeDataset('test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = HybridDetector(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for vid, aud, labels in test_loader:
            vid, aud = vid.to(DEVICE), aud.to(DEVICE)
            
            if mode == 'video_only':
                aud = torch.zeros_like(aud)
            elif mode == 'audio_only':
                vid = torch.zeros_like(vid)
                
            outputs = model(vid, aud)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
            
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    return acc, auc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    if not os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint not found.")
        exit(1)
        
    results = {}
    
    model = HybridDetector().to(DEVICE)
    params = count_parameters(model)
    results['parameters'] = params
    results['params_million'] = round(params / 1e6, 2)
    
    print(f"Total Parameters: {results['params_million']}M")
    
    for mode in ['full', 'video_only', 'audio_only']:
        print(f"Evaluating {mode}...")
        acc, auc = evaluate_ablation(mode)
        results[mode] = {'accuracy': round(acc * 100, 2), 'auc': round(auc, 4)}
        print(f"{mode.capitalize()} - Acc: {acc*100:.2f}%, AUC: {auc:.4f}")
        
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
