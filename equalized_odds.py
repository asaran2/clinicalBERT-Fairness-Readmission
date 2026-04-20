import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_rates(y_true, y_pred):
    """Compute TPR and FPR from true and predicted labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr, tn, fp, fn, tp


def main():
    df = pd.read_csv("result_discharge/output.csv")

    # Extract gender from clinical text
    df["gender"] = df["TEXT"].str.extract(r"(?i)sex:\s+([mf])", expand=False).str.upper()
    df = df.dropna(subset=["gender"])

    y_true = df["OUTPUT_LABEL"].astype(int)
    y_pred = df["pred_label"].astype(int)

    # Overall metrics
    tpr_all, fpr_all, tn, fp, fn, tp = compute_rates(y_true, y_pred)
    print("=" * 50)
    print("OVERALL")
    print(f"  Total samples: {len(df)}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  TPR = {tpr_all:.4f}")
    print(f"  FPR = {fpr_all:.4f}")

    # Per-gender metrics
    results = {}
    for gender in ["M", "F"]:
        mask = df["gender"] == gender
        tpr, fpr, tn, fp, fn, tp = compute_rates(y_true[mask], y_pred[mask])
        results[gender] = {"TPR": tpr, "FPR": fpr}
        label = "Male" if gender == "M" else "Female"
        print(f"\n{label} (n={mask.sum()})")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  TPR = {tpr:.4f}")
        print(f"  FPR = {fpr:.4f}")

    # Equalized Odds comparison
    tpr_diff = abs(results["M"]["TPR"] - results["F"]["TPR"])
    fpr_diff = abs(results["M"]["FPR"] - results["F"]["FPR"])

    print("\n" + "=" * 50)
    print("EQUALIZED ODDS ANALYSIS")
    print(f"  TPR difference (|Male - Female|): {tpr_diff:.4f}")
    print(f"  FPR difference (|Male - Female|): {fpr_diff:.4f}")

    if tpr_diff < 0.05 and fpr_diff < 0.05:
        print("  --> Model approximately satisfies Equalized Odds (both gaps < 0.05)")
    else:
        print("  --> Model does NOT satisfy Equalized Odds")
        if tpr_diff >= 0.05:
            print(f"      TPR gap ({tpr_diff:.4f}) >= 0.05 threshold")
        if fpr_diff >= 0.05:
            print(f"      FPR gap ({fpr_diff:.4f}) >= 0.05 threshold")

    # Recall Parity analysis
    # Recall = TP / (TP + FN), same as TPR
    recall_m = results["M"]["TPR"]
    recall_f = results["F"]["TPR"]
    recall_diff = abs(recall_m - recall_f)

    print("\n" + "=" * 50)
    print("RECALL PARITY ANALYSIS")
    print(f"  Recall (Male):   {recall_m:.4f}")
    print(f"  Recall (Female): {recall_f:.4f}")
    print(f"  Recall difference (|Male - Female|): {recall_diff:.4f}")

    if recall_diff < 0.05:
        print("  --> Model approximately satisfies Recall Parity (gap < 0.05)")
    else:
        print("  --> Model does NOT satisfy Recall Parity")
        print(f"      Recall gap ({recall_diff:.4f}) >= 0.05 threshold")


if __name__ == "__main__":
    main()
