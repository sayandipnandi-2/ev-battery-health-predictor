import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

print("Libraries loaded")

from pathlib import Path

# Paths relative to this script, so it works no matter where you run it from
BASE = Path(__file__).parent
DATA = DATA = BASE / "dataset" / "cleaned_dataset"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

meta = pd.read_csv(DATA / "metadata.csv")
print("Shape:", meta.shape)
print("Columns:", meta.columns.tolist())
print(meta.head(10))
print(meta["type"].value_counts())
print(meta["battery_id"].nunique(), "batteries")

# ---------- STEP 4: filter + clean + SOH ----------
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY = 2.0  # Ah

df = meta[(meta["battery_id"].isin(BATTERIES)) & (meta["type"] == "discharge")].copy()

# Capacity may be stored as text, so convert to numbers; bad values become NaN
df["Capacity"] = pd.to_numeric(df["Capacity"], errors="coerce")
print("\nMissing capacity values:", df["Capacity"].isna().sum())
df = df.dropna(subset=["Capacity"])

# Order by time within each battery, then number the cycles 1, 2, 3...
df = df.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
df["cycle"] = df.groupby("battery_id").cumcount() + 1

# State of Health in percent
df["SOH"] = df["Capacity"] / RATED_CAPACITY * 100

print("\nRows per battery:")
print(df["battery_id"].value_counts())
print("\nCapacity and SOH statistics per battery:")
print(df.groupby("battery_id")[["Capacity", "SOH"]].describe().T)
print("\nFirst rows:")
print(df[["battery_id", "cycle", "Capacity", "SOH"]].head())

# ---------- STEP 5: health labels, cleaned CSV, first charts ----------
def label_health(soh):
    if soh >= 80:
        return "Healthy"
    elif soh >= 70:
        return "Degrading"
    return "Critical"

df["health"] = df["SOH"].apply(label_health)

print("\nHealth label counts:")
print(df["health"].value_counts())
print("\nLabels per battery:")
print(pd.crosstab(df["battery_id"], df["health"]))

# Cleaned dataset: drop columns that are empty or messy for discharge rows
clean = df.drop(columns=["start_time", "Re", "Rct"])
clean.to_csv(OUT / "cleaned_battery_data.csv", index=False)
clean[["Capacity", "SOH"]].describe().to_csv(OUT / "basic_statistics.csv")

# Chart 1: SOH vs cycle
plt.figure(figsize=(9, 5))
for b, g in df.groupby("battery_id"):
    plt.plot(g["cycle"], g["SOH"], label=b)
plt.axhline(80, color="green", linestyle="--", label="Healthy threshold (80%)")
plt.axhline(70, color="red", linestyle="--", label="Critical threshold (70%)")
plt.xlabel("Cycle number")
plt.ylabel("SOH (%)")
plt.title("Battery SOH vs Cycle Number")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT / "chart_soh_vs_cycle.png", dpi=150, bbox_inches="tight")
plt.close()

# Chart 2: health status distribution
order = ["Healthy", "Degrading", "Critical"]
counts = df["health"].value_counts().reindex(order)
plt.figure(figsize=(6, 4))
plt.bar(order, counts.values, color=["green", "orange", "red"])
plt.xlabel("Health status")
plt.ylabel("Number of cycles")
plt.title("Health Status Distribution")
plt.savefig(OUT / "chart_health_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSaved files:", [p.name for p in OUT.iterdir()])

# ---------- STEP 6: feature extraction from per-cycle files ----------
DATA_DIR = DATA / "data"

def extract_features(filename):
    d = pd.read_csv(DATA_DIR / filename)
    v = d["Voltage_measured"]
    t = d["Temperature_measured"]
    i = d["Current_measured"]
    time = d["Time"]
    below = d.loc[v <= 3.5, "Time"]          # time when voltage first falls to 3.5 V
    return pd.Series({
        "discharge_time": time.max(),        # seconds until the discharge ended
        "v_mean": v.mean(),
        "v_min": v.min(),
        "v_start": v.iloc[0],
        "temp_mean": t.mean(),
        "temp_max": t.max(),
        "current_mean": i.mean(),
        "time_to_3_5V": below.iloc[0] if len(below) else time.max(),
    })

features = df["filename"].apply(extract_features)
df = pd.concat([df, features], axis=1)

feature_cols = ["cycle", "discharge_time", "v_mean", "v_min", "v_start",
                "temp_mean", "temp_max", "current_mean", "time_to_3_5V"]

print("\nMissing values in features:", df[feature_cols].isna().sum().sum())
print("\nFeature statistics:")
print(df[feature_cols].describe().T)
print("\nCorrelation with SOH:")
print(df[feature_cols + ["SOH"]].corr()["SOH"].sort_values())

# Re-save the cleaned dataset, now including the features
clean = df.drop(columns=["start_time", "Re", "Rct"])
clean.to_csv(OUT / "cleaned_battery_data.csv", index=False)

# Chart 3: mean discharge voltage vs cycle
plt.figure(figsize=(9, 5))
for b, g in df.groupby("battery_id"):
    plt.plot(g["cycle"], g["v_mean"], label=b)
plt.xlabel("Cycle number")
plt.ylabel("Mean discharge voltage (V)")
plt.title("Mean Discharge Voltage vs Cycle Number")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT / "chart_voltage_vs_cycle.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved:", [p.name for p in OUT.iterdir()])

# ---------- STEP 7: train and evaluate ----------
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df[feature_cols]
y = df["health"]
classes = ["Healthy", "Degrading", "Critical"]

def make_model():
    return RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")

# --- Main test: train on B0005, B0006, B0007 -> test on unseen battery B0018 ---
test_battery = "B0018"
is_test = df["battery_id"] == test_battery
model = make_model().fit(X[~is_test], y[~is_test])
pred = model.predict(X[is_test])

acc = accuracy_score(y[is_test], pred)
print(f"\nMAIN TEST (train on other 3 batteries, test on {test_battery})")
print(f"Accuracy: {acc:.3f}")
print(classification_report(y[is_test], pred, labels=classes, zero_division=0))

cm = confusion_matrix(y[is_test], pred, labels=classes)
print("Confusion matrix (rows = actual, columns = predicted):")
print(pd.DataFrame(cm, index=classes, columns=classes))

ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title(f"Confusion Matrix (test battery {test_battery})")
plt.savefig(OUT / "confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Robustness check: leave one battery out, for every battery ---
print("\nLeave-one-battery-out accuracy:")
for b in BATTERIES:
    m = df["battery_id"] == b
    p = make_model().fit(X[~m], y[~m]).predict(X[m])
    print(f"  {b}: {accuracy_score(y[m], p):.3f}")

# --- Comparison: random split (likely optimistic) ---
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rand_acc = accuracy_score(yte, make_model().fit(Xtr, ytr).predict(Xte))
print(f"\nRandom 80/20 split accuracy: {rand_acc:.3f}  (optimistic: neighbouring cycles leak between train and test)")

# --- Chart 4: which features matter most ---
imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
plt.figure(figsize=(7, 4))
imp.plot(kind="barh", color="steelblue")
plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.savefig(OUT / "chart_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nFeature importance:")
print(imp.sort_values(ascending=False))

# ---------- STEP 8: leakage check + refined features ----------
def extra_features(filename):
    d = pd.read_csv(DATA_DIR / filename)
    v, t = d["Voltage_measured"], d["Time"]
    def time_below(th):
        s = d.loc[v <= th, "Time"]
        return s.iloc[0] if len(s) else t.max()
    return pd.Series({
        "time_to_3_8V": time_below(3.8),
        "time_to_3_6V": time_below(3.6),
        "time_to_3_4V": time_below(3.4),
    })

df = pd.concat([df, df["filename"].apply(extra_features)], axis=1)

refined_cols = ["time_to_3_8V", "time_to_3_6V", "time_to_3_5V", "time_to_3_4V",
                "v_start", "temp_mean", "temp_max"]

def loo_eval(cols, name):
    accs = {}
    for b in BATTERIES:
        m = df["battery_id"] == b
        p = make_model().fit(df.loc[~m, cols], y[~m]).predict(df.loc[m, cols])
        accs[b] = accuracy_score(y[m], p)
    print(f"\n{name}: mean accuracy {np.mean(list(accs.values())):.3f}")
    for b, a in accs.items():
        print(f"  {b}: {a:.3f}")

loo_eval(feature_cols, "ORIGINAL features (with cycle/current/full discharge)")
loo_eval(refined_cols, "REFINED features (partial-curve only)")

# Detailed result for the refined model on B0018
m = df["battery_id"] == "B0018"
ref_model = make_model().fit(df.loc[~m, refined_cols], y[~m])
pred = ref_model.predict(df.loc[m, refined_cols])
print("\nREFINED model on unseen battery B0018:")
print(classification_report(y[m], pred, labels=classes, zero_division=0))
cm = confusion_matrix(y[m], pred, labels=classes)
print(pd.DataFrame(cm, index=classes, columns=classes))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - refined features (test battery B0018)")
plt.savefig(OUT / "confusion_matrix_refined.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nRefined feature importance:")
print(pd.Series(ref_model.feature_importances_, index=refined_cols).sort_values(ascending=False))

# ---------- STEP 9: baseline-normalised features ----------
base_cols = ["time_to_3_8V", "time_to_3_6V", "time_to_3_5V", "time_to_3_4V",
             "temp_mean", "temp_max"]
norm_cols = []
for c in base_cols:
    baseline = df[df["cycle"] <= 5].groupby("battery_id")[c].mean()
    df[c + "_rel"] = df[c] / df["battery_id"].map(baseline)
    norm_cols.append(c + "_rel")

def loo_pooled(cols):
    preds = pd.Series(index=df.index, dtype=object)
    for b in BATTERIES:
        m = df["battery_id"] == b
        preds[m] = make_model().fit(df.loc[~m, cols], y[~m]).predict(df.loc[m, cols])
    return preds

loo_eval(norm_cols, "BASELINE-NORMALISED features")

# Pooled leave-one-battery-out results (every cycle predicted by a model that never saw its battery)
pooled = loo_pooled(norm_cols)
print("\nPOOLED leave-one-battery-out results (all 636 cycles):")
print(f"Accuracy: {accuracy_score(y, pooled):.3f}")
print(classification_report(y, pooled, labels=classes, zero_division=0))
cm = confusion_matrix(y, pooled, labels=classes)
print(pd.DataFrame(cm, index=classes, columns=classes))

ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - leave-one-battery-out (normalised features)")
plt.savefig(OUT / "confusion_matrix_normalised.png", dpi=150, bbox_inches="tight")
plt.close()

full_model = make_model().fit(df[norm_cols], y)
print("\nFeature importance:")
print(pd.Series(full_model.feature_importances_, index=norm_cols).sort_values(ascending=False))

# ---------- STEP 10: final model ----------
import joblib

final_cols = refined_cols

def loo_mean(cols):
    accs = []
    for b in BATTERIES:
        m = df["battery_id"] == b
        p = make_model().fit(df.loc[~m, cols], y[~m]).predict(df.loc[m, cols])
        accs.append(accuracy_score(y[m], p))
    return np.mean(accs)

# Results table for the report
results = pd.DataFrame({
    "Feature set": ["Original (incl. cycle/current/full discharge)",
                    "Refined (partial-curve) - FINAL",
                    "Baseline-normalised",
                    "Random 80/20 split (optimistic)"],
    "Accuracy": [loo_mean(feature_cols), loo_mean(final_cols),
                 loo_mean(norm_cols), rand_acc],
    "Evaluation": ["Leave-one-battery-out"] * 3 + ["Random split"],
}).round(3)
print("\nResults summary:")
print(results.to_string(index=False))
results.to_csv(OUT / "results_summary.csv", index=False)

# Final evaluation: every cycle predicted by a model that never saw its battery
final_pred = loo_pooled(final_cols)
print(f"\nFINAL MODEL - pooled leave-one-battery-out accuracy: {accuracy_score(y, final_pred):.3f}")
print(classification_report(y, final_pred, labels=classes, zero_division=0))
cm = confusion_matrix(y, final_pred, labels=classes)
print(pd.DataFrame(cm, index=classes, columns=classes))

ConfusionMatrixDisplay(cm, display_labels=classes).plot(cmap="Blues", values_format="d")
plt.title("Final Model - Confusion Matrix\n(leave-one-battery-out, all 636 cycles)")
plt.savefig(OUT / "confusion_matrix_final.png", dpi=150, bbox_inches="tight")
plt.close()

# Train the final model on all data and save it
final_model = make_model().fit(df[final_cols], y)
joblib.dump(final_model, OUT / "battery_health_model.joblib")

# Save the final cleaned dataset with the final features
final_df = df[["battery_id", "cycle", "Capacity", "SOH", "health"] + final_cols]
final_df.to_csv(OUT / "cleaned_battery_data.csv", index=False)
print("\nSaved:", sorted(p.name for p in OUT.iterdir()))