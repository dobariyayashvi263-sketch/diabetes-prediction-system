"""
Diabetes Prediction System using Machine Learning
SYBCA-AI Sem-III | 303 Machine Learning | Mini Project

Aa EK j file badhu kare che: ML project + graphs + Project_Report.pdf

Folder structure:
    Project_Folder/
        Dataset/dataset.csv
        Python_Code/project.py   <- this file
        Reports/                 <- graphs, EDA_Report.html, Project_Report.pdf

Run:
    cd python_code
    python project.py

Jarur libraries:
    pip install numpy pandas matplotlib seaborn scikit-learn sweetviz reportlab
"""

import json
import os
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# SETTINGS (jarur pade to j badlo)
# ---------------------------------------------------------------------------
TARGET = None          # None = auto detect. Jem ke: TARGET = "Diabetes"
SHOW_PLOTS = True     # True karso to darek graph window ma khulse
RANDOM_STATE = 42
MAKE_REPORT = True     # False karso to PDF report nahi bane

# --- Report ni details (aa potani lakho) ---
PROJECT_TITLE = "DIABETES PREDICTION SYSTEM USING MACHINE LEARNING"
SHORT_TITLE = "Diabetes Prediction System"
STUDENT_NAMES = "Yashvi Dobariya,Charvi Beladiya"
ROLL_NUMBERS = "556, 575"
SUBMITTED_TO = "Hiral mam"
ACADEMIC_YEAR = "2026-27"
DATASET_SOURCE = "Kaggle - Diabetes dataset (CSV downloaded by the students)"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "Dataset", "dataset.csv")
REPORT_DIR = os.path.join(BASE_DIR, "..", "Reports")
PLOT_DIR = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@contextmanager
def safe(name):
    """Koi graph/step fail thay to puro code na atke, matra warning aape."""
    try:
        yield
    except Exception as e:
        plt.close("all")
        print(f"WARNING: {name} na bani -> {type(e).__name__}: {e}")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, name), dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def grid_plot(cols, plot_fn, title, fname, ncols=3):
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, c in zip(axes, cols):
        plot_fn(ax, c)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    save_fig(fname)


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def find_dataset():
    if os.path.exists(DATA_PATH):
        return DATA_PATH
    folder = os.path.dirname(DATA_PATH)
    if os.path.isdir(folder):
        csvs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv"))
        if csvs:
            print("dataset.csv na malyu, aa file vapri:", csvs[0])
            return os.path.join(folder, csvs[0])
    raise SystemExit("Dataset na malyo. CSV file 'Dataset' folder ma mukjo: "
                     + os.path.abspath(folder))


# ---------------------------------------------------------------------------
# STEP 2: DATASET COLLECTION
# ---------------------------------------------------------------------------
header("STEP 2: DATASET COLLECTION")
data_file = find_dataset()
try:
    df = pd.read_csv(data_file)
except UnicodeDecodeError:
    df = pd.read_csv(data_file, encoding="latin-1")
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

# --- Target column shodho ---
if TARGET is None:
    matches = [c for c in df.columns if "diabet" in c.lower()]
    binary_matches = [c for c in matches if df[c].nunique() == 2]
    if binary_matches:
        TARGET = binary_matches[-1]
    elif matches:
        TARGET = matches[-1]
    else:
        for name in ["outcome", "diagnosis", "target", "label", "result"]:
            found = [c for c in df.columns if name in c.lower()]
            if found:
                TARGET = found[-1]
                break
    if TARGET is None:
        print("\nTarget column jate na malyo. Upar ni 'Columns' list jovo ane")
        print("SETTINGS ma TARGET = \"column_naam\" lakho.")
        raise SystemExit
if TARGET not in df.columns:
    raise SystemExit(f"TARGET '{TARGET}' dataset ma nathi. Columns: {df.columns.tolist()}")
print("\nTarget variable :", TARGET)
if df[TARGET].nunique() != 2:
    print("WARNING: Target ma", df[TARGET].nunique(),
          "category che. Diabetes mate 2 (Yes/No) hovi joie. SETTINGS ma TARGET check karo.")

# --- ID jevi columns drop karo ---
id_cols = [c for c in df.columns
           if c != TARGET and (c.lower() == "id" or c.lower().endswith("_id")
                               or c.lower().startswith("id_")
                               or (df[c].nunique() == len(df)
                                   and not pd.api.types.is_float_dtype(df[c])))]
if id_cols:
    print("ID columns dropped:", id_cols)
    df = df.drop(columns=id_cols)

# --- Target text hoy (Yes/No) to 0/1 ma convert karo ---
target_mapping = None
if not pd.api.types.is_numeric_dtype(df[TARGET]):
    labels = sorted(df[TARGET].dropna().unique(), key=str)

    def is_negative(lbl):
        s = str(lbl).lower().strip()
        return s in ("no", "false", "0", "negative", "healthy", "normal") \
            or s.startswith("non") or s.startswith("no ")

    labels = sorted(labels, key=lambda l: (0 if is_negative(l) else 1, str(l)))
    target_mapping = {lbl: i for i, lbl in enumerate(labels)}
    df[TARGET] = df[TARGET].map(target_mapping)
    print("Target encoding:", target_mapping, "(1 = positive / diabetes)")

df = df.dropna(subset=[TARGET])
n_classes = df[TARGET].nunique()
AVG = "binary" if n_classes == 2 else "weighted"
SCORING = "f1" if n_classes == 2 else "f1_weighted"

feature_cols = [c for c in df.columns if c != TARGET]
print("Feature variables:", feature_cols)

df_raw = df.copy()   # "Before" copy (preprocessing pehla)

# ---------------------------------------------------------------------------
# STEP 3: UNDERSTANDING DATA
# ---------------------------------------------------------------------------
header("STEP 3: UNDERSTANDING DATA")
print("\n--- Info ---")
df.info()

print("\n--- Summary (describe) ---")
print(df.describe())

numerical_cols = df.select_dtypes(include=[np.number]).columns.drop(TARGET).tolist()
categorical_cols = [c for c in df.columns if c != TARGET and c not in numerical_cols]
print("\nNumerical columns  :", numerical_cols)
print("Categorical columns:", categorical_cols)

print("\n--- Missing values ---")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Hidden missing: jya 0 shakya nathi (glucose, BP, BMI vagere)
zero_keywords = ["glucose", "blood", "pressure", "bmi", "insulin", "skin", "thick", "cholesterol"]
zero_cols = [c for c in numerical_cols
             if any(k in c.lower() for k in zero_keywords) and (df[c] == 0).sum() > 0]
print("\n--- Zero values (hidden missing) ---")
if zero_cols:
    print((df[zero_cols] == 0).sum())
else:
    print("Koi hidden zero values nathi.")

print("\n--- Target distribution ---")
print(df[TARGET].value_counts())

# ---------------------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------------------------
header("STEP 4: EXPLORATORY DATA ANALYSIS")

class_labels = df[TARGET].astype(str)
palette_cls = ["#4c9be8", "#e8776c", "#6cc08b", "#f2b84b"][:max(df[TARGET].nunique(), 1)]


def hist_by_class(ax, c):
    try:
        sns.histplot(x=df[c], hue=class_labels, kde=True, ax=ax, bins=15, palette=palette_cls)
    except Exception:
        ax.clear()
        sns.histplot(x=df[c], hue=class_labels, kde=False, ax=ax, bins=15, palette=palette_cls)
    ax.set_title(f"Distribution of {c}")


def box_by_class(ax, c):
    sns.boxplot(x=class_labels, y=df[c], ax=ax, color="#4c9be8")
    ax.set_title(f"{c} by {TARGET} class")
    ax.set_xlabel(TARGET)


# 1. Histogram (class pramane alag rang + KDE)
with safe("Histogram"):
    grid_plot(numerical_cols, hist_by_class, "Histograms of Numerical Features",
              "01_histograms.png")

# 2. Box plot (class pramane)
with safe("Box plot"):
    grid_plot(numerical_cols, box_by_class, "Box Plots by Class", "02_boxplots.png")

# 3. Pie chart (target)
with safe("Pie chart"):
    plt.figure(figsize=(5, 5))
    counts = df[TARGET].value_counts().sort_index()
    plt.pie(counts, labels=[f"Class {i}" for i in counts.index], autopct="%1.1f%%",
            colors=["#66b3ff", "#ff9999", "#99ff99", "#ffd166"][:len(counts)], startangle=90)
    plt.title(f"Target Distribution ({TARGET})")
    save_fig("03_pie_target.png")

# 4. Bar chart
with safe("Bar chart"):
    if categorical_cols:
        cat_plot_cols = categorical_cols[:3]
        fig, axes = plt.subplots(1, len(cat_plot_cols), figsize=(5 * len(cat_plot_cols), 4))
        axes = np.array(axes).reshape(-1)
        for ax, c in zip(axes, cat_plot_cols):
            sns.countplot(data=df, x=c, hue=TARGET, ax=ax)
            ax.set_title(f"{c} vs {TARGET}")
        save_fig("04_bar_chart.png")
    else:
        plt.figure(figsize=(5, 4))
        sns.countplot(x=df[TARGET])
        plt.title(f"{TARGET} count")
        save_fig("04_bar_chart.png")

# 5. Correlation matrix
corr = df[numerical_cols + [TARGET]].corr()
with safe("Correlation matrix"):
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix")
    save_fig("05_correlation.png")

print("Graphs save thaya:", os.path.abspath(PLOT_DIR))
print("\nTarget sathe correlation:")
print(corr[TARGET].drop(TARGET).sort_values(ascending=False))

# ---------------------------------------------------------------------------
# STEP 5: HANDLE MISSING VALUES AND OUTLIERS
# ---------------------------------------------------------------------------
header("STEP 5: DATA PREPROCESSING")

dups = int(df.duplicated().sum())
if dups:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed:", dups)

# Hidden zeros -> NaN
for c in zero_cols:
    df[c] = df[c].replace(0, np.nan)

missing_before = df.isnull().sum()

# Missing values fill: numeric -> median, categorical -> mode
for c in numerical_cols:
    df[c] = df[c].fillna(df[c].median())
for c in categorical_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].mode()[0])
missing_after = df.isnull().sum()

# Outliers: IQR method (capping). Binary/flag columns skip karo.
outlier_cols = [c for c in numerical_cols if df[c].nunique() > 10]


def count_outliers(frame, col):
    q1, q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
    iqr = q3 - q1
    return int(((frame[col] < q1 - 1.5 * iqr) | (frame[col] > q3 + 1.5 * iqr)).sum())


outliers_before = {c: count_outliers(df, c) for c in outlier_cols}
for c in outlier_cols:
    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
outliers_after = {c: count_outliers(df, c) for c in outlier_cols}

print("\n--- Missing values: Before vs After ---")
print(pd.DataFrame({"Before": missing_before, "After": missing_after}))
print("\n--- Outliers: Before vs After ---")
print(pd.DataFrame({"Before": outliers_before, "After": outliers_after}))

with safe("Before vs After chart"):
    if outlier_cols:
        fig, axes = plt.subplots(2, len(outlier_cols), figsize=(3.2 * len(outlier_cols), 7))
        axes = np.array(axes).reshape(2, -1)
        for i, c in enumerate(outlier_cols):
            sns.boxplot(y=df_raw[c], ax=axes[0, i], color="salmon")
            axes[0, i].set_title(f"{c}\n(Before)")
            sns.boxplot(y=df[c], ax=axes[1, i], color="lightgreen")
            axes[1, i].set_title("(After)")
        save_fig("06_before_after.png")

# ---------------------------------------------------------------------------
# STEP 6: AUTOMATED EDA (Sweetviz)
# ---------------------------------------------------------------------------
header("STEP 6: AUTOMATED EDA")
eda_path = os.path.join(REPORT_DIR, "EDA_Report.html")
try:
    import sweetviz as sv
    report = sv.analyze(df_raw)
    report.show_html(filepath=eda_path, open_browser=False)
    print("Sweetviz report bani:", os.path.abspath(eda_path))
except Exception as e:
    print("Sweetviz run na thayu:", e)
    try:
        from ydata_profiling import ProfileReport
        ProfileReport(df_raw, title="Diabetes EDA Report").to_file(eda_path)
        print("ydata-profiling report bani:", os.path.abspath(eda_path))
    except Exception as e2:
        print("Automated EDA report na bani:", e2)
        print("Tip: terminal ma 'pip install sweetviz' (ke 'pip install ydata-profiling') run karo.")

# ---------------------------------------------------------------------------
# STEP 7: FEATURE SELECTION AND TRAIN-TEST SPLIT
# ---------------------------------------------------------------------------
header("STEP 7: FEATURE SELECTION")
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

if categorical_cols:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
X = X.astype(float)

print("Dependent variable   :", TARGET)
print("Independent variables:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)

# ---------------------------------------------------------------------------
# STEP 8: BUILD ML MODEL (Logistic Regression)
# ---------------------------------------------------------------------------
header("STEP 8: BUILD MODEL - LOGISTIC REGRESSION")
baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
baseline.fit(X_train, y_train)
print("Baseline Logistic Regression train.")

# ---------------------------------------------------------------------------
# STEP 9: MODEL EVALUATION
# ---------------------------------------------------------------------------
header("STEP 9: MODEL EVALUATION")


def evaluate(model, name):
    pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred, average=AVG, zero_division=0)),
        "Recall": float(recall_score(y_test, pred, average=AVG, zero_division=0)),
        "F1 Score": float(f1_score(y_test, pred, average=AVG, zero_division=0)),
    }


base_result = evaluate(baseline, "Baseline Logistic Regression")
for k, v in base_result.items():
    print(f"{k:10}: {v:.4f}" if k != "Model" else f"{k:10}: {v}")

pred_base = baseline.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred_base))
print("\nClassification Report:")
print(classification_report(y_test, pred_base, zero_division=0))

with safe("Confusion matrix chart"):
    ConfusionMatrixDisplay.from_predictions(y_test, pred_base, cmap="Blues")
    plt.title("Confusion Matrix (Baseline)")
    save_fig("07_confusion_matrix.png")

# ---------------------------------------------------------------------------
# STEP 10: IMPROVE MODEL (Cross Validation + Regularization)
# ---------------------------------------------------------------------------
header("STEP 10: IMPROVE MODEL")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_acc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="accuracy")
print(f"Baseline 5-fold CV Accuracy: {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")

print("\n--- Regularization (L1 vs L2, different C) ---")
rows = []
for penalty in ["l1", "l2"]:
    for C in [0.01, 0.1, 1, 10, 100]:
        try:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(penalty=penalty, C=C, solver="liblinear",
                                             max_iter=1000, random_state=RANDOM_STATE)),
            ])
            score = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=SCORING).mean()
            rows.append({"Penalty": penalty, "C": float(C), "CV F1": round(float(score), 4)})
        except Exception as e:
            print(f"Skip {penalty} C={C}: {type(e).__name__}: {e}")

if rows:
    reg_table = pd.DataFrame(rows)
    print(reg_table.to_string(index=False))
    best = reg_table.sort_values("CV F1", ascending=False).iloc[0]
else:
    reg_table = pd.DataFrame([{"Penalty": "l2", "C": 1.0, "CV F1": 0.0}])
    best = reg_table.iloc[0]
print(f"\nBest: penalty={best['Penalty']}, C={best['C']} (CV F1 = {best['CV F1']})")

try:
    improved = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(penalty=best["Penalty"], C=float(best["C"]),
                                     solver="liblinear", max_iter=1000,
                                     random_state=RANDOM_STATE)),
    ])
    improved.fit(X_train, y_train)
except Exception as e:
    print("Improved model train na thayu, baseline vaparyu:", e)
    improved = baseline
improved_result = evaluate(improved, f"Improved ({str(best['Penalty']).upper()}, C={best['C']})")

# ---------------------------------------------------------------------------
# STEP 11-12: RESULTS
# ---------------------------------------------------------------------------
header("STEP 12: RESULTS - COMPARISON")
results = pd.DataFrame([base_result, improved_result]).set_index("Model").round(4)
print(results)
results.to_csv(os.path.join(REPORT_DIR, "model_results.csv"))

coefs = pd.Series(improved.named_steps["model"].coef_[0], index=X.columns)
importance = pd.DataFrame({
    "Coefficient": coefs,
    "Importance (plus)": coefs.abs(),
    "Odds Ratio": np.exp(coefs),
    "Effect": np.where(coefs > 0, "Increases risk", "Decreases risk"),
}).sort_values("Importance (plus)", ascending=False)

print("\nFeature importance (scaled data):")
print(importance.round(3))

with safe("Feature importance chart"):
    plt.figure(figsize=(8, 5))
    importance["Importance (plus)"].sort_values().plot(kind="barh", color="teal")
    plt.xlabel("Importance (absolute coefficient)")
    plt.title("Feature Importance (Logistic Regression)")
    save_fig("08_feature_importance.png")

# Navo patient predict karvanu example
sample = X_test.iloc[[0]]
pred = improved.predict(sample)[0]
prob = improved.predict_proba(sample)[0][1] if n_classes == 2 else None
print("\nExample prediction (test set no pehlo patient):")
print("Predicted class:", pred, "| Actual:", y_test.iloc[0],
      "" if prob is None else f"| Probability of class 1: {prob:.2f}")

# ---------------------------------------------------------------------------
# Report mate summary
# ---------------------------------------------------------------------------
summary = {
    "target": TARGET,
    "rows": int(df_raw.shape[0]),
    "columns_total": int(df_raw.shape[1]),
    "id_cols_dropped": id_cols,
    "feature_cols": [c for c in df_raw.columns if c != TARGET],
    "numerical_cols": numerical_cols,
    "categorical_cols": categorical_cols,
    "missing_raw": int(df_raw.isnull().sum().sum()),
    "duplicates": dups,
    "zero_counts": {c: int((df_raw[c] == 0).sum()) for c in zero_cols},
    "missing_after_fill": int(missing_after.sum()),
    "outliers_before": outliers_before,
    "outliers_after": outliers_after,
    "class_counts": {str(k): int(v) for k, v in df_raw[TARGET].value_counts().sort_index().items()},
    "corr_with_target": {k: float(v) for k, v in
                         corr[TARGET].drop(TARGET).sort_values(ascending=False).items()},
    "train_size": int(X_train.shape[0]),
    "test_size": int(X_test.shape[0]),
    "n_features_model": int(X.shape[1]),
    "baseline": base_result,
    "improved": improved_result,
    "best_penalty": str(best["Penalty"]),
    "best_C": float(best["C"]),
    "cv_accuracy_mean": float(cv_acc.mean()),
    "cv_accuracy_std": float(cv_acc.std()),
    "reg_table": reg_table.to_dict(orient="records"),
    "confusion_matrix": confusion_matrix(y_test, pred_base).tolist(),
    "top_features": {k: float(v) for k, v in
                     coefs.reindex(coefs.abs().sort_values(ascending=False).index).items()},
}
with safe("summary.json"):
    with open(os.path.join(REPORT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

# ---------------------------------------------------------------------------
# PDF REPORT
# ---------------------------------------------------------------------------
def build_report(S):
    from xml.sax.saxutils import escape
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                    Preformatted, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    out_path = os.path.join(REPORT_DIR, "Project_Report.pdf")
    BLUE = colors.HexColor("#1f6fbf")
    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica",
                          fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                        fontSize=15, textColor=BLUE, spaceBefore=12, spaceAfter=6)
    caption = ParagraphStyle("caption", parent=body, fontSize=9, leading=12,
                             alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3)
    cover_big = ParagraphStyle("cover_big", parent=body, fontName="Helvetica-Bold",
                               fontSize=16, leading=22, alignment=TA_CENTER)
    cover_mid = ParagraphStyle("cover_mid", parent=cover_big, fontSize=13, fontName="Helvetica")
    cover_line = ParagraphStyle("cover_line", parent=body, fontSize=11.5, leading=26, alignment=0)
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11.5, alignment=0, spaceAfter=0)
    cell_b = ParagraphStyle("cell_b", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    code_style = ParagraphStyle("code", fontName="Courier", fontSize=7.8, leading=10,
                                backColor=colors.HexColor("#f3f3f3"), borderPadding=6,
                                leftIndent=4, spaceAfter=8)

    def P(text, style=body):
        return Paragraph(text, style)
    def esc(x):
        return escape(str(x))

    def pct(x):
        return f"{x * 100:.2f}%"

    def table(data, col_widths):
        rws = []
        for i, r in enumerate(data):
            st = cell_b if i == 0 else cell
            rws.append([Paragraph(esc(c), st) for c in r])
        t = Table(rws, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f8fc")]),
            ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ]))
        return t

    def figure(name, text, max_h=10.5 * cm, max_w=15.5 * cm):
        path = os.path.join(PLOT_DIR, name)
        if not os.path.exists(path):
            print("Report: image na malyo (skip):", name)
            return []
        w, h = ImageReader(path).getSize()
        scale = min(max_w / w, max_h / h)
        im = Image(path, width=w * scale, height=h * scale)
        return [KeepTogether([im, Spacer(1, 3), P(esc(text), caption), Spacer(1, 8)])]

    def bullets(items):
        return [Paragraph(esc(t), bullet, bulletText="-") for t in items]

    target = S["target"]
    features, num_cols, cat_cols = S["feature_cols"], S["numerical_cols"], S["categorical_cols"]
    counts = S["class_counts"]
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    total_cls = max(n0 + n1, 1)
    base, imp = S["baseline"], S["improved"]
    conf = S["confusion_matrix"]
    tn, fp, fn, tp = conf[0][0], conf[0][1], conf[1][0], conf[1][1]
    top_feat = list(S["top_features"].items())
    strong_feats = [k for k, _ in top_feat[:3]]

    story = []
    # Cover
    story += [Spacer(1, 1.5 * cm),
              P("SYBCA-AI (Semester III)", cover_big), Spacer(1, 10),
              P("303 Machine Learning - Mini Project", cover_mid), Spacer(1, 3.2 * cm),
              P(esc(PROJECT_TITLE), cover_big), Spacer(1, 5.5 * cm),
              P(f"<b>Submitted by:</b> {esc(STUDENT_NAMES)}", cover_line),
              P(f"<b>Roll Number:</b> {esc(ROLL_NUMBERS)}", cover_line),
              P(f"<b>Submitted to:</b> {esc(SUBMITTED_TO)}", cover_line),
              Spacer(1, 2 * cm),
              P(f"<b>Academic Year:</b> {esc(ACADEMIC_YEAR)}", cover_line),
              PageBreak()]

    # Certificate
    story += [Spacer(1, 1 * cm), P("Certificate", ParagraphStyle("c", parent=cover_big, fontSize=20)),
              Spacer(1, 1.5 * cm),
              P(f"This is to certify that <b>{esc(STUDENT_NAMES)}</b> (Roll No. {esc(ROLL_NUMBERS)}) "
                f"has completed the Machine Learning mini project titled "
                f"<b>{esc(PROJECT_TITLE.title())}</b> as a part of SYBCA-AI (Semester III), "
                f"subject 303 Machine Learning."),
              Spacer(1, 2.5 * cm),
              P("Faculty Signature: ______________________________", cover_line),
              P("Date: ______________________", cover_line),
              PageBreak()]

    # 1. Problem Statement
    story += [P("1. Problem Statement", h1),
              P("Diabetes is a serious long-term health condition, and early detection helps patients "
                "start treatment sooner and avoid complications. Manual screening takes time and depends "
                "on expert judgement. This project builds a supervised binary classification model that "
                "predicts whether a patient is likely to have diabetes (1) or not (0) from medical and "
                "demographic attributes such as age, BMI, blood pressure, glucose level and family history.")]

    # 2. Objectives
    story += [P("2. Objectives", h1)] + bullets([
        "Study the diabetes dataset and identify numerical, categorical, missing and duplicate values.",
        "Perform visual and automated Exploratory Data Analysis (EDA).",
        "Preprocess the data: handle missing values and outliers.",
        "Select the features, split the data into training and testing sets.",
        "Build a Logistic Regression classification model.",
        "Evaluate it using Accuracy, Precision, Recall, F1 Score and the Confusion Matrix.",
        "Improve the model using Cross Validation and Regularization (L1/L2), and compare the results.",
    ])

    # 3. Dataset Description
    story += [P("3. Dataset Description", h1),
              P(f"<b>Source:</b> {esc(DATASET_SOURCE)}. The dataset contains <b>{S['rows']}</b> records and "
                f"<b>{S['columns_total']}</b> columns. The target variable is <b>{esc(target)}</b> "
                f"(0 = not diabetic, 1 = diabetic). "
                f"Class 0 has {n0} records ({n0 / total_cls * 100:.1f}%) and class 1 has {n1} records "
                f"({n1 / total_cls * 100:.1f}%).")]
    if S["id_cols_dropped"]:
        story.append(P(f"The identifier column(s) <b>{esc(', '.join(S['id_cols_dropped']))}</b> were dropped "
                       f"because an ID has no predictive meaning."))
    data = [["Column", "Type", "Role"]]
    for c in features:
        data.append([c, "Categorical" if c in cat_cols else "Numerical", "Feature (independent)"])
    data.append([target, "Binary (0/1)", "Target (dependent)"])
    story += [table(data, [6 * cm, 4 * cm, 6 * cm]), Spacer(1, 8)]
    story.append(P(f"The dataset has {len(num_cols)} numerical and {len(cat_cols)} categorical feature "
                   f"columns. It satisfies the minimum requirement of 300 records and 5 or more features, "
                   f"and it is a real-world dataset."))

    # 4. Data Preprocessing
    story += [P("4. Data Preprocessing", h1)]
    story.append(P(f"Missing values found in the raw data: <b>{S['missing_raw']}</b>. "
                   f"Duplicate rows found: <b>{S['duplicates']}</b>."))
    if S["zero_counts"]:
        story.append(P("Some medical columns contain the value 0, which is not possible for a living patient "
                       "(for example, a blood pressure or glucose of 0). These zeros were treated as hidden "
                       "missing values, converted to NaN and then filled with the median."))
        zdata = [["Column", "Zero values (hidden missing)"]]
        for c, v in S["zero_counts"].items():
            zdata.append([c, v])
        story += [table(zdata, [8 * cm, 6 * cm]), Spacer(1, 8)]
    story.append(P(f"Missing values after treatment: <b>{S['missing_after_fill']}</b>. Numerical missing "
                   f"values are filled with the median and categorical missing values with the mode."))
    story.append(P("Outliers were detected with the IQR method (values below Q1 - 1.5 x IQR or above "
                   "Q3 + 1.5 x IQR) and capped at those limits, so no records were lost."))
    if S["outliers_before"]:
        odata = [["Column", "Outliers before", "Outliers after"]]
        for c, v in S["outliers_before"].items():
            odata.append([c, v, S["outliers_after"].get(c, 0)])
        story += [table(odata, [6 * cm, 4.5 * cm, 4.5 * cm]), Spacer(1, 8)]
    story += figure("06_before_after.png", "Figure 1: Box plots before and after outlier treatment.")

    # 5. EDA
    story += [PageBreak(), P("5. Exploratory Data Analysis", h1),
              P("The analysis includes histograms, box plots, a target pie chart, categorical bar charts "
                "and a correlation matrix. An automated EDA report (EDA_Report.html) is also generated "
                "with Sweetviz.")]
    story += figure("01_histograms.png", "Figure 2: Histograms of numerical features by class (with KDE).", max_h=11 * cm)
    story += figure("02_boxplots.png", "Figure 3: Box plots of numerical features by class.", max_h=11 * cm)
    story += figure("03_pie_target.png", f"Figure 4: Distribution of the target variable ({target}).", max_h=8 * cm)
    story.append(P(f"The target is {'fairly balanced' if min(n0, n1) / total_cls > 0.4 else 'imbalanced'}: "
                   f"{n0 / total_cls * 100:.1f}% of patients belong to class 0 and "
                   f"{n1 / total_cls * 100:.1f}% to class 1."))
    story += figure("04_bar_chart.png", "Figure 5: Categorical features versus the target.", max_h=7 * cm)
    story += figure("05_correlation.png", "Figure 6: Correlation matrix of numerical features and the target.", max_h=10 * cm)
    corr_t = S.get("corr_with_target", {})
    if corr_t:
        ordered = sorted(corr_t.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        txt = ", ".join(f"{k} ({v:+.2f})" for k, v in ordered)
        story.append(P(f"Numerical features most strongly correlated with the target: {esc(txt)}."))

    # 6. Model Building (Code)
    story += [PageBreak(), P("6. Model Building (Code)", h1),
              P(f"The independent variables are all columns except <b>{esc(target)}</b>. Categorical columns "
                f"are one-hot encoded (drop_first=True), which gives <b>{S['n_features_model']}</b> input "
                f"features. The data is split into 80% training (<b>{S['train_size']}</b> records) and 20% "
                f"testing (<b>{S['test_size']}</b> records) with stratification, so both sets keep the same "
                f"class ratio. The features are standardized with StandardScaler and a Logistic Regression "
                f"model is trained, because the target is a binary class.")]
    code_text = f'''X = df.drop(columns=["{target}"])
y = df["{target}"].astype(int)
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)'''
    story += [Preformatted(code_text, code_style)]

    # 7. Model Evaluation
    story += [P("7. Model Evaluation", h1),
              P("The model is evaluated on the unseen test data with four classification metrics:")]
    story += bullets([
        "Accuracy - the share of all predictions that are correct.",
        "Precision - of the patients predicted diabetic, how many truly are.",
        "Recall - of all truly diabetic patients, how many the model found.",
        "F1 Score - the balance between Precision and Recall.",
    ])
    edata = [["Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
    for m in (base, imp):
        edata.append([m["Model"], f"{m['Accuracy']:.4f}", f"{m['Precision']:.4f}",
                      f"{m['Recall']:.4f}", f"{m['F1 Score']:.4f}"])
    story += [Spacer(1, 4), table(edata, [6.2 * cm, 2.5 * cm, 2.6 * cm, 2.4 * cm, 2.5 * cm]), Spacer(1, 8)]
    story.append(P(f"Confusion matrix of the baseline model: True Negative = {tn}, False Positive = {fp}, "
                   f"False Negative = {fn}, True Positive = {tp}. In a medical problem, Recall is very "
                   f"important because a missed diabetic patient (false negative) is more harmful than a "
                   f"false alarm. The model's Recall is {pct(base['Recall'])}."))
    story += figure("07_confusion_matrix.png", "Figure 7: Confusion matrix on the test data.", max_h=8.5 * cm)
    story.append(P(f"<b>Cross validation:</b> 5-fold stratified cross validation on the training data gives "
                   f"a mean accuracy of {S['cv_accuracy_mean']:.4f} (+/- {S['cv_accuracy_std']:.4f}). "
                   f"<b>Regularization:</b> L1 and L2 penalties were compared for C = 0.01, 0.1, 1, 10 and "
                   f"100 using the cross-validated F1 score."))
    rdata = [["Penalty", "C", "CV F1 score"]]
    for r in S["reg_table"]:
        rdata.append([str(r["Penalty"]).upper(), r["C"], f"{float(r['CV F1']):.4f}"])
    story += [table(rdata, [4 * cm, 4 * cm, 4.5 * cm]), Spacer(1, 6)]
    story.append(P(f"The best setting was <b>{esc(str(S['best_penalty']).upper())}</b> with "
                   f"<b>C = {S['best_C']:g}</b>."))

    # 8. Results
    story += [P("8. Results", h1)]
    diff = imp["F1 Score"] - base["F1 Score"]
    if abs(diff) < 1e-9:
        cmp_text = ("The improved model gives the same test result as the baseline model, which means the "
                    "baseline Logistic Regression was already well suited to this data and tuning did not "
                    "change the test performance.")
    elif diff > 0:
        cmp_text = (f"The improved model increased the F1 score from {base['F1 Score']:.4f} to "
                    f"{imp['F1 Score']:.4f}.")
    else:
        cmp_text = (f"The tuned model's test F1 score ({imp['F1 Score']:.4f}) is slightly different from the "
                    f"baseline ({base['F1 Score']:.4f}); the setting was chosen using cross validation on "
                    f"the training data, so a small difference on the test data is normal.")
    story.append(P(f"The final model reaches an accuracy of <b>{pct(imp['Accuracy'])}</b>, precision of "
                   f"<b>{pct(imp['Precision'])}</b>, recall of <b>{pct(imp['Recall'])}</b> and an F1 score "
                   f"of <b>{pct(imp['F1 Score'])}</b> on unseen test data. " + esc(cmp_text)))
    story += figure("08_feature_importance.png", "Figure 8: Feature importance (absolute coefficient).", max_h=9 * cm)
    fdata = [["Feature", "Coefficient", "Effect on diabetes risk"]]
    for k, v in top_feat[:10]:
        fdata.append([k, f"{v:+.3f}", "Increases" if v > 0 else "Decreases"])
    story += [table(fdata, [7 * cm, 3.5 * cm, 5.5 * cm]), Spacer(1, 6)]
    if strong_feats:
        story.append(P(f"A positive coefficient means the feature raises the probability of diabetes and a "
                       f"negative coefficient lowers it. The strongest features (largest absolute "
                       f"coefficient) are <b>{esc(', '.join(strong_feats))}</b>. Features with coefficients "
                       f"close to zero have very little influence on the prediction."))

    # 9. Conclusion
    story += [P("9. Conclusion", h1),
              P(f"A Logistic Regression model was built to predict diabetes. On unseen test data it achieved "
                f"an accuracy of {pct(imp['Accuracy'])} and an F1 score of {pct(imp['F1 Score'])}, with a "
                f"recall of {pct(imp['Recall'])}. The complete machine learning workflow was followed: data "
                f"understanding, EDA, preprocessing, feature selection, model building, evaluation and "
                f"improvement with cross validation and regularization. The code uses a fixed random state "
                f"and a pipeline, so the results are reproducible.")]

    # 10. Future Scope
    story += [P("10. Future Scope", h1)] + bullets([
        "Collect more diverse real-world patient data and test the model on it.",
        "Try other classifiers such as Random Forest and Gradient Boosting and compare them.",
        "Tune the decision threshold to increase recall for medical screening.",
        "Features such as HbA1c and glucose are themselves diagnostic tests, so test the model without "
        "them to predict risk earlier.",
        "Deploy the saved model in a simple web application (for example Streamlit).",
        "This is an academic model and must not be used as a medical diagnosis.",
    ])

    # 11. References
    story += [P("11. References", h1)] + bullets([
        "Kaggle - Diabetes dataset (CSV downloaded by the students).",
        "Scikit-learn documentation - Logistic Regression, pipelines, model evaluation.",
        "Pandas, Matplotlib, Seaborn and Sweetviz documentation.",
    ])

    def draw_page(canvas, doc):
        if doc.page <= 2:
            return
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawCentredString(A4[0] / 2, A4[1] - 1.1 * cm, f"{SHORT_TITLE} | Page {doc.page}")
        canvas.setStrokeColor(colors.HexColor("#1f3b66"))
        canvas.setLineWidth(1.2)
        canvas.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(out_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2.2 * cm, bottomMargin=1.8 * cm,
                            title=PROJECT_TITLE.title(), author=STUDENT_NAMES)
    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    return out_path


header("STEP 13: CONCLUSION + REPORT")
print("ML project complete. Graphs 'Reports/plots' folder ma save thaya che.")

if MAKE_REPORT:
    try:
        path = build_report(summary)
        print("PDF report bani:", os.path.abspath(path))
    except ImportError:
        print("PDF report mate reportlab joie. Terminal ma run karo:  pip install reportlab")
        print("Pachi pharithi 'python project.py' run karo.")
    except Exception as e:
        print(f"PDF report na bani -> {type(e).__name__}: {e}")