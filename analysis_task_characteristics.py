# ==============================================================================
# AI利用モード分析（Colab用・v3：第9回キーワード調整版）
# ==============================================================================
# 修正点:
#   - 第9回: 「探索」→「プロンプト設計（指示・指定・具体的）」に変更
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = ['DejaVu Sans']

# -----------------------------------------------------------------------------
# 設定（v3：第9回キーワード調整）
# -----------------------------------------------------------------------------
AI_KEYWORDS = {
    "no_ai": ["使用しなかった", "使わなかった"],
    "explore": ["相談", "説明", "指標", "データ", "探索", "質問", "意味"],
    "code": ["コード", "Colab", "前処理", "エラー", "デバッグ", "実装"],
    "reflect": ["解説", "振り返り"],
}

TASK_WORDS = {
    7: ["コード", "エラー", "動かない", "修正", "書き方"],              # コード実装
    9: ["指示", "指定", "具体的", "形式", "縦軸", "横軸", "説明"],      # プロンプト設計
    11: ["ヒスト", "分布", "ビン", "列", "賃金", "選"],                 # データ選択
}

TASK_LABELS = {
    7: "コード実装",
    9: "プロンプト設計",
    11: "データ選択",
}

# -----------------------------------------------------------------------------
# ヘルパー関数
# -----------------------------------------------------------------------------
def find_col(df, kw):
    return next((c for c in df.columns if kw in c), None)

def has_any(text, words):
    t = str(text).lower() if pd.notna(text) else ""
    return any(w.lower() in t for w in words)

def classify_mode(text):
    if pd.isna(text):
        return "unknown"
    t = str(text)
    if has_any(t, AI_KEYWORDS["no_ai"]):
        return "no_ai"
    if not has_any(t, ["生成AI", "ChatGPT", "Gemini", "Claude", "AI"]):
        return "unknown"
    if has_any(t, AI_KEYWORDS["explore"]):
        return "ai_explore"
    if has_any(t, AI_KEYWORDS["code"]):
        return "ai_code"
    if has_any(t, AI_KEYWORDS["reflect"]):
        return "ai_reflect"
    return "ai_other"

def coalesce(row, cols):
    for c in cols:
        v = row.get(c)
        if pd.notna(v) and str(v).strip():
            return str(v).strip()
    return ""

# -----------------------------------------------------------------------------
# データ加工
# -----------------------------------------------------------------------------
action_col = find_col(df_all, "今日取り組んだ内容")
df_all["ai_mode"] = df_all[action_col].apply(classify_mode) if action_col else "unknown"

good_cols = [c for c in df_all.columns if "良かった点" in c or "改善" in c]
q_cols = [c for c in df_all.columns if "もう一問" in c or "追加" in c]
df_all["free_text"] = df_all.apply(
    lambda r: coalesce(r, good_cols) + " " + coalesce(r, q_cols), axis=1
)

for s, words in TASK_WORDS.items():
    df_all[f"flag_s{s}"] = df_all.apply(
        lambda r, s=s, words=words: has_any(r["free_text"], words) if r["session"] == s else False,
        axis=1
    )

# -----------------------------------------------------------------------------
# 分析結果
# -----------------------------------------------------------------------------
print("=" * 60)
print("AI利用モード分析（v3：第9回キーワード調整版）")
print("=" * 60)

print("\n[1] AI利用モード分布")
print("-" * 40)
print(df_all["ai_mode"].value_counts())

print("\n[2] セッション × AI利用モード クロス表")
print("-" * 40)
print(pd.crosstab(df_all["session"], df_all["ai_mode"], margins=True))

print("\n[3] 課題特性フラグ別 分析（主要結果）")
print("-" * 40)

results_summary = []
for s in [7, 9, 11]:
    flag = f"flag_s{s}"
    subset = df_all[df_all["session"] == s]
    if len(subset) > 0:
        true_n = subset[flag].sum()
        false_n = len(subset) - true_n
        
        print(f"\n{'='*50}")
        print(f"Session {s}：{TASK_LABELS[s]}")
        print(f"{'='*50}")
        print(f"フラグ分布: True={true_n}, False={false_n}")
        
        if true_n > 0:
            result = subset.groupby(flag)[["AI_support", "Metacognition"]].agg(["mean", "count"]).round(2)
            print(result)
            
            # 差を計算
            true_ai = subset[subset[flag] == True]["AI_support"].mean()
            false_ai = subset[subset[flag] == False]["AI_support"].mean()
            true_meta = subset[subset[flag] == True]["Metacognition"].mean()
            false_meta = subset[subset[flag] == False]["Metacognition"].mean()
            
            print(f"\n→ AI支援指数の差: {true_ai:.2f} - {false_ai:.2f} = {true_ai - false_ai:+.2f}")
            print(f"→ メタ認知指数の差: {true_meta:.2f} - {false_meta:.2f} = {true_meta - false_meta:+.2f}")
            
            results_summary.append({
                "session": s,
                "label": TASK_LABELS[s],
                "true_n": true_n,
                "false_n": false_n,
                "ai_diff": true_ai - false_ai,
                "meta_diff": true_meta - false_meta,
            })
        else:
            print("→ フラグTrue該当なし")

# -----------------------------------------------------------------------------
# 結果サマリー表
# -----------------------------------------------------------------------------
if results_summary:
    print("\n" + "=" * 60)
    print("結果サマリー（論文用）")
    print("=" * 60)
    summary_df = pd.DataFrame(results_summary)
    summary_df.columns = ["Session", "課題特性", "言及あり(N)", "言及なし(N)", "AI支援差", "メタ認知差"]
    print(summary_df.to_string(index=False))

# -----------------------------------------------------------------------------
# 可視化：課題特性フラグ別の比較
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, s in enumerate([7, 9, 11]):
    flag = f"flag_s{s}"
    subset = df_all[df_all["session"] == s].copy()
    subset["flag_label"] = subset[flag].map({True: f"With {TASK_LABELS[s]}", False: "Without"})
    
    # 両指数を縦持ちに変換
    melted = subset.melt(
        id_vars=["flag_label"],
        value_vars=["AI_support", "Metacognition"],
        var_name="Index",
        value_name="Score"
    )
    
    sns.barplot(data=melted, x="Index", y="Score", hue="flag_label", ax=axes[idx])
    axes[idx].set_ylim(0, 5.5)
    axes[idx].set_title(f"Session {s}: {TASK_LABELS[s]}")
    axes[idx].set_xlabel("")
    axes[idx].legend(title="", loc="upper right")

plt.tight_layout()
plt.savefig("fig_task_characteristics.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 60)
print("✅ 分析完了: fig_task_characteristics.png")
print("=" * 60)
