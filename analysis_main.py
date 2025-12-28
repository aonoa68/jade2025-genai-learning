# ============================================================
# リメディアル論文2025 - 分析コード（完全版）
# ============================================================
# 
# 論文タイトル：内なる他者との対話：生成AIを用いた統計演習における学習支援の実践報告
# Title: Dialogue with the Inner Other: A Practical Report on Learning Support 
#        Using Generative AI in Statistics Exercises
# 
# 掲載誌：リメディアル教育研究（日本リメディアル教育学会）
#
# ============================================================
# 分析内容：
#   1. 基本集計（AI支援指数・メタ認知指数）- 表2
#   2. 縦断変化分析 - 図1
#   3. 感情プロファイル分析 - 表3
#   4. 課題特性分析 - 表5
#   5. 感情×学習指数の相関分析（探索的・論文未掲載）
#   6. 対応分析（探索的・論文未掲載）
#   7. 信頼性係数の計算（参考・論文未掲載）
#
# 注：5-7は探索的分析として実施したが、床効果・自己選択バイアス等の
#     理由により論文本文には含めていない。分析過程の透明性のため公開。
# ============================================================

# ============================================================
# ① 準備：ライブラリのインポート
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr
import itertools

# 日本語フォント設定（Google Colab用）
try:
    import japanize_matplotlib
except ImportError:
    print("japanize_matplotlibがインストールされていません。")
    print("pip install japanize_matplotlib でインストールしてください。")

# ============================================================
# ② データ読み込み関数
# ============================================================
def load_and_clean(path, session):
    """
    各回のExcelファイルを読み込み、共通形式に整形
    
    Parameters:
    -----------
    path : str
        Excelファイルのパス
    session : int
        セッション番号（5, 7, 9, 11）
    
    Returns:
    --------
    df : DataFrame
        整形済みデータフレーム
    """
    df = pd.read_excel(path)
    df["session"] = session
    # ID列の特定（列名に"ID"を含む列）
    id_col = [c for c in df.columns if "ID" in c][0]
    df = df.rename(columns={id_col: "id"})
    df["id"] = df["id"].astype(str)
    return df


# ============================================================
# ③ Google Colabでのファイルアップロード（必要に応じて使用）
# ============================================================
def upload_files_colab():
    """Google Colabでファイルをアップロード"""
    from google.colab import files
    uploaded = files.upload()
    return uploaded


# ============================================================
# ④ メイン分析クラス
# ============================================================
class GenAILearningAnalysis:
    """生成AI学習支援の分析クラス"""
    
    def __init__(self):
        self.df_all = None
        self.dfE_fixed = None
        self.ai_support_items = []
        self.metacog_items = []
        self.emotion_names = ["期待", "驚き", "喜び", "信頼", "怒り", "嫌悪", "悲しみ", "恐れ"]
        
    def load_data(self, file_paths):
        """
        データの読み込みと結合
        
        Parameters:
        -----------
        file_paths : dict
            セッション番号をキー、ファイルパスを値とする辞書
            例: {5: 'session5.xlsx', 7: 'session7.xlsx', ...}
        """
        dfs = []
        for session, path in file_paths.items():
            df = load_and_clean(path, session)
            dfs.append(df)
            print(f"第{session}回: {len(df)}件")
        
        self.df_all = pd.concat(dfs, ignore_index=True)
        print(f"\n合計: {len(self.df_all)}件")
        
    def define_indices(self):
        """AI支援指数・メタ認知指数の構成項目を定義"""
        # AI支援指数（6項目）
        self.ai_support_items = [
            c for c in self.df_all.columns
            if ("生成AI" in c) and ("役立" in c or "理解" in c or "整理" in c)
        ]
        
        # メタ認知指数（9項目）
        self.metacog_items = [
            c for c in self.df_all.columns
            if "説明できる" in c or "復習" in c or "良い質問" in c or "良い問い" in c
        ]
        
        print(f"AI支援指数: {len(self.ai_support_items)}項目")
        for c in self.ai_support_items:
            print(f"  - {c[:60]}...")
            
        print(f"\nメタ認知指数: {len(self.metacog_items)}項目")
        for c in self.metacog_items:
            print(f"  - {c[:60]}...")
        
        # 指数の算出
        self.df_all["AI_support"] = self.df_all[self.ai_support_items].mean(axis=1)
        self.df_all["Metacognition"] = self.df_all[self.metacog_items].mean(axis=1)
        
    def summary_statistics(self):
        """
        表2：セッション別要約統計量
        """
        print("\n" + "="*60)
        print("【表2：セッション別 AI支援指数・メタ認知指数】")
        print("="*60)
        
        summary = (
            self.df_all
            .groupby("session")[["AI_support", "Metacognition"]]
            .agg(["mean", "std", "count"])
            .round(3)
        )
        print(summary)
        return summary
    
    def longitudinal_analysis(self, save_fig=True):
        """
        図1：縦断変化分析
        """
        print("\n" + "="*60)
        print("【縦断分析】")
        print("="*60)
        
        # 複数セッションに回答したIDを特定
        id_session_df = (
            self.df_all.groupby('id')['session']
            .apply(lambda x: sorted(x.unique().tolist()))
            .reset_index()
        )
        id_session_df['n_sessions'] = id_session_df['session'].apply(len)
        multi_session_ids = id_session_df[id_session_df['n_sessions'] >= 2]
        
        print(f"異なるセッションに回答したID: {len(multi_session_ids)}名")
        print(multi_session_ids.to_string())
        
        valid_ids = multi_session_ids['id'].tolist()
        df_unique = self.df_all.groupby(["id", "session"], as_index=False).mean(numeric_only=True)
        df_long2 = df_unique[df_unique["id"].isin(valid_ids)]
        
        # 図1：縦断変化（モノクロ版）
        styles = [
            {'color': '0.3', 'marker': 'o', 'linestyle': '-', 'fillstyle': 'full'},
            {'color': '0.5', 'marker': 's', 'linestyle': '--', 'fillstyle': 'full'},
            {'color': '0.4', 'marker': '^', 'linestyle': ':', 'fillstyle': 'none'},
        ]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        case_labels = ["Case 1", "Case 2", "Case 3"]
        
        for i, (pid, g) in enumerate(df_long2.groupby("id")):
            ax.plot(g["session"], g["AI_support"],
                    color=styles[i]['color'],
                    linestyle=styles[i]['linestyle'],
                    linewidth=2,
                    marker=styles[i]['marker'],
                    markersize=8,
                    fillstyle=styles[i]['fillstyle'],
                    markeredgewidth=1.5,
                    markeredgecolor=styles[i]['color'],
                    label=case_labels[i],
                    alpha=0.9)
        
        # 平均線（追跡可能者）
        mean_tracking = df_long2.groupby("session")["AI_support"].mean()
        ax.plot(mean_tracking.index, mean_tracking.values,
                color='black', linewidth=2.5, marker='D', markersize=8,
                fillstyle='full', label="平均（追跡可能者）", linestyle='-')
        
        # 平均線（全体）
        mean_all = self.df_all.groupby("session")["AI_support"].mean()
        ax.plot(mean_all.index, mean_all.values,
                color='0.2', linewidth=2.5, marker='x', markersize=9,
                label="平均（全体）", linestyle='-.')
        
        ax.set_xlabel("回", fontsize=12)
        ax.set_ylabel("AI支援指数", fontsize=12)
        ax.set_title("生成AI支援の縦断変化（内なる他者モデル）", fontsize=14)
        ax.set_xticks([5, 7, 9, 11])
        ax.set_ylim(0, 6)
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_fig:
            plt.savefig("fig1_longitudinal_ai_support_mono.png", dpi=300, bbox_inches='tight')
            print("📊 図1を保存しました: fig1_longitudinal_ai_support_mono.png")
        plt.show()
        
        return df_long2
    
    def emotion_analysis(self):
        """
        表3：感情プロファイル分析
        """
        print("\n" + "="*60)
        print("【表3：感情プロファイル】")
        print("="*60)
        
        def normalize_emotion_col(col):
            """感情列名を正規化"""
            m = re.search(r"\[(.*?)\]", str(col))
            if not m:
                return col
            name = m.group(1).replace(" ", "").replace("\u3000", "").strip()
            return name
        
        def map_emotion(val):
            """感情の文字列を数値にマッピング"""
            if pd.isna(val):
                return np.nan
            val = str(val).strip()
            if "感じなかった" in val:
                return 0
            if "弱く" in val:
                return 1
            if "やや" in val:
                return 2
            if "強く" in val:
                return 3
            return np.nan
        
        def coalesce_duplicate_cols(df, colname):
            """重複列を統合"""
            cols = df.loc[:, df.columns == colname]
            if cols.shape[1] == 1:
                return cols.iloc[:, 0]
            return cols.bfill(axis=1).iloc[:, 0]
        
        # 感情列を特定
        emotion_cols_all = [
            c for c in self.df_all.columns 
            if "感情" in str(c) and "[" in str(c) and "]" in str(c)
        ]
        
        # 数値変換
        df_emotion = self.df_all.copy()
        for c in emotion_cols_all:
            df_emotion[c] = df_emotion[c].apply(map_emotion)
        
        # 列名を正規化
        rename_dict = {c: normalize_emotion_col(c) for c in emotion_cols_all}
        dfE = df_emotion.rename(columns=rename_dict)
        
        # 感情列を統合
        self.dfE_fixed = pd.DataFrame({"session": dfE["session"]})
        for name in self.emotion_names:
            self.dfE_fixed[name] = coalesce_duplicate_cols(dfE, name)
        
        # 回別の感情平均
        emotion_summary = self.dfE_fixed.groupby("session")[self.emotion_names].mean().round(3)
        print(emotion_summary)
        
        # Kruskal-Wallis検定
        print("\n■ セッション間比較（Kruskal-Wallis検定）")
        for emotion in self.emotion_names:
            groups = [self.dfE_fixed[self.dfE_fixed["session"]==s][emotion].dropna() for s in [5,7,9,11]]
            if all(len(g) > 0 for g in groups):
                stat, p = kruskal(*groups)
                sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {emotion}: H={stat:.2f}, p={p:.4f} {sig}")
        
        return emotion_summary
    
    def task_characteristics_analysis(self):
        """
        表5：課題特性分析
        """
        print("\n" + "="*60)
        print("【表5：課題特性分析】")
        print("="*60)
        
        TASK_WORDS = {
            7: ["コード", "エラー", "動かない", "修正", "書き方"],
            9: ["指示", "指定", "具体的", "形式", "縦軸", "横軸", "説明"],
            11: ["ヒスト", "分布", "ビン", "列", "賃金", "選"],
        }
        
        TASK_LABELS = {7: "コード実装", 9: "プロンプト設計", 11: "データ選択"}
        
        def has_any(text, words):
            t = str(text).lower() if pd.notna(text) else ""
            return any(w.lower() in t for w in words)
        
        def coalesce(row, cols):
            for c in cols:
                v = row.get(c)
                if pd.notna(v) and str(v).strip():
                    return str(v).strip()
            return ""
        
        # 自由記述の統合
        good_cols = [c for c in self.df_all.columns if "良かった点" in c or "改善" in c]
        q_cols = [c for c in self.df_all.columns if "もう一問" in c or "追加" in c]
        self.df_all["free_text"] = self.df_all.apply(
            lambda r: coalesce(r, good_cols) + " " + coalesce(r, q_cols), axis=1
        )
        
        # フラグ作成
        for s, words in TASK_WORDS.items():
            self.df_all[f"flag_s{s}"] = self.df_all.apply(
                lambda r, s=s, words=words: has_any(r["free_text"], words) if r["session"] == s else False,
                axis=1
            )
        
        # 結果出力
        results = []
        for s in [7, 9, 11]:
            flag = f"flag_s{s}"
            subset = self.df_all[self.df_all["session"] == s]
            true_n = subset[flag].sum()
            false_n = len(subset) - true_n
            
            print(f"\n■ Session {s}：{TASK_LABELS[s]}")
            print(f"  フラグ分布: True={true_n}, False={false_n}")
            
            if true_n > 0:
                result = subset.groupby(flag)[["AI_support", "Metacognition"]].agg(["mean", "count"]).round(2)
                print(result)
                
                true_ai = subset[subset[flag] == True]["AI_support"].mean()
                false_ai = subset[subset[flag] == False]["AI_support"].mean()
                true_meta = subset[subset[flag] == True]["Metacognition"].mean()
                false_meta = subset[subset[flag] == False]["Metacognition"].mean()
                
                results.append({
                    "session": s,
                    "label": TASK_LABELS[s],
                    "true_n": true_n,
                    "false_n": false_n,
                    "ai_true": true_ai,
                    "ai_false": false_ai,
                    "meta_true": true_meta,
                    "meta_false": false_meta,
                })
        
        return pd.DataFrame(results)
    
    # ============================================================
    # 探索的分析（論文未掲載）
    # ============================================================
    
    def correlation_analysis(self):
        """
        探索的分析：感情×学習指数の相関
        
        注：床効果・自己選択バイアスのため論文本文には未掲載
        """
        print("\n" + "="*60)
        print("【探索的分析：感情×学習指数の相関】")
        print("注：床効果・自己選択バイアスのため論文本文には未掲載")
        print("="*60)
        
        df_corr = self.dfE_fixed.copy()
        df_corr["AI_support"] = self.df_all["AI_support"].values
        df_corr["Metacognition"] = self.df_all["Metacognition"].values
        
        print("\n■ 相関係数と有意性検定（Spearman's ρ）")
        results = []
        for emotion in self.emotion_names:
            for outcome in ["AI_support", "Metacognition"]:
                valid = df_corr[[emotion, outcome]].dropna()
                if len(valid) > 10:
                    rho, p = spearmanr(valid[emotion], valid[outcome])
                    sig = "**" if p < 0.01 else "*" if p < 0.05 else ""
                    results.append({
                        "感情": emotion, "指標": outcome,
                        "ρ": round(rho, 3), "p": round(p, 4), "有意": sig, "N": len(valid)
                    })
        
        corr_df = pd.DataFrame(results)
        print(corr_df.to_string(index=False))
        
        # 度数分布（床効果の確認）
        print("\n■ 感情データの度数分布（床効果の確認）")
        for emotion in self.emotion_names:
            total = df_corr[emotion].notna().sum()
            n_zero = (df_corr[emotion] == 0).sum()
            pct_zero = n_zero / total * 100 if total > 0 else 0
            effect = "⚠️床効果" if pct_zero >= 50 else ""
            print(f"  {emotion}: 0の割合={pct_zero:.1f}% {effect}")
        
        return corr_df
    
    def correspondence_analysis(self):
        """
        探索的分析：対応分析
        
        注：パターンが不明瞭なため論文本文には未掲載
        """
        try:
            import prince
        except ImportError:
            print("princeがインストールされていません。")
            print("pip install prince でインストールしてください。")
            return None
        
        print("\n" + "="*60)
        print("【探索的分析：対応分析】")
        print("注：パターンが不明瞭なため論文本文には未掲載")
        print("="*60)
        
        learning_items = self.ai_support_items + self.metacog_items
        
        df_ca = self.df_all.copy()
        df_ca["respondent_id"] = df_ca["session"].astype(str) + "_" + df_ca.index.astype(str)
        
        df_learning = df_ca[["respondent_id", "session"] + learning_items].copy()
        df_learning["valid_count"] = df_learning[learning_items].notna().sum(axis=1)
        df_learning_valid = df_learning[df_learning["valid_count"] >= 2].copy()
        
        print(f"有効回答数: {len(df_learning_valid)}")
        
        df_matrix = df_learning_valid.set_index("respondent_id")[learning_items].copy()
        for col in df_matrix.columns:
            col_mean = df_matrix[col].mean()
            df_matrix[col] = df_matrix[col].fillna(col_mean)
        
        ca = prince.CA(n_components=2, random_state=42)
        ca = ca.fit(df_matrix)
        
        print(f"第1軸の寄与率: {ca.percentage_of_variance_[0]:.1f}%")
        print(f"第2軸の寄与率: {ca.percentage_of_variance_[1]:.1f}%")
        print(f"累積寄与率: {sum(ca.percentage_of_variance_[:2]):.1f}%")
        
        return ca
    
    def reliability_analysis(self):
        """
        参考：信頼性係数の計算
        
        注：セッションごとに質問が異なるため、全体のα係数は算出不可
        """
        print("\n" + "="*60)
        print("【参考：信頼性係数】")
        print("注：セッションごとに質問が異なるため、全体のα係数は算出不可")
        print("="*60)
        
        def cronbachs_alpha(df_items):
            df_clean = df_items.dropna()
            k = df_clean.shape[1]
            if k < 2 or len(df_clean) < 2:
                return np.nan
            item_variances = df_clean.var(axis=0, ddof=1)
            total_variance = df_clean.sum(axis=1).var(ddof=1)
            if total_variance == 0:
                return np.nan
            return (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
        
        df_ai_items = self.df_all[self.ai_support_items].copy()
        df_meta_items = self.df_all[self.metacog_items].copy()
        
        alpha_ai = cronbachs_alpha(df_ai_items)
        alpha_meta = cronbachs_alpha(df_meta_items)
        
        print(f"AI支援指数: 有効N={df_ai_items.dropna().shape[0]}, α={alpha_ai:.3f}")
        print(f"メタ認知指数: 有効N={df_meta_items.dropna().shape[0]}, α={alpha_meta:.3f}")
        
        return {"AI_support": alpha_ai, "Metacognition": alpha_meta}
    
    def run_all_analyses(self, save_figures=True):
        """全ての分析を実行"""
        print("\n" + "="*60)
        print("🔬 分析開始")
        print("="*60)
        
        # 指標定義
        self.define_indices()
        
        # 論文掲載分析
        self.summary_statistics()
        self.longitudinal_analysis(save_fig=save_figures)
        self.emotion_analysis()
        self.task_characteristics_analysis()
        
        # 探索的分析（論文未掲載）
        self.correlation_analysis()
        self.correspondence_analysis()
        self.reliability_analysis()
        
        print("\n" + "="*60)
        print("🔚 分析完了")
        print("="*60)
        print(f"""
✅ データ: {len(self.df_all)}件（4セッション）
✅ AI支援指数: {len(self.ai_support_items)}項目
✅ メタ認知指数: {len(self.metacog_items)}項目
""")


# ============================================================
# メイン実行部分
# ============================================================
if __name__ == "__main__":
    # 使用例
    print("""
    ============================================================
    使用方法
    ============================================================
    
    # 1. インスタンス作成
    analysis = GenAILearningAnalysis()
    
    # 2. データ読み込み
    file_paths = {
        5: '【回答】生成AIを活用した学習支援に関するアンケート（第5回）.xlsx',
        7: '【回答】生成AIを活用した学習支援に関するアンケート（第7回：人口統計演習）.xlsx',
        9: '【回答】生成AIを活用した学習支援に関するアンケート（第9回：労働統計演習）.xlsx',
        11: '【回答】生成AIを活用した学習支援に関するアンケート（第11回：賃金統計演習）.xlsx',
    }
    analysis.load_data(file_paths)
    
    # 3. 全分析実行
    analysis.run_all_analyses()
    
    # または個別に実行
    analysis.define_indices()
    analysis.summary_statistics()      # 表2
    analysis.longitudinal_analysis()   # 図1
    analysis.emotion_analysis()        # 表3
    analysis.task_characteristics_analysis()  # 表5
    """)
