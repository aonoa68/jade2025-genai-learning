"""
統計演習における生成AI学習支援の分析コード
Analysis Code for Generative AI Learning Support in Statistics Exercises

著者: ■■■
論文: 内なる他者との対話：生成AIを用いた統計演習における学習支援の実践報告

使用環境: Google Colab / Python 3.10+
必要ライブラリ: pandas, numpy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =============================================================================
# 設定
# =============================================================================

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 感情ラベル（Plutchikの8基本感情）
EMOTION_LABELS = {
    'ja': ['期待', '驚き', '喜び', '信頼', '怒り', '嫌悪', '悲しみ', '恐れ'],
    'en': ['Anticipation', 'Surprise', 'Joy', 'Trust', 'Anger', 'Disgust', 'Sadness', 'Fear']
}

# セッション情報
SESSION_INFO = {
    5: "地理・移動平均",
    7: "人口統計",
    9: "労働統計",
    11: "賃金統計"
}

# =============================================================================
# データ読み込み・前処理
# =============================================================================

def load_survey_data(file_paths: dict) -> pd.DataFrame:
    """
    複数のアンケートCSVを読み込み、統合する
    
    Parameters:
        file_paths: {session_number: file_path} の辞書
    
    Returns:
        統合されたDataFrame
    """
    dfs = []
    for session, path in file_paths.items():
        df = pd.read_csv(path)
        df['session'] = session
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def find_column(df: pd.DataFrame, keywords: list) -> str:
    """キーワードを含む列名を検索"""
    for col in df.columns:
        if any(kw in col for kw in keywords):
            return col
    return None


def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    AI支援指数とメタ認知指数を算出
    
    AI支援指数: 6項目の平均
    メタ認知指数: 9項目の平均
    """
    # AI支援関連の列を特定
    ai_keywords = ['理解が整理', '不足点', '気づき', 'フィードバック', '対話', '支援']
    ai_cols = [c for c in df.columns if any(kw in c for kw in ai_keywords)]
    
    # メタ認知関連の列を特定
    meta_keywords = ['達成度', '説明できる', '復習', '質問', '目標', '振り返り', '計画', '理解度', '自己評価']
    meta_cols = [c for c in df.columns if any(kw in c for kw in meta_keywords)]
    
    # 指数算出
    if ai_cols:
        df['AI_support'] = df[ai_cols].mean(axis=1)
    if meta_cols:
        df['Metacognition'] = df[meta_cols].mean(axis=1)
    
    return df


def extract_emotions(df: pd.DataFrame) -> pd.DataFrame:
    """感情列を抽出・整理"""
    emotion_cols = []
    for label in EMOTION_LABELS['ja']:
        col = find_column(df, [label])
        if col:
            emotion_cols.append(col)
    
    if emotion_cols:
        df['emotions'] = df[emotion_cols].values.tolist()
    
    return df


# =============================================================================
# 分析関数
# =============================================================================

def summarize_by_session(df: pd.DataFrame) -> pd.DataFrame:
    """セッション別の要約統計量"""
    summary = df.groupby('session').agg({
        'AI_support': ['mean', 'std', 'count'],
        'Metacognition': ['mean', 'std', 'count']
    }).round(2)
    return summary


def compute_emotion_profile(df: pd.DataFrame) -> pd.DataFrame:
    """セッション別の感情プロファイル"""
    emotion_cols = [c for c in df.columns if any(e in c for e in EMOTION_LABELS['ja'])]
    
    profile = df.groupby('session')[emotion_cols].mean().round(2)
    profile.columns = EMOTION_LABELS['en']
    
    return profile


def identify_longitudinal_participants(df: pd.DataFrame, id_col: str = None) -> pd.DataFrame:
    """縦断追跡可能な参加者を特定"""
    if id_col is None:
        id_col = find_column(df, ['ID', 'id', '学籍'])
    
    if id_col is None:
        print("Warning: ID column not found")
        return pd.DataFrame()
    
    # 複数回参加者を抽出
    participation = df.groupby(id_col)['session'].agg(['count', list])
    longitudinal = participation[participation['count'] > 1]
    longitudinal.columns = ['参加数', '参加回']
    
    return longitudinal


# =============================================================================
# 可視化関数
# =============================================================================

def plot_session_summary(df: pd.DataFrame, save_path: str = None):
    """セッション別指数の推移をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # AI支援指数
    session_means = df.groupby('session')['AI_support'].mean()
    session_stds = df.groupby('session')['AI_support'].std()
    
    axes[0].bar(session_means.index, session_means.values, 
                yerr=session_stds.values, capsize=5, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Session')
    axes[0].set_ylabel('AI Support Index')
    axes[0].set_title('AI Support Index by Session')
    axes[0].set_ylim(0, 5.5)
    axes[0].set_xticks(list(SESSION_INFO.keys()))
    
    # メタ認知指数
    meta_means = df.groupby('session')['Metacognition'].mean()
    meta_stds = df.groupby('session')['Metacognition'].std()
    
    axes[1].bar(meta_means.index, meta_means.values,
                yerr=meta_stds.values, capsize=5, color='coral', alpha=0.7)
    axes[1].set_xlabel('Session')
    axes[1].set_ylabel('Metacognition Index')
    axes[1].set_title('Metacognition Index by Session')
    axes[1].set_ylim(0, 5.5)
    axes[1].set_xticks(list(SESSION_INFO.keys()))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_emotion_heatmap(df: pd.DataFrame, save_path: str = None):
    """感情プロファイルのヒートマップ"""
    profile = compute_emotion_profile(df)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.heatmap(profile, annot=True, fmt='.2f', cmap='RdYlGn',
                vmin=0, vmax=3, ax=ax, cbar_kws={'label': 'Intensity (0-3)'})
    
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Session')
    ax.set_title('Emotion Profile by Session (Plutchik\'s 8 Basic Emotions)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def plot_longitudinal_trajectory(df: pd.DataFrame, id_col: str, 
                                  target_ids: list = None, save_path: str = None):
    """縦断追跡参加者の軌跡をプロット"""
    if target_ids is None:
        longitudinal = identify_longitudinal_participants(df, id_col)
        target_ids = longitudinal.index.tolist()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 個人の軌跡（灰色）
    for pid in target_ids:
        person_data = df[df[id_col] == pid].sort_values('session')
        ax.plot(person_data['session'], person_data['AI_support'],
                color='gray', alpha=0.5, linewidth=2, marker='o', markersize=6)
    
    # 平均線（赤）
    if len(target_ids) > 0:
        longitudinal_df = df[df[id_col].isin(target_ids)]
        means = longitudinal_df.groupby('session')['AI_support'].mean()
        ax.plot(means.index, means.values, color='red', linewidth=3,
                marker='s', markersize=8, label=f'Mean (N={len(target_ids)})')
    
    ax.set_xlabel('Session')
    ax.set_ylabel('AI Support Index')
    ax.set_title('Longitudinal Change in AI Support Index\n(gray: individual, red: mean)')
    ax.set_ylim(0, 5.5)
    ax.set_xticks(list(SESSION_INFO.keys()))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


# =============================================================================
# メイン実行
# =============================================================================

def run_analysis(df: pd.DataFrame, output_dir: str = './output'):
    """
    全分析を実行
    
    Parameters:
        df: 前処理済みのDataFrame（AI_support, Metacognition列を含む）
        output_dir: 出力ディレクトリ
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("統計演習における生成AI学習支援の分析")
    print("=" * 60)
    
    # 1. 基本統計量
    print("\n[1] セッション別要約統計量")
    print("-" * 40)
    summary = summarize_by_session(df)
    print(summary)
    
    # 2. 感情プロファイル
    print("\n[2] 感情プロファイル")
    print("-" * 40)
    profile = compute_emotion_profile(df)
    print(profile)
    
    # 3. 可視化
    print("\n[3] 可視化を生成中...")
    plot_session_summary(df, output_path / 'fig1_session_summary.png')
    plot_emotion_heatmap(df, output_path / 'fig2_emotion_heatmap.png')
    
    print("\n" + "=" * 60)
    print("分析完了")
    print(f"出力先: {output_path}")
    print("=" * 60)


# =============================================================================
# 使用例
# =============================================================================

if __name__ == "__main__":
    # サンプルデータでのテスト
    print("使用方法:")
    print("1. CSVファイルを読み込み: df = load_survey_data({5: 'session5.csv', ...})")
    print("2. 指数を算出: df = compute_indices(df)")
    print("3. 分析を実行: run_analysis(df)")
