# 統計演習における生成AI学習支援の分析コード
# Analysis Code for Generative AI Learning Support in Statistics Exercises

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要 / Overview

本リポジトリは、以下の論文で使用した分析コードを公開しています。

**論文タイトル**: 内なる他者との対話：生成AIを用いた統計演習における学習支援の実践報告

**Title**: Dialogue with the Inner Other: A Practical Report on Learning Support Using Generative AI in Statistics Exercises

**掲載誌**: リメディアル教育研究（日本リメディアル教育学会）

## ファイル構成 / File Structure

```
.
├── README.md                          # 本ファイル
├── analysis_main.py                   # メイン分析コード
├── analysis_task_characteristics.py   # 課題特性分析コード
├── requirements.txt                   # 必要ライブラリ
└── figures/                           # 出力図
    ├── fig1_longitudinal_ai_support.png  # 縦断変化図
    ├── fig2_emotion_heatmap.png          # 感情ヒートマップ
    └── fig3_task_characteristics.png     # 課題特性分析図（実行時生成）
```

## 必要環境 / Requirements

- Python 3.10+
- Google Colab（推奨）または Jupyter Notebook

### ライブラリ / Libraries

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

## 使用方法 / Usage

### 1. メイン分析（analysis_main.py）

```python
# ライブラリのインポート
from analysis_main import *

# データ読み込み
file_paths = {
    5: 'session5.csv',
    7: 'session7.csv',
    9: 'session9.csv',
    11: 'session11.csv'
}
df = load_survey_data(file_paths)

# 指数算出
df = compute_indices(df)

# 分析実行
run_analysis(df, output_dir='./output')
```

### 2. 課題特性分析（analysis_task_characteristics.py）

```python
# df_all が読み込まれた状態で実行
exec(open('analysis_task_characteristics.py').read())
```

## 分析内容 / Analysis Contents

### メイン分析
- **セッション別要約統計量**: AI支援指数・メタ認知指数の平均・標準偏差
- **感情プロファイル**: Plutchikの8基本感情の強度分析
- **縦断追跡分析**: 複数回参加者の軌跡分析

### 課題特性分析
- **AI利用モード分類**: 探索・コード・振り返りモードの自動分類
- **課題特性フラグ**: 自由記述からのキーワード抽出
- **群間比較**: 課題特性言及あり/なし群の指数比較

## 出力ファイル / Output Files

| ファイル名 | 内容 | 論文での使用 |
|-----------|------|-------------|
| fig1_longitudinal_ai_support.png | 縦断追跡参加者のAI支援指数の推移 | 図1 |
| fig2_emotion_heatmap.png | セッション別感情プロファイル | 図2 |
| fig3_task_characteristics.png | 課題特性フラグ別の指数比較 | 図3（任意） |

### 図の説明

**図1**: 縦断追跡可能な3名の個人軌跡（灰線）と平均（赤線）

**図2**: Plutchikの8基本感情の強度をヒートマップで可視化

**図3**: 課題特性（コード実装・プロンプト設計・データ選択）への言及有無による指数の比較

## 引用 / Citation

本コードを使用する場合は、以下の論文を引用してください。

```
（著者名）. (2026). 内なる他者との対話：生成AIを用いた統計演習における学習支援の実践報告. 
リメディアル教育研究, XX(X), XX-XX.
```

## ライセンス / License

MIT License

## 連絡先 / Contact

質問やフィードバックは、GitHubのIssueまたは論文記載の連絡先までお願いします。
