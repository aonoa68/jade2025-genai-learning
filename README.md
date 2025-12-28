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
├── README.md                 # 本ファイル
├── analysis_main.py          # メイン分析コード（全分析を含む）
├── requirements.txt          # 必要ライブラリ
├── LICENSE                   # MITライセンス
└── figures/                  # 出力図
    └── fig1_longitudinal_ai_support_mono.png  # 縦断変化図（図1）
```

## 分析内容 / Analysis Contents

### 論文掲載分析

| 分析 | 内容 | 論文での対応 |
|------|------|-------------|
| 基本集計 | AI支援指数・メタ認知指数のセッション別平均・SD | 表2 |
| 縦断変化分析 | 複数回参加者（N=3）の軌跡分析 | 図1 |
| 感情プロファイル | Plutchikの8基本感情の強度分析 | 表3 |
| 課題特性分析 | 自由記述からのキーワード抽出と群間比較 | 表5 |

### 探索的分析（論文未掲載）

以下の分析は実施しましたが、データの制約により論文本文には含めていません。
分析過程の透明性のため、コードを公開しています。

| 分析 | 未掲載の理由 |
|------|-------------|
| 感情×学習指数の相関分析 | ネガティブ感情に床効果（65-75%が「感じなかった」）、自己選択バイアスの影響 |
| 対応分析 | パターンが不明瞭、セッション間の違いが主に反映 |
| 信頼性係数（Cronbach's α） | セッションごとに質問項目が異なるため算出不可 |

## 必要環境 / Requirements

- Python 3.10+
- Google Colab（推奨）または Jupyter Notebook

### 必要ライブラリ / Libraries

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
japanize_matplotlib>=1.1.0
prince>=0.7.0  # 対応分析用（探索的分析のみ）
```

インストール:
```bash
pip install -r requirements.txt
```

## 使用方法 / Usage

### Google Colabでの実行（推奨）

```python
# 1. ライブラリのインストール
!pip install japanize_matplotlib prince

# 2. analysis_main.py をアップロード後、インポート
from analysis_main import GenAILearningAnalysis

# 3. インスタンス作成
analysis = GenAILearningAnalysis()

# 4. データファイルをアップロード
from google.colab import files
uploaded = files.upload()

# 5. データ読み込み
file_paths = {
    5: '【回答】生成AIを活用した学習支援に関するアンケート（第5回）.xlsx',
    7: '【回答】生成AIを活用した学習支援に関するアンケート（第7回：人口統計演習）.xlsx',
    9: '【回答】生成AIを活用した学習支援に関するアンケート（第9回：労働統計演習）.xlsx',
    11: '【回答】生成AIを活用した学習支援に関するアンケート（第11回：賃金統計演習）.xlsx',
}
analysis.load_data(file_paths)

# 6. 全分析実行
analysis.run_all_analyses()
```

### 個別分析の実行

```python
# 指標の定義（必須）
analysis.define_indices()

# 表2：セッション別要約統計量
analysis.summary_statistics()

# 図1：縦断変化分析
analysis.longitudinal_analysis()

# 表3：感情プロファイル
analysis.emotion_analysis()

# 表5：課題特性分析
analysis.task_characteristics_analysis()

# 探索的分析（論文未掲載）
analysis.correlation_analysis()      # 感情×学習指数の相関
analysis.correspondence_analysis()   # 対応分析
analysis.reliability_analysis()      # 信頼性係数
```

## 出力ファイル / Output Files

| ファイル名 | 内容 | 論文での使用 |
|-----------|------|-------------|
| fig1_longitudinal_ai_support_mono.png | 縦断追跡参加者のAI支援指数の推移（モノクロ版） | 図1 |

### 図1の説明

- 縦断追跡可能な3名の個人軌跡（Case 1-3）
- 追跡可能者の平均（◆）
- 全体平均（×）
- モノクロ印刷対応（グレースケール＋異なるマーカー・線種）

## データについて / About Data

本研究で使用したアンケートデータは、個人情報保護の観点から公開していません。
コードの動作確認には、同様の形式のデータをご用意ください。

### 必要なデータ形式

- Excel形式（.xlsx）
- 必須列：ID列（"ID"を含む列名）
- AI支援項目：「生成AI」を含み、「役立」「理解」「整理」のいずれかを含む列
- メタ認知項目：「説明できる」「復習」「良い質問」「良い問い」のいずれかを含む列
- 感情項目：「感情」を含み、[感情名]形式の列

## 研究の限界 / Limitations

本研究には以下の限界があります（論文4.4節参照）：

1. **自己選択バイアス**: 回答率が約4〜15%（243名中10〜36名）と低く、AIをうまく活用できた学習者が回答しやすい傾向がある
2. **縦断追跡の限界**: 追跡可能なサンプルが3名に限定
3. **床効果**: ネガティブ感情の65〜75%が「感じなかった」と回答

## 引用 / Citation

本コードを使用する場合は、以下の論文を引用してください。

```
小野原彩香. (2026). 内なる他者との対話：生成AIを用いた統計演習における学習支援の実践報告. 
リメディアル教育研究, XX(X), XX-XX.
```

## ライセンス / License

MIT License

## 連絡先 / Contact

質問やフィードバックは、GitHubのIssueまたは論文記載の連絡先までお願いします。

## 更新履歴 / Changelog

- **2025-12-29**: 探索的分析（相関・対応分析・信頼性係数）を追加、コードをクラス化
- **2025-12-25**: 初版公開
