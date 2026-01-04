import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from data_preprocessing import TextPreprocessor, FeatureExtractor
from models import TraditionalModels, LSTMModel, ModelEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import sys
import warnings

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

warnings.filterwarnings('ignore')


class SentimentAnalysisTrainer:
    def __init__(self, language='chinese', data_dir='./'):
        self.language = language
        self.data_dir = data_dir
        self.preprocessor = TextPreprocessor(language)
        self.feature_extractor = FeatureExtractor(language=language)
        self.traditional_models = TraditionalModels()
        self.lstm_model = None
        self.evaluator = ModelEvaluator()

    def load_dataset_from_csv(self, csv_filename, text_column='review', label_column='label'):
        """
        从CSV文件加载数据集，自动适配中文酒店评论和IMDb电影评论格式。
        返回：原始文本列表，标签列表
        """
        csv_path = os.path.join(self.data_dir, csv_filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据集文件未找到: {csv_path}")

        print(f"正在从 {csv_filename} 加载数据集...")
        df = pd.read_csv(csv_path)
        print(f"数据集形状: {df.shape}")
        print(f"数据列名: {list(df.columns)}")

        # 自动检测列名
        if text_column not in df.columns:
            # 尝试常见的中文列名
            chinese_text_cols = ['review', 'text', 'comment', '评论', '内容']
            for col in chinese_text_cols:
                if col in df.columns:
                    text_column = col
                    print(f"自动检测到文本列: {text_column}")
                    break

        if label_column not in df.columns:
            # 尝试常见的标签列名
            label_cols = ['label', 'sentiment', 'emotion', '评分', '情感']
            for col in label_cols:
                if col in df.columns:
                    label_column = col
                    print(f"自动检测到标签列: {label_column}")
                    break

        texts = df[text_column].astype(str).fillna('').tolist()
        labels = df[label_column].tolist()

        # 标签格式统一化
        if isinstance(labels[0], str):
            print(f"正在转换标签格式（检测到字符串标签）...")
            label_mapping = {'positive': 1, 'negative': 0, '正面': 1, '负面': 0, '1': 1, '0': 0}
            labels = [label_mapping.get(str(label).lower().strip(), 0) for label in labels]

        # 确保标签是整数
        labels = [int(label) for label in labels]

        # 检查数据平衡性
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"正面样本: {pos_count} 条，负面样本: {neg_count} 条")
        print(f"数据加载完成，共 {len(texts)} 条记录")

        return texts, labels

    def load_sample_data(self):
        """
        根据语言加载相应的数据集
        """
        if self.language == 'chinese':
            # 中文酒店评论数据集
            try:
                texts, labels = self.load_dataset_from_csv(
                    'ChnSentiCorp_htl_all.csv',
                    text_column='review',
                    label_column='label'
                )
                return texts, labels
            except FileNotFoundError:
                print("警告：未找到中文数据集文件，使用内置示例数据")
                return self._get_sample_chinese_data()
        else:
            # 英文IMDb电影评论数据集
            try:
                texts, labels = self.load_dataset_from_csv(
                    'IMDB Dataset.csv',
                    text_column='review',
                    label_column='sentiment'
                )
                return texts, labels
            except FileNotFoundError:
                print("警告：未找到英文数据集文件，使用内置示例数据")
                return self._get_sample_english_data()

    def _get_sample_chinese_data(self):
        """中文示例数据"""
        texts = [
            "这个产品非常好用，质量很棒！",
            "非常失望，产品质量很差",
            "性价比高，推荐购买",
            "完全不值得这个价格",
            "服务态度很好，物流很快",
            "包装破损，体验很差",
            "物超所值，非常满意",
            "质量一般，没有想象中好",
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        return texts, labels

    def _get_sample_english_data(self):
        """英文示例数据"""
        texts = [
            "This movie is fantastic, great acting!",
            "Terrible movie, waste of time",
            "Amazing plot and characters",
            "Boring and poorly made",
            "One of the best films I've seen",
            "Disappointing and overrated",
            "Absolutely loved it, highly recommend",
            "Not as good as expected",
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        return texts, labels

    def train_traditional_models(self, texts, labels, feature_method='tfidf', test_size=0.2):
        """
        训练传统机器学习模型（朴素贝叶斯、SVM、集成模型）
        修复：确保训练/测试集在预处理前分割
        """
        # 1. 先分割数据集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"传统模型训练集: {len(train_texts)} 条，测试集: {len(test_texts)} 条")

        # 2. 分别预处理训练集和测试集
        print("预处理训练集文本...")
        processed_train_texts, train_labels = self.preprocessor.preprocess_dataset(train_texts, train_labels)

        print("预处理测试集文本...")
        processed_test_texts, test_labels = self.preprocessor.preprocess_dataset(test_texts, test_labels)

        if len(processed_train_texts) == 0 or len(processed_test_texts) == 0:
            print("错误：预处理后没有有效文本")
            return {}, None, None, None, None

        # 3. 特征提取（只在训练集上拟合）
        print("特征提取（TF-IDF）...")
        self.feature_extractor.fit_tfidf(processed_train_texts, max_features=5000)

        X_train = self.feature_extractor.transform_tfidf(processed_train_texts)
        X_test = self.feature_extractor.transform_tfidf(processed_test_texts)

        # 4. 训练模型
        print("训练朴素贝叶斯模型...")
        nb_model = self.traditional_models.train_naive_bayes(X_train, train_labels)

        print("训练SVM模型...")
        svm_model = self.traditional_models.train_svm(X_train, train_labels, kernel='linear')

        print("训练集成模型...")
        ensemble_model = self.traditional_models.create_ensemble(X_train, train_labels)

        # 5. 评估模型
        models_to_evaluate = ['naive_bayes', 'svm', 'ensemble']
        results = {}

        for model_name in models_to_evaluate:
            y_pred = self.traditional_models.predict(model_name, X_test)
            y_proba = self.traditional_models.predict_proba(model_name, X_test)
            metrics = self.evaluator.evaluate_model(test_labels, y_pred, y_proba)
            self.evaluator.print_metrics(metrics, model_name)
            results[model_name] = metrics

        return results, X_train, X_test, train_labels, test_labels

    def train_lstm_model(self, texts, labels, test_size=0.2, vocab_size=5000, max_length=100):
        """
        训练LSTM模型 - 修复版本
        核心修复：先分割数据集，再分别准备序列数据，避免数据泄露
        """
        # 1. 先分割原始数据集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )

        print(f"LSTM训练集: {len(train_texts)} 条，测试集: {len(test_texts)} 条")

        # 2. 分别预处理训练集和测试集
        print("预处理训练集文本...")
        processed_train_texts, train_labels = self.preprocessor.preprocess_dataset(train_texts, train_labels)

        print("预处理测试集文本...")
        processed_test_texts, test_labels = self.preprocessor.preprocess_dataset(test_texts, test_labels)

        if len(processed_train_texts) == 0 or len(processed_test_texts) == 0:
            print("错误：预处理后没有有效文本")
            return {}, None

        # 3. 准备LSTM数据 - 关键修复：只在训练集上拟合Tokenizer
        print("准备LSTM序列数据（严格隔离训练集和测试集）...")
        self.lstm_model = LSTMModel(vocab_size=vocab_size, max_length=max_length, embedding_dim=100)

        # 只在训练集上拟合tokenizer
        self.lstm_model.tokenizer = Tokenizer(num_words=vocab_size)
        self.lstm_model.tokenizer.fit_on_texts(processed_train_texts)

        # 分别转换训练集和测试集
        X_train = self.lstm_model.prepare_texts(processed_train_texts)
        X_test = self.lstm_model.prepare_texts(processed_test_texts)

        y_train = np.array(train_labels)
        y_test = np.array(test_labels)

        # 4. 进一步分割出验证集
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )

        print(f"LSTM最终训练集: {len(X_train_final)} 条，验证集: {len(X_val)} 条，测试集: {len(X_test)} 条")

        # 5. 训练模型（添加早停）
        print("训练LSTM模型...")
        history = self.lstm_model.train(
            X_train_final, y_train_final,
            X_val=X_val, y_val=y_val,
            epochs=10,
            batch_size=32
        )

        # 6. 评估模型
        print("评估LSTM模型...")
        # 注意：这里使用与训练时相同的prepare_texts方法，但已经是测试集数据
        y_pred, y_proba = self.lstm_model.predict(processed_test_texts)
        metrics = self.evaluator.evaluate_model(y_test, y_pred, y_proba)
        self.evaluator.print_metrics(metrics, "LSTM")

        return metrics, history

    def save_models(self, model_dir='./models'):
        """保存训练好的模型"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"创建模型目录: {model_dir}")

        # 保存传统模型
        traditional_path = os.path.join(model_dir, 'traditional_models.pkl')
        joblib.dump(self.traditional_models, traditional_path)

        # 保存特征提取器
        feature_path = os.path.join(model_dir, 'feature_extractor.pkl')
        joblib.dump(self.feature_extractor, feature_path)

        # 保存LSTM模型
        if self.lstm_model and self.lstm_model.model:
            lstm_path = os.path.join(model_dir, 'lstm_model.h5')
            self.lstm_model.model.save(lstm_path)
            print(f"LSTM模型保存到: {lstm_path}")

        print(f"所有模型已保存到: {model_dir}")

    def plot_results(self, results, save_path='model_comparison.png'):
        """绘制结果对比图"""
        if not results:
            print("没有结果可绘制")
            return

        models = list(results.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for i, metric in enumerate(metrics_names):
            values = [results[model][metric] for model in models]

            bars = axes[i].bar(models, values, color=colors, edgecolor='black')
            axes[i].set_title(f'{metric.upper()} 对比', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(metric.capitalize(), fontsize=12)
            axes[i].set_ylim(0, 1.05)
            axes[i].tick_params(axis='x', rotation=45)

            # 在柱状图上显示数值
            for bar, v in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                             f'{v:.4f}', ha='center', va='bottom', fontsize=10)

            # 添加网格线
            axes[i].grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"性能对比图已保存为: {save_path}")
        plt.show()


def train_chinese_models():
    """训练中文情感分析模型"""
    print("=" * 60)
    print("开始训练中文情感分析模型")
    print("=" * 60)

    trainer = SentimentAnalysisTrainer(language='chinese')

    try:
        # 1. 加载数据
        texts, labels = trainer.load_sample_data()

        # 2. 训练传统模型
        print("\n" + "-" * 40)
        print("训练传统机器学习模型")
        print("-" * 40)
        traditional_results, _, _, _, _ = trainer.train_traditional_models(
            texts, labels, test_size=0.2
        )

        # 3. 训练LSTM模型
        print("\n" + "-" * 40)
        print("训练LSTM深度学习模型")
        print("-" * 40)
        lstm_results, _ = trainer.train_lstm_model(
            texts, labels,
            test_size=0.2,
            vocab_size=5000,
            max_length=100
        )

        # 4. 合并结果并可视化
        all_results = {**traditional_results, 'LSTM': lstm_results}
        trainer.plot_results(all_results, 'chinese_model_comparison.png')

        # 5. 保存模型
        trainer.save_models('chinese_models')

        print("\n✅ 中文模型训练完成！")
        return True

    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_english_models():
    """训练英文情感分析模型"""
    print("=" * 60)
    print("开始训练英文情感分析模型")
    print("=" * 60)

    trainer = SentimentAnalysisTrainer(language='english')

    try:
        # 1. 加载数据
        texts, labels = trainer.load_sample_data()

        # 对于大数据集，可以先采样一部分进行快速测试
        if len(texts) > 10000:
            print("数据集较大，采样10000条进行训练...")
            # 保持正负样本比例
            from sklearn.utils import resample
            sample_size = min(10000, len(texts))
            texts, labels = resample(texts, labels, n_samples=sample_size, random_state=42, stratify=labels)

        # 2. 训练传统模型
        print("\n" + "-" * 40)
        print("训练传统机器学习模型")
        print("-" * 40)
        traditional_results, _, _, _, _ = trainer.train_traditional_models(
            texts, labels, test_size=0.2
        )

        # 3. 训练LSTM模型
        print("\n" + "-" * 40)
        print("训练LSTM深度学习模型")
        print("-" * 40)
        lstm_results, _ = trainer.train_lstm_model(
            texts, labels,
            test_size=0.2,
            vocab_size=8000,
            max_length=150
        )

        # 4. 合并结果并可视化
        all_results = {**traditional_results, 'LSTM': lstm_results}
        trainer.plot_results(all_results, 'english_model_comparison.png')

        # 5. 保存模型
        trainer.save_models('english_models')

        print("\n✅ 英文模型训练完成！")
        return True

    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("文本情感分析模型训练系统")
    print("=" * 60)

    # 设置matplotlib中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 训练中文模型
    success_chinese = train_chinese_models()

    if success_chinese:
        print("\n" + "=" * 60)
        choice = input("是否继续训练英文模型? (y/n): ").strip().lower()

        if choice == 'y' or choice == 'yes':
            # 训练英文模型
            success_english = train_english_models()

            if success_english:
                print("\n🎉 所有模型训练完成！")
                print("中文模型保存在: chinese_models/")
                print("英文模型保存在: english_models/")
            else:
                print("\n⚠️ 英文模型训练失败，但中文模型已保存")
        else:
            print("\n✅ 中文模型训练完成！模型保存在: chinese_models/")
    else:
        print("\n❌ 模型训练失败，请检查错误信息")


if __name__ == "__main__":
    main()