import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings

warnings.filterwarnings('ignore')


class TraditionalModels:
    def __init__(self):
        self.models = {}
        self.vectorizer = None

    def train_naive_bayes(self, X_train, y_train, alpha=1.0):
        """训练朴素贝叶斯模型"""
        nb_model = MultinomialNB(alpha=alpha)
        nb_model.fit(X_train, y_train)
        self.models['naive_bayes'] = nb_model
        return nb_model

    def train_svm(self, X_train, y_train, C=1.0, kernel='linear'):
        """训练SVM模型"""
        svm_model = SVC(C=C, kernel=kernel, probability=True, random_state=42, verbose=True)
        svm_model.fit(X_train, y_train)
        self.models['svm'] = svm_model
        return svm_model

    def create_ensemble(self, X_train, y_train):
        """创建集成模型"""
        if 'naive_bayes' not in self.models or 'svm' not in self.models:
            raise ValueError("需要先训练朴素贝叶斯和SVM模型")

        estimators = [
            ('nb', self.models['naive_bayes']),
            ('svm', self.models['svm'])
        ]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        self.models['ensemble'] = ensemble
        return ensemble

    def predict(self, model_name, X):
        """预测"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        return self.models[model_name].predict(X)

    def predict_proba(self, model_name, X):
        """预测概率"""
        if model_name not in self.models:
            raise ValueError(f"模型 {model_name} 不存在")
        return self.models[model_name].predict_proba(X)


class LSTMModel:
    def __init__(self, vocab_size=10000, max_length=200, embedding_dim=100):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model = None
        self.tokenizer = None

    def build_model(self):
        """构建LSTM模型"""
        self.model = Sequential([
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_length),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return self.model

    def prepare_texts(self, texts):
        """准备文本数据"""
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=self.vocab_size)
            self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_length)

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """训练模型"""
        if self.model is None:
            self.build_model()

        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        return history

    def predict(self, texts):
        """预测"""
        if not texts:
            return np.array([]), np.array([])

        X = self.prepare_texts(texts)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten(), predictions.flatten()


class ModelEvaluator:
    @staticmethod
    def evaluate_model(y_true, y_pred, y_proba=None):
        """评估模型性能"""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }

        accuracy = accuracy_score(y_true, y_pred)

        # 处理只有一个类别的情况
        if len(set(y_true)) == 1:
            precision = 1.0 if y_pred[0] == y_true[0] else 0.0
            recall = 1.0 if y_pred[0] == y_true[0] else 0.0
            f1 = 1.0 if y_pred[0] == y_true[0] else 0.0
        else:
            precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        return metrics

    @staticmethod
    def print_metrics(metrics, model_name):
        """打印评估指标"""
        print(f"\n=== {model_name} 性能评估 ===")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1_score']:.4f}")