import joblib
import numpy as np
from data_preprocessing import TextPreprocessor
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import os


class SentimentPredictor:
    def __init__(self, language='chinese', model_path='models/'):
        self.language = language
        self.preprocessor = TextPreprocessor(language)

        # 加载模型
        self.traditional_models = None
        self.feature_extractor = None
        self.lstm_model = None
        self.lstm_tokenizer = None

        try:
            self.traditional_models = joblib.load(f'{model_path}/traditional_models.pkl')
            self.feature_extractor = joblib.load(f'{model_path}/feature_extractor.pkl')
            print(f"传统模型加载成功: {model_path}")
        except Exception as e:
            print(f"传统模型加载失败: {e}")

        # 加载LSTM模型
        try:
            self.lstm_model = load_model(f'{model_path}/lstm_model.h5')
            # 加载tokenizer
            with open(f'{model_path}/tokenizer.pkl', 'rb') as f:
                self.lstm_tokenizer = pickle.load(f)
            print(f"LSTM模型加载成功: {model_path}")
        except Exception as e:
            print(f"LSTM模型加载失败: {e}")

    def predict_traditional(self, text, model_name='ensemble'):
        """使用传统模型预测"""
        if self.traditional_models is None or self.feature_extractor is None:
            return None, None

        processed_text = self.preprocessor.preprocess_text(text)
        if not processed_text:
            return None, None

        # 特征提取
        try:
            if hasattr(self.feature_extractor.vectorizer, 'transform'):
                features = self.feature_extractor.transform_tfidf([processed_text])
            else:
                features = self.feature_extractor.texts_to_w2v([processed_text])
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None, None

        # 预测
        try:
            prediction = self.traditional_models.predict(model_name, features)[0]
            probability = self.traditional_models.predict_proba(model_name, features)[0]
            confidence = probability[prediction]
            return prediction, confidence
        except Exception as e:
            print(f"传统模型预测失败: {e}")
            return None, None

    def predict_lstm(self, text):
        """使用LSTM模型预测"""
        if self.lstm_model is None or self.lstm_tokenizer is None:
            return None, None

        processed_text = self.preprocessor.preprocess_text(text)
        if not processed_text:
            return None, None

        try:
            # 准备文本
            sequences = self.lstm_tokenizer.texts_to_sequences([processed_text])
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            X = pad_sequences(sequences, maxlen=100)

            # 预测
            prediction = self.lstm_model.predict(X, verbose=0)
            binary_prediction = (prediction > 0.5).astype(int).flatten()[0]
            confidence = prediction.flatten()[0] if binary_prediction == 1 else 1 - prediction.flatten()[0]

            return binary_prediction, confidence
        except Exception as e:
            print(f"LSTM模型预测失败: {e}")
            return None, None

    def predict_ensemble(self, text):
        """集成预测"""
        results = {}

        # 传统模型预测
        if self.traditional_models:
            nb_pred, nb_conf = self.predict_traditional(text, 'naive_bayes')
            svm_pred, svm_conf = self.predict_traditional(text, 'svm')
            ensemble_pred, ensemble_conf = self.predict_traditional(text, 'ensemble')

            if nb_pred is not None:
                results['naive_bayes'] = (nb_pred, nb_conf)
            if svm_pred is not None:
                results['svm'] = (svm_pred, svm_conf)
            if ensemble_pred is not None:
                results['ensemble'] = (ensemble_pred, ensemble_conf)

        # LSTM预测
        if self.lstm_model:
            lstm_pred, lstm_conf = self.predict_lstm(text)
            if lstm_pred is not None:
                results['lstm'] = (lstm_pred, lstm_conf)

        return results


def main():
    # 命令行预测接口
    predictor = SentimentPredictor(language='chinese', model_path='chinese_models/')

    print("=== 中文情感分析预测系统 ===")
    print("输入 'quit' 退出程序")

    while True:
        text = input("\n请输入要分析的文本: ")

        if text.lower() == 'quit':
            break

        if not text.strip():
            continue

        results = predictor.predict_ensemble(text)

        print(f"\n文本: {text}")
        print("-" * 50)

        if not results:
            print("无法进行情感分析，请检查模型是否已训练")
            continue

        for model_name, (prediction, confidence) in results.items():
            if prediction is not None and confidence is not None:
                sentiment = "正面" if prediction == 1 else "负面"
                print(f"{model_name:15} | 情感: {sentiment:5} | 置信度: {confidence:.4f}")
            else:
                print(f"{model_name:15} | 预测失败")


if __name__ == "__main__":
    main()