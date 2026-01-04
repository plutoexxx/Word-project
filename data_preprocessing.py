import pandas as pd
import numpy as np
import jieba
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import os

# 下载nltk数据（如果还没有下载）
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TextPreprocessor:
    def __init__(self, language='chinese'):
        self.language = language
        self.chinese_stopwords = self.load_chinese_stopwords()
        self.english_stopwords = set(stopwords.words('english'))

    def load_chinese_stopwords(self):
        """加载中文停用词表"""
        stopwords_path = 'hit_stopwords.txt'
        if os.path.exists(stopwords_path):
            with open(stopwords_path, 'r', encoding='utf-8') as f:
                return set([line.strip() for line in f])
        else:
            # 如果没有哈工大停用词表，使用默认中文停用词
            return set(
                ['的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到',
                 '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])

    def clean_text(self, text):
        """文本清洗"""
        if self.language == 'chinese':
            # 中文文本清洗
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
        else:
            # 英文文本清洗
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.lower()
        return text.strip()

    def tokenize(self, text):
        """分词"""
        if self.language == 'chinese':
            words = jieba.lcut(text)
        else:
            words = word_tokenize(text)
        return words

    def remove_stopwords(self, words):
        """去除停用词"""
        if self.language == 'chinese':
            return [word for word in words if word not in self.chinese_stopwords and len(word) > 1]
        else:
            return [word for word in words if word not in self.english_stopwords and len(word) > 2]

    def preprocess_text(self, text):
        """完整的文本预处理流程"""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self.clean_text(text)
        words = self.tokenize(text)
        words = self.remove_stopwords(words)
        return ' '.join(words)

    def preprocess_dataset(self, texts, labels=None):
        """预处理整个数据集"""
        processed_texts = []
        for text in texts:
            processed_text = self.preprocess_text(text)
            if processed_text:  # 只保留非空文本
                processed_texts.append(processed_text)

        if labels is not None:
            # 确保文本和标签长度一致
            filtered_labels = [labels[i] for i, text in enumerate(texts) if self.preprocess_text(text)]
            return processed_texts, filtered_labels
        return processed_texts


class FeatureExtractor:
    def __init__(self, method='tfidf', language='chinese'):
        self.method = method
        self.language = language
        self.vectorizer = None
        self.w2v_model = None

    def fit_tfidf(self, texts, ngram_range=(1, 2), max_features=5000):
        """训练TF-IDF向量化器"""
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=2,
            max_df=0.8
        )
        return self.vectorizer.fit(texts)

    def transform_tfidf(self, texts):
        """转换文本为TF-IDF特征"""
        if self.vectorizer is None:
            raise ValueError("TF-IDF向量化器尚未训练")
        return self.vectorizer.transform(texts)

    def train_word2vec(self, texts, vector_size=100, window=5, min_count=2):
        """训练Word2Vec模型"""
        tokenized_texts = [text.split() for text in texts]
        self.w2v_model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=4
        )
        return self.w2v_model

    def text_to_w2v(self, text):
        """将文本转换为Word2Vec向量"""
        if self.w2v_model is None:
            raise ValueError("Word2Vec模型尚未训练")

        words = text.split()
        word_vectors = []
        for word in words:
            if word in self.w2v_model.wv:
                word_vectors.append(self.w2v_model.wv[word])

        if len(word_vectors) == 0:
            return np.zeros(self.w2v_model.vector_size)

        return np.mean(word_vectors, axis=0)

    def texts_to_w2v(self, texts):
        """将多个文本转换为Word2Vec向量"""
        return np.array([self.text_to_w2v(text) for text in texts])