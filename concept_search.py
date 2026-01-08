# Copyright (c) 2026 tuanzi. All rights reserved.
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pickle
import os

# --- 1. 领域知识库 (同义词与翻译) ---
class DomainDictionary:
    """
    领域知识字典，提供查询扩展功能。
    包含常用 AI 术语的同义词和相关词。
    """
    def __init__(self):
        """
        初始化领域字典，定义同义词映射。
        """
        # 键可以是中文或英文缩写，值是扩展的英文同义词列表
        self.synonyms = {
            # 大模型相关
            "llm": ["large language model", "foundation model", "gpt", "generative pre-trained"],
            "大模型": ["large language model", "llm", "foundation model"],
            "语言模型": ["language model"],
            
            # 架构
            "transformer": ["self-attention", "multi-head attention", "vision transformer", "vit"],
            "attention": ["attention mechanism"],
            "注意力": ["attention"],
            
            # 生成式 AI
            "diffusion": ["score-based generative model", "ddpm", "latent diffusion", "stable diffusion"],
            "扩散模型": ["diffusion model", "score-based"],
            "gan": ["generative adversarial network"],
            "生成对抗网络": ["generative adversarial network", "gan"],
            
            # 强化学习
            "rl": ["reinforcement learning", "q-learning", "policy gradient", "actor-critic"],
            "强化学习": ["reinforcement learning", "rl"],
            "dr": ["deep reinforcement learning"],
            
            # 视觉
            "cv": ["computer vision", "image recognition", "object detection"],
            "计算机视觉": ["computer vision"],
            "cnn": ["convolutional neural network", "convnet"],
            "卷积神经网络": ["convolutional neural network"],
            "nerf": ["neural radiance fields", "view synthesis"],
            
            # 优化与训练
            "optimization": ["sgd", "adam", "gradient descent"],
            "优化": ["optimization"],
            "finetuning": ["fine-tuning", "transfer learning", "adaptation"],
            "微调": ["fine-tuning"],
            
            # 其他
            "graph": ["gnn", "graph neural network"],
            "图神经网络": ["graph neural network", "gnn"],
            "multimodal": ["vision-language", "cross-modal"],
            "多模态": ["multimodal", "vision-language"],
        }

    def expand_query(self, query):
        """
        扩展查询：
        1. 识别查询中的关键词
        2. 如果命中字典，加入同义词
        3. 构造 "term OR synonym1 OR synonym2" 形式的字符串 (用于 TF-IDF 权重增强)
        
        Args:
            query (str): 原始查询字符串。
            
        Returns:
            str: 扩展后的查询字符串。
        """
        query_lower = query.lower().strip()
        expanded_terms = [query_lower]
        
        # 简单的全字匹配扩展
        # 也可以改进为正则匹配
        for key, values in self.synonyms.items():
            # 如果查询包含 key (简单的字符串包含，实际可能需要分词)
            if key in query_lower:
                expanded_terms.extend(values)
        
        # 去重
        unique_terms = list(set(expanded_terms))
        return " ".join(unique_terms)

# --- 2. 概念搜索引擎 (TF-IDF) ---
class ConceptSearchEngine:
    """
    基于 TF-IDF 和余弦相似度的概念搜索引擎。
    """
    def __init__(self, df):
        """
        初始化搜索引擎。
        
        Args:
            df (pd.DataFrame): 包含论文数据的 DataFrame。
        """
        self.df = df
        self.domain_dict = DomainDictionary()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._initialize_index()

    def _initialize_index(self):
        """
        初始化或加载 TF-IDF 索引。
        """
        # 简单的缓存路径
        cache_dir = ".cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # 既然是 Streamlit 应用，我们通常利用 st.cache_resource 来管理这个实例，
        # 但为了类内部独立性，我们在这里做一次拟合。
        # 实际生产中建议保存 vectorizer 到 disk，这里为了简化直接内存计算 (5万条数据 fit 很快)
        
        print("Building TF-IDF Index...")
        # 组合标题和摘要作为语料
        corpus = self.df['title'].fillna('') + " " + self.df['abstract'].fillna('')
        
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print("Index Built.")

    def search(self, query):
        """
        执行语义搜索。
        
        Args:
            query (str): 查询字符串。
            
        Returns:
            pd.DataFrame: 包含 'relevance' 分数列的 DataFrame (未筛选)。
        """
        if not query or not query.strip():
            return self.df.copy()
            
        # 1. 查询扩展
        expanded_query = self.domain_dict.expand_query(query)
        # print(f"Original: {query} -> Expanded: {expanded_query}") # Debug
        
        # 2. 向量化查询
        query_vec = self.vectorizer.transform([expanded_query])
        
        # 3. 计算相似度
        # cosine_similarity 返回 shape (1, n_docs)
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # 4. 附加分数
        result_df = self.df.copy()
        result_df['relevance'] = scores
        
        # 5. 过滤掉相关度过低的结果 (例如 < 0.05) 以减少噪音
        # 但为了用户能看到东西，我们暂不硬性过滤，而是交给前端排序
        return result_df

    def get_similar_papers(self, paper_index, top_k=5):
        """
        获取相似论文。
        
        Args:
            paper_index (int or str): DataFrame 中的 index。
            top_k (int): 返回的相似论文数量。默认为 5。
            
        Returns:
            pd.DataFrame: 相似论文 DataFrame。
        """
        if paper_index not in self.df.index:
            return pd.DataFrame()
            
        # 获取目标论文的向量 (使用 iloc 位置索引，因为 tfidf_matrix 是 numpy 矩阵)
        # 需要确保 df 的 index 是连续的 range(0, N)，如果不是，需要映射
        # 这里假设 df 是 reset_index 过的，index 从 0 到 N-1
        
        # 为了安全，我们通过 row number 获取
        try:
            row_loc = self.df.index.get_loc(paper_index)
        except KeyError:
            return pd.DataFrame()
            
        target_vec = self.tfidf_matrix[row_loc]
        
        # 计算相似度
        scores = cosine_similarity(target_vec, self.tfidf_matrix).flatten()
        
        # 获取 top k (排除自己)
        # argsort 返回从小到大的索引，取最后 k+1 个
        top_indices = scores.argsort()[-(top_k+1):][::-1]
        
        similar_papers = []
        for idx in top_indices:
            # 排除自己 (通过分数判断或者 index 判断)
            # 浮点数比较 safe: 分数接近 1.0 且 index 相同
            if idx == row_loc:
                continue
                
            # 从原始 df 获取数据
            # 注意：tfidf_matrix 的行号 idx 对应 df 的第 idx 行
            paper_data = self.df.iloc[idx].to_dict()
            paper_data['relevance'] = scores[idx]
            similar_papers.append(paper_data)
            
            if len(similar_papers) >= top_k:
                break
                
        return pd.DataFrame(similar_papers)
