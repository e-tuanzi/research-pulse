# Copyright (c) 2026 tuanzi. All rights reserved.
import pandas as pd
import glob
import os
import streamlit as st

class DataManager:
    """
    数据管理类，负责加载、清洗和筛选论文数据。
    """

    @staticmethod
    @st.cache_data
    def load_data(data_dir="ai_papers_data"):
        """
        加载所有论文数据并缓存。
        
        Args:
            data_dir (str): 数据文件所在的根目录。默认为 "ai_papers_data"。
            
        Returns:
            pd.DataFrame: 包含所有论文数据的 DataFrame。如果未找到数据，返回空的 DataFrame。
        """
        all_files = glob.glob(os.path.join(data_dir, "*", "*.csv"))
        if not all_files:
            return pd.DataFrame()
        
        df_list = []
        for filename in all_files:
            try:
                df = pd.read_csv(filename)
                # 确保必要的列存在
                if 'year' in df.columns:
                    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
                if 'citations' in df.columns:
                    df['citations'] = pd.to_numeric(df['citations'], errors='coerce').fillna(0).astype(int)
                df_list.append(df)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                pass
                
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            # 简单的清洗
            full_df = full_df.drop_duplicates(subset=['title', 'venue', 'year'])
            # 填充空值以避免后续错误
            full_df['abstract'] = full_df['abstract'].fillna("")
            full_df['title'] = full_df['title'].fillna("")
            return full_df
        else:
            return pd.DataFrame()

    @staticmethod
    def filter_data(df, venues=None, year_range=None, min_citations=0):
        """
        根据条件筛选数据。
        
        Args:
            df (pd.DataFrame): 原始数据 DataFrame。
            venues (list, optional): 需要筛选的会议列表。默认为 None（不筛选）。
            year_range (tuple, optional): 年份范围 (start_year, end_year)。默认为 None（不筛选）。
            min_citations (int, optional): 最小引用数。默认为 0。
            
        Returns:
            pd.DataFrame: 筛选后的 DataFrame。
        """
        if df.empty:
            return df
            
        mask = pd.Series([True] * len(df), index=df.index)
        
        # 1. 会议筛选
        if venues:
            mask &= df['venue'].isin(venues)
            
        # 2. 年份筛选 (tuple: (start, end))
        if year_range:
            mask &= (df['year'] >= year_range[0]) & (df['year'] <= year_range[1])
            
        # 3. 引用筛选
        if min_citations > 0:
            mask &= (df['citations'] >= min_citations)
            
        return df[mask]

    @staticmethod
    def get_venue_stats(df):
        """
        获取数据集的统计信息，包括所有会议名称和年份范围。
        
        Args:
            df (pd.DataFrame): 论文数据 DataFrame。
            
        Returns:
            tuple: (venues_list, (min_year, max_year))
        """
        if df.empty:
            return [], (2000, 2024)
            
        venues = sorted(df['venue'].unique().tolist())
        min_year = int(df['year'].min())
        max_year = int(df['year'].max())
        return venues, (min_year, max_year)
