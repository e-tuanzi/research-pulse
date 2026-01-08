# Copyright (c) 2026 tuanzi. All rights reserved.
import streamlit as st
import pandas as pd
import plotly.express as px
from data_manager import DataManager
from search_engine import BooleanSearchEngine

# --- 页面配置 ---
st.set_page_config(
    page_title="AI会议论文检索工具",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """
    主应用程序入口函数。
    负责渲染 Streamlit 界面，处理用户交互，以及展示搜索和分析结果。
    """
    st.title("🧠 AI会议论文检索工具")
    
    # Session State 初始化
    if 'page' not in st.session_state:
        st.session_state.page = 0
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # 1. 加载数据
    with st.spinner("正在加载论文数据库..."):
        full_df = DataManager.load_data()
        
    if full_df.empty:
        st.error("⚠️ 未找到数据文件。请检查 `ai_papers_data` 目录。")
        return

    # --- 2. 侧边栏：全局筛选 (Faceted Filtering) ---
    st.sidebar.header("🛠️ 全局筛选 (Filters)")
    
    # 获取统计信息
    all_venues, (min_year, max_year) = DataManager.get_venue_stats(full_df)
    
    # 年份筛选
    selected_years = st.sidebar.slider(
        "年份范围 (Year Range)",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, 2018), max_year) # 默认最近几年
    )
    
    # 会议筛选
    selected_venues = st.sidebar.multiselect(
        "会议 (Venues)",
        all_venues,
        default=all_venues # 默认全选
    )
    
    # 引用筛选 - 改为数字输入框
    min_citations = st.sidebar.number_input(
        "最少引用数 (Min Citations)",
        min_value=0,
        value=0,
        step=1,
        help="输入最小引用数进行过滤"
    )
    
    # 数据概览
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"📊 **总收录**: {len(full_df):,} 篇")
    
    # --- 3. 主界面 Tabs ---
    tab_search, tab_insight = st.tabs(["🔍 智能检索 (Search)", "📈 深度洞察 (Insights)"])
    
    # ==================== Tab 1: 智能检索 ====================
    with tab_search:
        # 移除了模式选择，直接展示输入框
        query = st.text_input(
            "输入关键词 (支持布尔搜索)",
            placeholder="例如: 'transformer AND (vision OR image)' 或 'diffusion model'",
            label_visibility="visible"
        )

        # 语法提示
        st.caption("💡 提示: 支持 `AND`, `OR`, `NOT`, `*`, `()`。例如: `diffusion AND (image OR video)`")

        # 检测查询变化重置页码
        if query != st.session_state.last_query:
            st.session_state.page = 0
            st.session_state.last_query = query

        # 执行搜索
        if query:
            # 仅使用 Expert Mode (Boolean Search)
            bool_engine = BooleanSearchEngine(full_df)
            search_result = bool_engine.search(query)
            
            # B. 筛选阶段
            final_result = DataManager.filter_data(
                search_result,
                venues=selected_venues,
                year_range=selected_years,
                min_citations=min_citations
            )
            
            # C. 排序与展示
            if not final_result.empty:
                # 默认按年份和引用排序
                final_result = final_result.sort_values(by=['year', 'citations'], ascending=[False, False])
                
                # --- 新增功能：结果展示与导出 ---
                # 使用列布局优化界面：左侧显示结果统计，右侧放置导出按钮
                col_res, col_download = st.columns([3, 1])
                
                with col_res:
                    st.success(f"找到 {len(final_result)} 篇相关论文")
                    
                with col_download:
                    # 生成 CSV 数据
                    # 注意：使用 utf-8-sig 编码以确保 Windows Excel 能正确显示中文
                    csv_data = final_result.to_csv(index=False).encode('utf-8-sig')
                    
                    st.download_button(
                        label="📥 导出结果 (CSV)",
                        data=csv_data,
                        file_name='search_results.csv',
                        mime='text/csv',
                        help="点击将当前搜索结果下载为 CSV 文件"
                    )
                # -----------------------------
                
                # --- 分页逻辑 (Pagination) ---
                ITEMS_PER_PAGE = 10
                total_docs = len(final_result)
                total_pages = max(1, (total_docs - 1) // ITEMS_PER_PAGE + 1)
                
                # 翻页控件
                col_p1, col_p2, col_p3 = st.columns([1, 3, 1])
                with col_p1:
                    if st.button("⬅️ 上一页", disabled=st.session_state.page == 0):
                        st.session_state.page -= 1
                        st.rerun()
                with col_p2:
                    st.markdown(f"<div style='text-align: center; padding-top: 5px;'>第 <b>{st.session_state.page + 1}</b> / {total_pages} 页</div>", unsafe_allow_html=True)
                with col_p3:
                    if st.button("下一页 ➡️", disabled=st.session_state.page >= total_pages - 1):
                        st.session_state.page += 1
                        st.rerun()
                
                # 切片显示
                start_idx = st.session_state.page * ITEMS_PER_PAGE
                end_idx = start_idx + ITEMS_PER_PAGE
                
                for idx, row in final_result.iloc[start_idx:end_idx].iterrows():
                    with st.expander(f"{'🔥 ' if row['citations']>100 else ''}{row['title']} ({row['year']} {row['venue']})"):
                        st.markdown(f"**引用数**: {row['citations']}")
                        st.markdown(f"**摘要**: {row['abstract']}")
                        # 移除了相似论文推荐功能
            else:
                st.warning("未找到匹配的论文，请尝试放宽筛选条件。")
                
    # ==================== Tab 2: 深度洞察 ====================
    with tab_insight:
        analysis_source = st.radio(
            "分析数据源", 
            ["当前筛选全量数据", "当前搜索结果"], 
            horizontal=True,
            index=0 if not query else 1
        )
        
        if analysis_source == "当前搜索结果" and query:
            base_df = BooleanSearchEngine(full_df).search(query)
        else:
            base_df = full_df
            
        viz_df = DataManager.filter_data(
            base_df,
            venues=selected_venues,
            year_range=selected_years,
            min_citations=min_citations
        )
        
        if viz_df.empty:
            st.warning("当前筛选条件下无数据可分析。")
        else:
            col_chart1, col_chart2 = st.columns(2)
            
            # 1. 趋势图 (Trend) - 增加会议拆分与交互
            with col_chart1:
                st.subheader("📈 研究热度趋势")
                
                c1, c2 = st.columns(2)
                with c1:
                    show_relative = st.checkbox("显示相对热度 (%)", help="该主题论文占当年总收录数的百分比")
                with c2:
                    show_venue_breakdown = st.checkbox("按会议拆分 (By Venue)", value=False)
                
                # 数据准备
                if show_venue_breakdown:
                    trend_data = viz_df.groupby(['year', 'venue']).size().reset_index(name='count')
                    if show_relative:
                        # 分母：该会议当年的总论文数
                        all_venue_stats = full_df.groupby(['year', 'venue']).size().reset_index(name='total')
                        trend_data = pd.merge(trend_data, all_venue_stats, on=['year', 'venue'])
                        trend_data['value'] = (trend_data['count'] / trend_data['total']) * 100
                        y_label = "相对热度 (%)"
                    else:
                        trend_data['value'] = trend_data['count']
                        y_label = "论文数量"
                    
                    fig_trend = px.line(
                        trend_data, x='year', y='value', color='venue', markers=True,
                        labels={'value': y_label, 'year': '年份'},
                        title="分会议发表趋势"
                    )
                else:
                    trend_data = viz_df.groupby('year').size().reset_index(name='count')
                    if show_relative:
                        # 分母：选定会议当年的总论文数
                        all_venue_df = DataManager.filter_data(full_df, venues=selected_venues)
                        total_per_year = all_venue_df.groupby('year').size().reset_index(name='total')
                        trend_data = pd.merge(trend_data, total_per_year, on='year')
                        trend_data['value'] = (trend_data['count'] / trend_data['total']) * 100
                        y_label = "相对热度 (%)"
                    else:
                        trend_data['value'] = trend_data['count']
                        y_label = "论文数量"
                        
                    fig_trend = px.line(
                        trend_data, x='year', y='value', markers=True,
                        labels={'value': y_label, 'year': '年份'},
                        title="总体发表趋势 (可点击数据点)"
                    )
                
                # 交互式图表 (on_select="rerun")
                # 允许选择数据点，捕获年份
                event = st.plotly_chart(fig_trend, use_container_width=True, on_select="rerun", selection_mode="points")

            # 2. 会议分布 (Venue)
            with col_chart2:
                st.subheader("🏛️ 会议分布")
                venue_counts = viz_df['venue'].value_counts().reset_index()
                venue_counts.columns = ['venue', 'count']
                fig_bar = px.pie(venue_counts, values='count', names='venue', title="论文来源分布")
                st.plotly_chart(fig_bar, use_container_width=True)
                
            # 3. 影响力分析 (Impact Analysis) - 联动更新
            st.markdown("---")
            st.subheader("⭐ 年度高影响力论文 (Top Cited)")
            st.caption("👇 点击上方趋势图的年份点，或拖动下方滑块切换年份")
            
            # 动态更新 Slider 状态
            available_years = sorted(viz_df['year'].unique(), reverse=True)
            default_year = available_years[0] if available_years else 2024
            
            # 处理图表点击事件
            if event and event.selection and event.selection['points']:
                try:
                    clicked_year = int(event.selection['points'][0]['x'])
                    if clicked_year in available_years:
                        st.session_state['year_slider_key'] = clicked_year
                except:
                    pass
            
            # 确保 session state 有值且有效
            if 'year_slider_key' not in st.session_state or st.session_state['year_slider_key'] not in available_years:
                st.session_state['year_slider_key'] = default_year
            
            if available_years:
                # 绑定 key 到 session_state
                selected_year_impact = st.select_slider(
                    "选择年份查看 Top 3", 
                    options=available_years, 
                    value=st.session_state['year_slider_key'],
                    key="year_slider_key" 
                )
                
                top_papers = viz_df[viz_df['year'] == selected_year_impact].nlargest(3, 'citations')
                
                col_p1, col_p2, col_p3 = st.columns(3)
                if top_papers.empty:
                    st.info(f"{selected_year_impact} 年无数据")
                else:
                    for i, (idx, row) in enumerate(top_papers.iterrows()):
                        with [col_p1, col_p2, col_p3][i]:
                            st.info(f"🏆 Top {i+1}")
                            st.markdown(f"**{row['title']}**")
                            st.markdown(f"*{row['venue']}* | 引用: **{row['citations']}**")
                            with st.expander("摘要"):
                                st.caption(row['abstract'][:200] + "..." if isinstance(row['abstract'], str) else "无摘要")

if __name__ == "__main__":
    main()
