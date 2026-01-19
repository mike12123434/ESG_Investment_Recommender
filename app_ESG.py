"""
ESG Investment Recommendation System
Real web scraping with semantic analysis - No simulated data
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import concurrent.futures
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# ========== Configuration ==========

ESG_KEYWORDS = {
    'env': ['綠能', '減碳', '碳排放', '氣候', '再生能源', '淨零', '永續', '環保', '能源轉型', '循環經濟',
            'green', 'carbon', 'climate', 'renewable', 'sustainable', 'esg', 'environmental'],
    'social': ['社會', '員工', '人權', '多元', '平等', '勞工', '福利', '社區', '供應鏈',
               'social', 'labor', 'employee', 'diversity', 'welfare', 'community'],
    'gov': ['治理', '董事會', '透明', '獨立董事', '風險管理', '內控', '合規', '股東',
            'governance', 'board', 'compliance', 'transparency', 'ethics', 'risk']
}

# ========== Web Scraping Functions ==========

def scrape_esg_etfs():
    """Scrape ESG ETF list from Taiwan Stock Exchange"""
    try:
        url = "https://www.twse.com.tw/zh/esg-index-product/etf"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None
            
        soup = BeautifulSoup(response.text, 'html.parser')
        etfs = []
        
        for row in soup.find_all('tr')[1:]:
            cols = row.find_all('td')
            if len(cols) >= 2:
                code = cols[0].get_text(strip=True)
                name = cols[1].get_text(strip=True).split('(')[0].strip()
                
                if code and len(code) >= 4 and code[:4].isdigit():
                    etfs.append({'code': code, 'name': name})
        
        unique_etfs = list({e['code']: e for e in etfs}.values())[:25]
        return unique_etfs if unique_etfs else None
        
    except Exception as e:
        return None


def scrape_goodinfo_data(code):
    """Scrape company info from Goodinfo"""
    try:
        url = f"https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={code}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content[:3000] if text_content else ""
        
    except Exception:
        return ""


def scrape_moneydj_news(code):
    """Scrape news from MoneyDJ"""
    try:
        url = f"https://www.moneydj.com/etf/x/basic/basic0004.xdjhtm?etfid={code}.tw"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract text from main content areas
        texts = []
        for tag in soup.find_all(['p', 'div', 'span', 'td']):
            text = tag.get_text(strip=True)
            if len(text) > 20:
                texts.append(text)
        
        return ' '.join(texts[:50])
        
    except Exception:
        return ""


def scrape_yahoo_finance_info(code):
    """Scrape Yahoo Finance Taiwan stock page"""
    try:
        url = f"https://tw.stock.yahoo.com/quote/{code}.TW"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text(separator=' ', strip=True)
        return text_content[:2000] if text_content else ""
        
    except Exception:
        return ""


def scrape_cnyes_news(code):
    """Scrape news from Anue (鉅亨網)"""
    try:
        url = f"https://fund.cnyes.com/detail/{code}/profile"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        texts = []
        
        for tag in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'span']):
            text = tag.get_text(strip=True)
            if len(text) > 15:
                texts.append(text)
        
        return ' '.join(texts[:40])
        
    except Exception:
        return ""


def comprehensive_web_scrape(code, name):
    """Scrape multiple sources and combine text"""
    all_text = []
    sources_used = []
    
    # Add ETF name
    all_text.append(name)
    
    # Scrape from multiple sources
    scrapers = [
        ('Yahoo Finance', lambda: scrape_yahoo_finance_info(code)),
        ('MoneyDJ', lambda: scrape_moneydj_news(code)),
        ('Goodinfo', lambda: scrape_goodinfo_data(code)),
        ('Anue', lambda: scrape_cnyes_news(code))
    ]
    
    for source_name, scraper_func in scrapers:
        try:
            text = scraper_func()
            if text and len(text) > 50:
                all_text.append(text)
                sources_used.append(source_name)
                time.sleep(0.5)  # Polite delay
        except Exception:
            continue
    
    combined_text = ' '.join(all_text)
    sources_str = ', '.join(sources_used) if sources_used else 'ETF Name Only'
    
    return combined_text, sources_str


# ========== Semantic Analysis ==========

def analyze_esg_with_tfidf(text):
    """Semantic analysis using TF-IDF and keyword matching"""
    text_lower = text.lower()
    
    # Keyword frequency scoring
    def keyword_score(keywords):
        matches = sum(1 for kw in keywords if kw in text_lower)
        # More matches = higher score
        return min(3 + matches * 1.2, 10)
    
    e_base = keyword_score(ESG_KEYWORDS['env'])
    s_base = keyword_score(ESG_KEYWORDS['social'])
    g_base = keyword_score(ESG_KEYWORDS['gov'])
    
    # TF-IDF semantic similarity
    try:
        # Create reference documents for each ESG dimension
        e_ref = ' '.join(ESG_KEYWORDS['env'])
        s_ref = ' '.join(ESG_KEYWORDS['social'])
        g_ref = ' '.join(ESG_KEYWORDS['gov'])
        
        vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text, e_ref, s_ref, g_ref])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:4])[0]
        
        # Combine keyword score (60%) with TF-IDF similarity (40%)
        e_score = e_base * 0.6 + similarities[0] * 10 * 0.4
        s_score = s_base * 0.6 + similarities[1] * 10 * 0.4
        g_score = g_base * 0.6 + similarities[2] * 10 * 0.4
        
    except Exception:
        e_score, s_score, g_score = e_base, s_base, g_base
    
    return {
        'E': round(max(e_score, 3.0), 1),
        'S': round(max(s_score, 3.0), 1),
        'G': round(max(g_score, 3.0), 1)
    }


# ========== Market Data ==========

def fetch_market_data(code):
    """Fetch market data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(f"{code}.TW")
        hist = ticker.history(period="1y")
        
        if hist.empty or len(hist) < 30:
            return None
        
        start_price = hist['Close'].iloc[0]
        end_price = hist['Close'].iloc[-1]
        
        if start_price <= 0 or end_price <= 0:
            return None
            
        returns = ((end_price / start_price) - 1) * 100
        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
        
        volatility = max(volatility, 0.1)
        sharpe = ((returns / 100) - 0.02) / (volatility / 100)
        risk_level = int(np.clip(round(volatility / 5), 1, 5))
        
        return {
            'annual_return': round(returns, 2),
            'volatility': round(volatility, 2),
            'risk_level': risk_level,
            'sharpe_ratio': round(sharpe, 2),
            'avg_volume': int(hist['Volume'].mean())
        }
        
    except Exception:
        return None


def process_etf(etf_info):
    """Process single ETF with real web scraping"""
    
    # Fetch market data
    market_data = fetch_market_data(etf_info['code'])
    if not market_data:
        return None
    
    # Comprehensive web scraping
    scraped_text, sources = comprehensive_web_scrape(etf_info['code'], etf_info['name'])
    
    if not scraped_text or len(scraped_text) < 100:
        return None
    
    # Semantic ESG analysis
    esg_scores = analyze_esg_with_tfidf(scraped_text)
    
    return {
        'code': etf_info['code'],
        'name': etf_info['name'],
        **market_data,
        'E_score': esg_scores['E'],
        'S_score': esg_scores['S'],
        'G_score': esg_scores['G'],
        'scraped_text': scraped_text[:1000],  # First 1000 chars for display
        'full_text_length': len(scraped_text),
        'data_sources': sources,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


@st.cache_data(ttl=3600)
def build_dataset():
    """Build complete dataset with real web scraping"""
    etf_list = scrape_esg_etfs()
    
    if not etf_list:
        return None
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(etf_list)
    
    for idx, etf in enumerate(etf_list):
        status_text.text(f"Processing {etf['code']} - {etf['name']} ({idx+1}/{total})")
        progress_bar.progress((idx + 1) / total)
        
        result = process_etf(etf)
        if result:
            results.append(result)
        
        time.sleep(0.3)  # Polite delay between requests
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results) if results else None


def recommend_etfs(df, user_weights, risk_tolerance):
    """Generate recommendations based on user preferences"""
    data = df.copy()
    
    scaler = MinMaxScaler()
    data[['E_norm', 'S_norm', 'G_norm']] = scaler.fit_transform(
        data[['E_score', 'S_score', 'G_score']]
    )
    
    total_weight = user_weights['E'] + user_weights['S'] + user_weights['G']
    user_vec = np.array([[
        user_weights['E'] / total_weight,
        user_weights['S'] / total_weight,
        user_weights['G'] / total_weight
    ]])
    
    data['esg_match'] = cosine_similarity(
        user_vec,
        data[['E_norm', 'S_norm', 'G_norm']]
    )[0]
    
    data['sharpe_norm'] = scaler.fit_transform(data[['sharpe_ratio']])
    data['return_norm'] = scaler.fit_transform(data[['annual_return']])
    
    data['total_score'] = (
        data['esg_match'] * 0.5 +
        data['sharpe_norm'] * 0.3 +
        data['return_norm'] * 0.2
    )
    
    data = data[data['risk_level'] <= risk_tolerance]
    
    if data.empty:
        return None, None
    
    top_5 = data.nlargest(5, 'total_score').copy()
    top_5['allocation'] = (top_5['total_score'] / top_5['total_score'].sum() * 100).round(1)
    
    return top_5, data


# ========== UI ==========

def main():
    st.set_page_config(page_title="ESG Recommender", layout="wide")
    
    st.title("ESG Investment Recommender")
    st.caption("Real-time web scraping with semantic ESG analysis")
    
    with st.sidebar:
        st.subheader("Settings")
        
        risk = st.slider("Risk Tolerance", 1, 5, 3)
        
        st.write("ESG Weights")
        e_weight = st.number_input("Environmental (E)", 0, 100, 40, step=5)
        s_weight = st.number_input("Social (S)", 0, 100, 30, step=5)
        g_weight = st.number_input("Governance (G)", 0, 100, 30, step=5)
        
        total = e_weight + s_weight + g_weight
        if total != 100:
            st.warning(f"Total: {total}%. Should equal 100%")
        
        run_btn = st.button("Start Analysis", type="primary", use_container_width=True)
    
    if run_btn:
        if total != 100:
            st.error("ESG weights must sum to 100%")
            return
        
        st.info("Scraping ETF list from TWSE...")
        df = build_dataset()
        
        if df is None or df.empty:
            st.error("Unable to scrape data. Please try again later.")
            return
        
        st.success(f"Successfully scraped and analyzed {len(df)} ETFs")
        
        user_weights = {'E': e_weight, 'S': s_weight, 'G': g_weight}
        top_etfs, full_data = recommend_etfs(df, user_weights, risk)
        
        if top_etfs is None or top_etfs.empty:
            st.warning("No ETFs match your risk tolerance")
            return
        
        st.subheader("Top Recommendations")
        
        display_cols = ['name', 'code', 'allocation', 'annual_return', 
                       'sharpe_ratio', 'volatility', 'E_score', 'S_score', 'G_score']
        
        st.dataframe(
            top_etfs[display_cols],
            column_config={
                "allocation": st.column_config.ProgressColumn(
                    "Portfolio %", format="%.1f%%", min_value=0, max_value=100
                ),
                "annual_return": st.column_config.NumberColumn("Return %", format="%.2f%%"),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "volatility": st.column_config.NumberColumn("Volatility %", format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True
        )
        
        st.subheader("Raw Scraped Data")
        st.caption("All data from real web scraping - No simulation")
        
        for idx, row in df.iterrows():
            with st.expander(f"{row['name']} ({row['code']}) - {row['data_sources']}"):
                st.write(f"**Data Sources:** {row['data_sources']}")
                st.write(f"**Text Length:** {row['full_text_length']} characters")
                st.write(f"**Scraped at:** {row['timestamp']}")
                st.write(f"**ESG Scores:** E={row['E_score']}, S={row['S_score']}, G={row['G_score']}")
                st.text_area("Scraped Text Preview", row['scraped_text'], height=200, key=f"text_{idx}")
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Complete Dataset (CSV)",
            csv,
            "esg_scraped_data.csv",
            "text/csv",
            use_container_width=True
        )


if __name__ == "__main__":
    main()