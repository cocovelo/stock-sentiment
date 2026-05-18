import os
import praw
import requests
import datetime
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import feedparser
import numpy as np
import ta  # For technical indicators
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ====== Hugging Face sentiment model (GPU) ======
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=0)

# ====== Helper: batch sentiment analysis ======
def analyze_sentiment_batch(texts, batch_size=16):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Batches"):
        batch = texts[i:i+batch_size]
        batch_results = sentiment_analyzer(batch)
        results.extend(batch_results)
    return results

# ====== Reddit setup ======
reddit = praw.Reddit(
    client_id="hqEcxvSbEt4nUbr3AzUIpQ",
    client_secret="1Gs7eHsmthaoymOV3dhQQ9ETOFl-eg",
    user_agent="crypto_sentiment_script"
)
subreddits = ["CryptoCurrency", "Bitcoin", "Ethereum", "CryptoMarkets", "altcoin"]

def fetch_reddit_posts(limit=1000):
    posts_data = []
    earliest = datetime.datetime.utcnow()
    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        titles, timestamps, urls = [], [], []
        print(f"Fetching Reddit posts from r/{subreddit_name}...")
        for post in tqdm(subreddit.new(limit=limit), desc=f"Reddit: {subreddit_name}"):
            ts = datetime.datetime.utcfromtimestamp(post.created_utc)
            if ts < earliest:
                earliest = ts
            titles.append(post.title)
            timestamps.append(ts)
            urls.append(post.url)
        sentiments = analyze_sentiment_batch(titles)
        for title, ts, url, sentiment in zip(titles, timestamps, urls, sentiments):
            posts_data.append({
                "source": "reddit",
                "subreddit": subreddit_name,
                "title": title,
                "url": url,
                "timestamp": ts,
                "score": sentiment['score'],
                "label": sentiment['label']
            })
    print(f"Earliest Reddit post: {earliest}")
    return posts_data, earliest

# ====== Crypto news via RSS (CoinDesk & CoinTelegraph) ======
rss_feeds = {
    "CoinDesk": ["https://www.coindesk.com/arc/outboundfeeds/rss/"],
    "CoinTelegraph": ["https://cointelegraph.com/rss"]
}

def fetch_rss_articles(feeds=rss_feeds, limit_per_feed=1000):
    articles_data = []
    earliest = datetime.datetime.utcnow()
    for source, urls in feeds.items():
        for url in urls:
            print(f"Fetching RSS feed from {source}...")
            feed = feedparser.parse(url)
            entries = feed.entries[:limit_per_feed]
            titles, texts, timestamps, links = [], [], [], []
            for entry in tqdm(entries, desc=f"RSS: {source}"):
                ts = pd.to_datetime(entry.get("published", datetime.datetime.utcnow()))
                if ts < earliest:
                    earliest = ts
                title = entry.get("title", "")
                desc = entry.get("summary", "")
                text = title + " " + desc
                link = entry.get("link", "")
                titles.append(title)
                texts.append(text)
                timestamps.append(ts)
                links.append(link)
            sentiments = analyze_sentiment_batch(texts)
            for title, text, ts, link, sentiment in zip(titles, texts, timestamps, links, sentiments):
                articles_data.append({
                    "source": source,
                    "title": title,
                    "description": text,
                    "url": link,
                    "timestamp": ts,
                    "score_news": sentiment['score'],
                    "label_news": sentiment['label']
                })
    print(f"Earliest RSS article: {earliest}")
    return articles_data, earliest

# ====== KuCoin market data (dynamic start date) ======
def fetch_kucoin_prices(symbols=["BTC-USDT","ETH-USDT"], start=None, interval="1hour"):
    prices_data = {}
    end_ts = int(datetime.datetime.utcnow().timestamp())
    start_ts = int(start.timestamp()) if start else int((datetime.datetime.utcnow() - pd.Timedelta(days=730)).timestamp())
    for symbol in symbols:
        print(f"Fetching KuCoin data for {symbol} from {datetime.datetime.utcfromtimestamp(start_ts)}...")
        all_data = []
        while start_ts < end_ts:
            params = {"symbol": symbol, "type": interval, "startAt": start_ts, "endAt": end_ts}
            response = requests.get("https://api.kucoin.com/api/v1/market/candles", params=params)
            if response.status_code != 200:
                print(f"Failed to fetch {symbol}: {response.text}")
                break
            data = response.json().get("data", [])
            if not data:
                break
            all_data.extend(data)
            last_ts = int(data[-1][0])
            if last_ts <= start_ts:
                break
            start_ts = last_ts + 1
        if all_data:
            df = pd.DataFrame(all_data, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df = df.sort_values("timestamp").reset_index(drop=True)
            prices_data[symbol] = df
            print(f"Fetched {len(df)} rows for {symbol}")
    return prices_data

# ====== Aggregate daily sentiment ======
def aggregate_daily_sentiment(posts, news):
    df_posts = pd.DataFrame(posts)
    df_posts['date'] = df_posts['timestamp'].dt.date
    daily_reddit = df_posts.groupby('date')['score'].mean().reset_index()

    df_news = pd.DataFrame(news)
    df_news['date'] = df_news['timestamp'].dt.date
    daily_news = df_news.groupby('date')['score_news'].mean().reset_index()

    daily_sentiment = pd.merge(daily_reddit, daily_news, on='date', how='outer').fillna(0)
    daily_sentiment['score_combined'] = (daily_sentiment['score'] + daily_sentiment['score_news']) / 2
    daily_sentiment = daily_sentiment.sort_values('date').reset_index(drop=True)
    for lag in range(1,8):
        daily_sentiment[f'score_combined_lag{lag}'] = daily_sentiment['score_combined'].shift(lag)
    return daily_sentiment

# ====== Add technical indicators, time features, interaction features ======
def add_features(df):
    df = df.copy()
    df['MA7'] = df['close'].rolling(7).mean()
    df['MA21'] = df['close'].rolling(21).mean()
    df['volatility'] = df['daily_return'].rolling(7).std()
    # RSI
    df['RSI'] = RSIIndicator(df['close']).rsi()
    # MACD
    macd = MACD(df['close']).macd()
    df['MACD'] = macd
    df['MACD_signal'] = MACD(df['close']).macd_signal()
    # Time-based features
    df['day_of_week'] = df['date'].apply(lambda x: x.weekday())
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day_of_month'] = df['date'].apply(lambda x: x.day)
    # Interaction features
    for lag in range(1,8):
        df[f'sentiment_return_lag{lag}'] = df[f'score_combined_lag{lag}'] * df['daily_return'].shift(lag)
    return df

# ====== Align sentiment with price data and compute targets ======
def align_prices_with_sentiment(prices_df, daily_sentiment):
    prices_df['date'] = prices_df['timestamp'].dt.date
    daily_prices = prices_df.groupby('date')['close'].mean().reset_index()
    merged = pd.merge(daily_prices, daily_sentiment, on='date', how='left').fillna(0)
    # Compute daily returns
    merged['daily_return'] = merged['close'].pct_change()
    # Targets T+1, T+3, T+7
    for i in [1,3,7]:
        merged[f'close_T+{i}'] = merged['close'].shift(-i)
        merged[f'daily_return_T+{i}'] = merged[f'close_T+{i}'].pct_change()
        merged[f'up_T+{i}'] = (merged[f'daily_return_T+{i}'] > 0).astype(int)
    merged = add_features(merged)
    return merged

# ====== Fetch and process combined data ======
def fetch_combined_data():
    print("Fetching Reddit posts...")
    reddit_data, earliest_reddit = fetch_reddit_posts()
    print("Fetching RSS news articles...")
    news_data, earliest_news = fetch_rss_articles()
    earliest_date = min(earliest_reddit, earliest_news)
    print(f"Earliest date to fetch crypto prices: {earliest_date}")
    print("Fetching KuCoin price data...")
    crypto_prices_raw = fetch_kucoin_prices(start=earliest_date)
    print("Aggregating daily sentiment...")
    daily_sentiment = aggregate_daily_sentiment(reddit_data, news_data)
    print("Aligning sentiment with crypto prices and generating ML features...")
    crypto_prices_aligned = {}
    for symbol, df in crypto_prices_raw.items():
        crypto_prices_aligned[symbol] = align_prices_with_sentiment(df, daily_sentiment)
    return {
        "reddit": reddit_data,
        "news": news_data,
        "crypto_prices": crypto_prices_aligned
    }

# ====== Example usage ======
if __name__ == "__main__":
    combined_data = fetch_combined_data()
    print("Reddit posts:", len(combined_data["reddit"]))
    print("News articles:", len(combined_data["news"]))
    for crypto, df in combined_data["crypto_prices"].items():
        print(f"{crypto} aligned data points:", len(df))
        print(df.head())
