import streamlit as st
import yfinance as yf
import requests
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Set up the Streamlit app
st.title("Stock Market News Sentiment Bot")

# Function to fetch historical stock data
def get_historical_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    return stock.history(start=start_date, end=end_date)

# Function to fetch news data from NewsAPI
def get_news(query, from_date, to_date):
    YOUR_NEWS_API_KEY = "7c9628099fbd4d63be8c502113ad9ec7"
    url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&apiKey={YOUR_NEWS_API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to analyze sentiment using TextBlob
def get_sentiment(news_article):
    analysis = TextBlob(news_article)
    return analysis.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)

# Function to prepare the dataset for machine learning
def prepare_dataset(stock_data, news_data):
    dataset = pd.DataFrame(stock_data)
    sentiment_scores = []
    
    for date in dataset.index:
        articles = [article for article in news_data['articles'] if article['publishedAt'][:10] == str(date)[:10]]
        sentiment = sum(get_sentiment(article['description']) for article in articles) / len(articles) if articles else 0
        sentiment_scores.append(sentiment)
    
    dataset['sentiment'] = sentiment_scores
    return dataset

# Function to train the machine learning model
def train_model(data):
    data['target'] = data['Close'].pct_change().apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
    X = data[['Close', 'Volume', 'sentiment']]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    score = clf.score(X_test, y_test)
    return clf, score

# Risk management strategy
def risk_management(prediction, confidence, max_loss=0.05):
    if confidence < 0.7:
        return "Hold"
    if prediction == 1 and confidence >= 0.7:
        return "Buy"
    elif prediction == -1 and confidence >= 0.7:
        return "Sell"
    else:
        return "Hold"

# Function to generate portfolio-based recommendations
def portfolio_recommendation(user_portfolio, model, news_data):
    recommendations = {}
    for stock, shares in user_portfolio.items():
        stock_data = get_historical_data(stock, '2020-01-01', '2023-09-01')
        dataset = prepare_dataset(stock_data, news_data)
        
        X = dataset[['Close', 'Volume', 'sentiment']]
        prediction = model.predict(X)
        confidence = max(model.predict_proba(X)[0])
        
        recommendation = risk_management(prediction, confidence)
        recommendations[stock] = recommendation
    return recommendations

# User portfolio (you can later extend this to dynamic input)
user_portfolio = {'AAPL': 50, 'GOOGL': 20}

# Get user input for stocks and dates
stock = st.text_input("Enter stock symbol (e.g., AAPL, GOOGL)", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-09-01"))

# Fetch historical stock data and news data
if st.button("Fetch Data"):
    stock_data = get_historical_data(stock, start_date, end_date)
    st.write(f"Historical data for {stock} from {start_date} to {end_date}")
    st.dataframe(stock_data)

    news_data = get_news(stock, start_date, end_date)
    st.write(f"News related to {stock}")
    for article in news_data['articles']:
        st.write(f"Title: {article['title']}")
        st.write(f"Description: {article['description']}")
        st.write("---")

    # Prepare dataset for machine learning
    data = prepare_dataset(stock_data, news_data)
    
    # Train the model and display accuracy
    model, accuracy = train_model(data)
    st.write(f"Model accuracy: {accuracy:.2f}")

    # Display portfolio recommendations
    recommendations = portfolio_recommendation(user_portfolio, model, news_data)
    st.write("Portfolio Recommendations:")
    for stock, recommendation in recommendations.items():
        st.write(f"{stock}: {recommendation}")

# Note: Replace 'YOUR_NEWS_API_KEY' with your actual NewsAPI key.

