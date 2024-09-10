# import streamlit as st
# import yfinance as yf
# import requests
# import pandas as pd
# import csv
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier

# # Load company data from CSV
# def load_company_data(file_path='stock.csv'):
#     company_data = {}
#     try:
#         with open(file_path, mode='r') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 ticker = row['Symbol'].strip().upper()
#                 company_data[ticker] = row['Name']
#     except Exception as e:
#         st.error(f"Error loading company data: {e}")
#     return company_data

# # Function to fetch news articles from the News API
# def fetch_news(query, num_articles=10, language='en'):
#     api_key = "7c9628099fbd4d63be8c502113ad9ec7"  # Your News API key
#     url = f"https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={api_key}&pageSize={num_articles}"
#     try:
#         response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
#         response.raise_for_status()  # Check for request errors
#         news_data = response.json()
#         return news_data.get('articles', [])
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error fetching news: {e}")
#         return []

# # Function to fetch historical stock data
# def get_historical_data(stock_symbol, start_date, end_date):
#     stock = yf.Ticker(stock_symbol)
#     return stock.history(start=start_date, end=end_date)

# # Sentiment Analysis using VADER
# def get_sentiment(news_article):
#     analyzer = SentimentIntensityAnalyzer()
#     sentiment = analyzer.polarity_scores(news_article)
#     return sentiment['compound']  # Compound score ranges from -1 (negative) to 1 (positive)

# # Add moving averages to stock data
# def add_moving_averages(data):
#     data['MA50'] = data['Close'].rolling(window=50).mean()
#     data['MA200'] = data['Close'].rolling(window=200).mean()
#     return data

# # Add volatility (standard deviation of returns)
# def add_volatility(data):
#     data['Returns'] = data['Close'].pct_change()
#     data['Volatility'] = data['Returns'].rolling(window=10).std()
#     return data

# # Function to prepare the dataset for machine learning
# def prepare_dataset(stock_data, news_articles):
#     dataset = pd.DataFrame(stock_data)
#     dataset = add_moving_averages(dataset)
#     dataset = add_volatility(dataset)
    
#     sentiment_scores = []
#     for date in dataset.index:
#         articles = [article for article in news_articles if article['publishedAt'][:10] == str(date)[:10]]
#         sentiment = sum(get_sentiment(article['description']) for article in articles) / len(articles) if articles else 0
#         sentiment_scores.append(sentiment)
    
#     dataset['sentiment'] = sentiment_scores
#     dataset = dataset.dropna()  # Remove rows with NaN values (e.g., missing data)
#     return dataset

# # Function to train the machine learning model using XGBoost
# def train_model(data):
#     # Define the target variable
#     data['target'] = data['Close'].pct_change().apply(lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1))
#     X = data[['Close', 'Volume', 'sentiment', 'MA50', 'MA200', 'Volatility']]
#     y = data['target']
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Define the model
#     clf = RandomForestClassifier()  # Use RandomForestClassifier or XGBClassifier
    
#     # Fit the model
#     clf.fit(X_train, y_train)
    
#     # Evaluate model performance
#     score = clf.score(X_test, y_test)
    
#     # Return the classifier and the test data to evaluate it later
#     return clf, X_test, y_test, score

# # Function to evaluate model performance
# def evaluate_model(clf, X_test, y_test):
#     y_pred = clf.predict(X_test)
#     st.write("Confusion Matrix:")
#     st.write(confusion_matrix(y_test, y_pred))
#     st.write("Classification Report:")
#     st.write(classification_report(y_test, y_pred))

# # Risk management strategy
# def risk_management(prediction, confidence, max_loss=0.05):
#     if confidence < 0.7:
#         return "Hold"
#     if prediction == 2 and confidence >= 0.7:
#         return "Buy"
#     elif prediction == 0 and confidence >= 0.7:
#         return "Sell"
#     else:
#         return "Hold"

# # Function to generate portfolio-based recommendations
# def portfolio_recommendation(user_portfolio, model, news_articles):
#     recommendations = {}
#     for stock, shares in user_portfolio.items():
#         stock_data = get_historical_data(stock, '2020-01-01', '2023-09-01')
#         dataset = prepare_dataset(stock_data, news_articles)
        
#         X = dataset[['Close', 'Volume', 'sentiment', 'MA50', 'MA200', 'Volatility']]
#         if not X.empty:
#             prediction = model.predict(X)
#             confidence = max(model.predict_proba(X)[0])
#             recommendation = risk_management(prediction[-1], confidence)  # Use the last prediction
#             recommendations[stock] = recommendation
#     return recommendations

# # Load company data
# company_data = load_company_data()

# # Set up the Streamlit app
# st.title("Stock Market News Sentiment Bot")

# # Get user input for stocks and dates
# stock = st.text_input("Enter stock symbol (e.g., AAPL, GOOGL)", value="AAPL").strip().upper()
# start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# end_date = st.date_input("End Date", value=pd.to_datetime("2023-09-01"))

# # Fetch historical stock data and news data
# if st.button("Fetch Data"):
#     if stock not in company_data:
#         st.error(f"Stock symbol '{stock}' not found in company data.")
#     else:
#         stock_data = get_historical_data(stock, start_date, end_date)
#         st.write(f"Historical data for {stock} from {start_date} to {end_date}")
#         st.dataframe(stock_data)

#         news_articles = fetch_news(stock)
#         st.write(f"News related to {stock}")
#         if news_articles:
#             for article in news_articles:
#                 st.write(f"Title: {article.get('title', 'No title')}")
#                 st.write(f"Description: {article.get('description', 'No description')}")
#                 st.write("---")
#         else:
#             st.write("No news articles found.")

#         # Prepare dataset for machine learning
#         dataset = prepare_dataset(stock_data, news_articles)
        
#         # Train the model and display the best parameters
#         model, X_test, y_test, score = train_model(dataset)
#         st.write(f"Model accuracy: {score:.2f}")

#         # Evaluate model performance
#         evaluate_model(model, X_test, y_test)

#         # Display portfolio recommendations
#         user_portfolio = {'AAPL': 50, 'GOOGL': 20}  # Example portfolio
#         recommendations = portfolio_recommendation(user_portfolio, model, news_articles)
#         st.write("Portfolio Recommendations:")
#         for stock, recommendation in recommendations.items():
#             st.write(f"{stock}: {recommendation}")

import streamlit as st
import yfinance as yf
import requests
import pandas as pd
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load company data from CSV
def load_company_data(file_path='stock.csv'):
    company_data = {}
    try:
        with open(file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                ticker = row['Symbol'].strip().upper()
                company_data[ticker] = row['Name']
    except Exception as e:
        st.error(f"Error loading company data: {e}")
    return company_data

# Function to fetch news articles from the News API
def fetch_news(query, num_articles=10, language='en'):
    api_key = "7c9628099fbd4d63be8c502113ad9ec7"  # Your News API key
    url = f"https://newsapi.org/v2/everything?q={query}&language={language}&apiKey={api_key}&pageSize={num_articles}"
    try:
        response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
        response.raise_for_status()  # Check for request errors
        news_data = response.json()
        return news_data.get('articles', [])
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news: {e}")
        return []

# Function to fetch historical stock data
def get_historical_data(stock_symbol, start_date, end_date):
    stock = yf.Ticker(stock_symbol)
    return stock.history(start=start_date, end=end_date)

# Sentiment Analysis using VADER
def get_sentiment(news_article):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(news_article)
    return sentiment['compound']  # Compound score ranges from -1 (negative) to 1 (positive)

# Add moving averages to stock data
def add_moving_averages(data):
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    return data

# Add volatility (standard deviation of returns)
def add_volatility(data):
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=10).std()
    return data

# Function to prepare the dataset for machine learning
def prepare_dataset(stock_data, news_articles):
    dataset = pd.DataFrame(stock_data)
    dataset = add_moving_averages(dataset)
    dataset = add_volatility(dataset)
    
    sentiment_scores = []
    for date in dataset.index:
        articles = [article for article in news_articles if article['publishedAt'][:10] == str(date)[:10]]
        sentiment = sum(get_sentiment(article['description']) for article in articles) / len(articles) if articles else 0
        sentiment_scores.append(sentiment)
    
    dataset['sentiment'] = sentiment_scores
    dataset = dataset.dropna()  # Remove rows with NaN values (e.g., missing data)
    return dataset

# Function to train the machine learning model
def train_model(data):
    data['target'] = data['Close'].pct_change().apply(lambda x: 2 if x > 0.02 else (0 if x < -0.02 else 1))
    X = data[['Close', 'Volume', 'sentiment', 'MA50', 'MA200', 'Volatility']]
    y = data['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the model
    clf = RandomForestClassifier()  # Use RandomForestClassifier or XGBClassifier
    
    # Fit the model
    clf.fit(X_train, y_train)
    
    # Evaluate model performance
    score = clf.score(X_test, y_test)
    
    # Return the classifier and the test data to evaluate it later
    return clf, X_test, y_test, score

# Function to evaluate model performance
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, y_pred)
    st.dataframe(pd.DataFrame(conf_matrix, columns=['Predicted 0', 'Predicted 1', 'Predicted 2'], index=['Actual 0', 'Actual 1', 'Actual 2']))
    
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# Risk management strategy
def risk_management(prediction, confidence, max_loss=0.05):
    if confidence < 0.7:
        return "Hold"
    if prediction == 2 and confidence >= 0.7:
        return "Buy"
    elif prediction == 0 and confidence >= 0.7:
        return "Sell"
    else:
        return "Hold"

# Function to generate portfolio-based recommendations
def portfolio_recommendation(user_portfolio, model, news_articles):
    recommendations = {}
    for stock, shares in user_portfolio.items():
        stock_data = get_historical_data(stock, '2020-01-01', '2023-09-01')
        dataset = prepare_dataset(stock_data, news_articles)
        
        X = dataset[['Close', 'Volume', 'sentiment', 'MA50', 'MA200', 'Volatility']]
        if not X.empty:
            prediction = model.predict(X)
            confidence = max(model.predict_proba(X)[0])
            recommendation = risk_management(prediction[-1], confidence)  # Use the last prediction
            recommendations[stock] = recommendation
    return recommendations

# # Load company data
# company_data = load_company_data()

# # Set up the Streamlit app
# st.title("Stock Market News Sentiment Bot")

# # Get user input for stocks and dates
# stock = st.text_input("Enter stock symbol (e.g., AAPL, GOOGL)", value="AAPL").strip().upper()
# start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
# end_date = st.date_input("End Date", value=pd.to_datetime("2023-09-01"))

# # Fetch historical stock data and news data
# if st.button("Fetch Data"):
#     if stock not in company_data:
#         st.error(f"Stock symbol '{stock}' not found in company data.")
#     else:
#         stock_data = get_historical_data(stock, start_date, end_date)
#         st.write(f"### Historical Data for {stock} from {start_date} to {end_date}")
#         st.dataframe(stock_data)

#         news_articles = fetch_news(stock)
#         st.write(f"### News Related to {stock}")
#         if news_articles:
#             for article in news_articles:
#                 st.write(f"**Title:** {article.get('title', 'No title')}")
#                 st.write(f"**Description:** {article.get('description', 'No description')}")
#                 st.write(f"**Published At:** {article.get('publishedAt', 'No date')}")
#                 st.write("---")
#         else:
#             st.write("No news articles found.")

#         # Prepare dataset for machine learning
#         dataset = prepare_dataset(stock_data, news_articles)
        
#         # Train the model and display the accuracy
#         model, X_test, y_test, score = train_model(dataset)
#         st.write(f"### Model Accuracy: {score:.2f}")

#         # Evaluate model performance
#         evaluate_model(model, X_test, y_test)

#         # Display portfolio recommendations
#         user_portfolio = {'AAPL': 50, 'GOOGL': 20}  # Example portfolio
#         recommendations = portfolio_recommendation(user_portfolio, model, news_articles)
#         st.write("### Portfolio Recommendations:")
#         for stock, recommendation in recommendations.items():
#             st.write(f"{stock}: {recommendation}")
# Function to get user portfolio input
def get_user_portfolio():
    st.write("### Enter Your Portfolio")
    portfolio_input = st.text_area("Enter stocks and quantities (one per line, e.g., AAPL,50)", 
                                   value="AAPL,50\nGOOGL,20")
    portfolio = {}
    try:
        for line in portfolio_input.splitlines():
            if line.strip():
                ticker, quantity = line.split(',')
                portfolio[ticker.strip().upper()] = int(quantity.strip())
    except ValueError as e:
        st.error(f"Error parsing portfolio input: {e}")
    return portfolio

# Load company data
company_data = load_company_data()

# Set up the Streamlit app
st.title("Stock Market News Sentiment Bot")

# Get user input for stocks and dates
stock = st.text_input("Enter stock symbol (e.g., AAPL, GOOGL)", value="AAPL").strip().upper()
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-09-01"))

# Fetch historical stock data and news data
if st.button("Fetch Data"):
    if stock not in company_data:
        st.error(f"Stock symbol '{stock}' not found in company data.")
    else:
        stock_data = get_historical_data(stock, start_date, end_date)
        st.write(f"### Historical Data for {stock} from {start_date} to {end_date}")
        st.dataframe(stock_data)

        news_articles = fetch_news(stock)
        st.write(f"### News Related to {stock}")
        if news_articles:
            for article in news_articles:
                st.write(f"**Title:** {article.get('title', 'No title')}")
                st.write(f"**Description:** {article.get('description', 'No description')}")
                st.write(f"**Published At:** {article.get('publishedAt', 'No date')}")
                st.write("---")
        else:
            st.write("No news articles found.")

        # Prepare dataset for machine learning
        dataset = prepare_dataset(stock_data, news_articles)
        
        # Train the model and display the accuracy
        model, X_test, y_test, score = train_model(dataset)
        st.write(f"### Model Accuracy: {score:.2f}")

        # Evaluate model performance
        evaluate_model(model, X_test, y_test)

        # Get user portfolio input
        user_portfolio = get_user_portfolio()
        
        if user_portfolio:
            # Display portfolio recommendations
            recommendations = portfolio_recommendation(user_portfolio, model, news_articles)
            st.write("### Portfolio Recommendations:")
            for stock, recommendation in recommendations.items():
                st.write(f"{stock}: {recommendation}")
        else:
            st.write("No portfolio data provided.")
