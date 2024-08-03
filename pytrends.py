import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import requests
from datetime import datetime, timedelta
from pytrends.request import TrendReq  # Ensure this import works

# Function to get API keys and parameters from user input
def get_api_keys_and_params():
    st.sidebar.header("API Keys and Parameters")
    
    # OpenAI API Key
    openai_api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    
    # NewsAPI Key and Parameters
    news_api_key = st.sidebar.text_input("Enter your NewsAPI API Key:", type="password")
    keyword = st.sidebar.text_input("Enter search keyword:", value="bitcoin")
    from_date = st.sidebar.date_input("From Date:", value=datetime.now() - timedelta(days=7))
    to_date = st.sidebar.date_input("To Date:", value=datetime.now())
    
    # Google Trends Parameters
    google_trends_keyword = st.sidebar.text_input("Google Trends Keyword:", value="bitcoin")
    google_trends_timeframe = st.sidebar.text_input("Google Trends Timeframe:", value="today 7-d") # e.g., "today 7-d"
    
    return openai_api_key, news_api_key, keyword, from_date, to_date, google_trends_keyword, google_trends_timeframe

# Function to fetch news data from NewsAPI
def fetch_news(news_api_key, keyword, from_date, to_date):
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': keyword,
        'from': from_date.strftime('%Y-%m-%d'),
        'to': to_date.strftime('%Y-%m-%d'),
        'sortBy': 'publishedAt',
        'apiKey': news_api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        if articles:
            return pd.DataFrame(articles)
        else:
            st.warning("No articles found for the specified criteria.")
            return pd.DataFrame()
    else:
        st.error("Failed to fetch news. Please check your API key and parameters.")
        return pd.DataFrame()

# Function to fetch Google Trends data
def fetch_google_trends(google_trends_keyword, google_trends_timeframe):
    pytrends = TrendReq()
    pytrends.build_payload(kw_list=[google_trends_keyword], timeframe=google_trends_timeframe)
    data = pytrends.interest_over_time()
    if not data.empty:
        return data.reset_index()
    else:
        st.warning("No Google Trends data found for the specified keyword and timeframe.")
        return pd.DataFrame()

# Function to get sentiment summary
def get_sentiment_summary(api_key, text):
    openai.api_key = api_key
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Analyze the sentiment of the following text and provide a brief summary:\n\n{text}\n\nSentiment Summary:"}
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=50,
            temperature=0.0
        )
        sentiment_summary = response.choices[0].message['content'].strip()
        return sentiment_summary
    except Exception as e:
        st.error(f"Error: {e}")
        return "Error"

# Function to process the DataFrame and compute hourly sentiment scores
def process_df(api_key, df):
    if 'description' not in df.columns or 'publishedAt' not in df.columns:
        st.error("DataFrame must contain 'description' and 'publishedAt' columns.")
        return
    
    # Get sentiment summary
    df['sentiment_summary'] = df['description'].apply(lambda x: get_sentiment_summary(api_key, x))
    
    # Compute sentiment score
    df['sentiment'] = df['sentiment_summary'].apply(lambda x: TextBlob(x).sentiment.polarity if x else None)
    
    # Convert 'publishedAt' to datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    
    # Extract date and hour
    df['date'] = df['publishedAt'].dt.date
    df['hour'] = df['publishedAt'].dt.hour
    
    # Calculate hourly sentiment scores
    hourly_sentiment = df.groupby(['date', 'hour'])['sentiment'].mean().reset_index()
    hourly_sentiment.columns = ['Date', 'Hour', 'Average Sentiment']
    
    # Display results and plot
    st.write("Sentiment Analysis Results:")
    st.dataframe(df)
    
    st.write("Hourly Sentiment Scores:")
    st.dataframe(hourly_sentiment)
    
    # Plot hourly sentiment scores
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=hourly_sentiment, x='Hour', y='Average Sentiment', hue='Date', ax=ax)
    ax.set_title("Hourly Average Sentiment")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Average Sentiment")
    plt.legend(title='Date')
    st.pyplot(fig)
    
    # Save and provide download link for results
    output_file_path = "sentiment_analysis_results.csv"
    df.to_csv(output_file_path, index=False)
    with open(output_file_path, "rb") as file:
        st.download_button(label="Download Sentiment Analysis Results", data=file, file_name=output_file_path)

# Streamlit app layout
st.title("Sentiment Analysis Dashboard")

# Get API keys and parameters
openai_api_key, news_api_key, keyword, from_date, to_date, google_trends_keyword, google_trends_timeframe = get_api_keys_and_params()

if openai_api_key:
    # Sidebar radio button to choose between uploading a file, fetching news, or Google Trends
    option = st.sidebar.radio("Choose an option", ("Upload CSV File", "Fetch News", "Fetch Google Trends"))

    if option == "Upload CSV File":
        st.write("Upload a CSV file containing 'description' and 'publishedAt' columns to analyze sentiment.")
        uploaded_file = st.file_uploader("Drag and drop a file here or click to browse", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            process_df(openai_api_key, df)
    
    elif option == "Fetch News":
        if news_api_key:
            if st.button("Fetch News"):
                df = fetch_news(news_api_key, keyword, from_date, to_date)
                if not df.empty:
                    # Ensure the DataFrame has the necessary columns
                    if 'description' not in df.columns:
                        df['description'] = df['title'] + " " + df['description'].fillna('')
                    if 'publishedAt' not in df.columns:
                        df['publishedAt'] = df['publishedAt']
                    
                    process_df(openai_api_key, df)
        else:
            st.warning("Please enter your NewsAPI key in the sidebar.")
    
    elif option == "Fetch Google Trends":
        if google_trends_keyword:
            df = fetch_google_trends(google_trends_keyword, google_trends_timeframe)
            if not df.empty:
                st.write("Google Trends Data:")
                st.dataframe(df)
                
                # Plot Google Trends data
                fig, ax = plt.subplots(figsize=(12, 6))
                if 'date' in df.columns:
                    sns.lineplot(data=df, x='date', y=df.columns[1], ax=ax)  # Assumes the second column is the trends data
                    ax.set_title(f"Google Trends Over Time for '{google_trends_keyword}'")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Trends")
                    st.pyplot(fig)
        else:
            st.warning("Please enter a Google Trends keyword in the sidebar.")
else:
    st.warning("Please enter your OpenAI API key in the sidebar.")
