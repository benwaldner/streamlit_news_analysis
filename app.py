import openai
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Function to get OpenAI API key from user input
def get_openai_api_key():
    st.sidebar.header("OpenAI API Key")
    api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")
    return api_key

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

# Function to process the CSV file and compute hourly sentiment scores
def process_csv(api_key, file):
    df = pd.read_csv(file)
    if 'description' not in df.columns:
        st.error("CSV file must contain a 'description' column")
        return
    
    if 'publishedAt' not in df.columns:
        st.error("CSV file must contain a 'publishedAt' column")
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
api_key = get_openai_api_key()

if api_key:
    st.write("Upload a CSV file containing 'description' and 'publishedAt' columns to analyze sentiment.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        process_csv(api_key, uploaded_file)
else:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
