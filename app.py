import openai
import pandas as pd
import streamlit as st

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
            model="gpt-4",
            messages=messages,
            max_tokens=50,
            temperature=0.0
        )
        sentiment_summary = response.choices[0].message['content'].strip()
        return sentiment_summary
    except Exception as e:
        st.error(f"Error: {e}")
        return "Error"

# Function to process the CSV file
def process_csv(api_key, file):
    df = pd.read_csv(file)
    if 'description' not in df.columns:
        st.error("CSV file must contain a 'description' column")
        return
    df['sentiment_summary'] = df['description'].apply(lambda x: get_sentiment_summary(api_key, x))
    st.write("Sentiment Analysis Results:")
    st.dataframe(df)
    output_file_path = "sentiment_analysis_results.csv"
    df.to_csv(output_file_path, index=False)
    with open(output_file_path, "rb") as file:
        st.download_button(label="Download Sentiment Analysis Results", data=file, file_name=output_file_path)

# Streamlit app layout
st.title("Sentiment Analysis Dashboard")
api_key = get_openai_api_key()

if api_key:
    st.write("Upload a CSV file containing a 'description' column to analyze sentiment.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        process_csv(api_key, uploaded_file)
else:
    st.warning("Please enter your OpenAI API Key in the sidebar.")
