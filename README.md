- v1.py currently bugged


# streamlit_news_analysis
streamlit_news_analysis

- You need a chatgpt API key
- Be aware of the chatgpt model you select

Newsapi:
- Has a 24h delay
- No caching, max. 100 articles per request due to free api limitations

Cost control:
- Be aware of the token you use doing the analysis when using chatgpt-turbo - its very expensive
- The max_token per sentiment score isnt optimized yet
- by setting model to chatgpt3.5-turbo the cost could be controlled

Potential services:
- Google Trends
- Alternative news sources / apiservices
- More data

- Prices
- Adding Econ-kw
- Hourly graph based on sentiment score
- Feature sentiment to price diff.
- Google Trend peaks
- Email alerts
 
if you have questions reach out to me on linkedin or twitter, same user name.

Deploy on streamlit or on your local machine (even bugged win cmd with no openai migration support):
https://benwaldner-streamlit-news-analysis-app-kscuva.streamlit.app/
