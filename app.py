import streamlit as st
import pandas as pd
import plotly.express as px

# import our custom backend modules
from news_fetcher import get_company_news
from sentiment_analyzer import analyze_news_dataframe

#1. Configure browser tab title and layout width
st.set_page_config(page_title="AI News Sentiment Analyzer", layout="wide")

st.title("Corporate News Sentiment Dashboard")
st.write("An interactive AI pipeline powered by Hugging Face BERT and NewsAPI.")

#2. Sidebar for secure API Key input (type="password" hides the input)
st.sidebar.header("Authentication")
api_key = st.sidebar.text_input("Enter your NewsAPI Key:", type="password")

#3. Main Dashboard input box
company_name = st.text_input("Enter a company name to analyze its news sentiment:", value="Tesla")

#4. Action button to trigger analysis
if st.button("Analyze News"):
    if not api_key:
        st.error("Please enter your NewsAPI Key in the sidebar before proceeding!")
    else:
        # show a loading spinner while backeend processes the request
        with st.spinner(f"Fetching global news for '{company_name}' and running BERT AI analysis... Please Wait..."):

            # Step A: Fetch raw news using the key from the sidebar
            raw_news_df = get_company_news(company_name, api_key=api_key)

            # Step B: Check if we actually found articles
            if raw_news_df.empty:
                st.warning(f"No news articles found for '{company_name}' or API connection failed.")
            else:
                # Step C: Feed the table into our Hugging Face BERT sentiment analyzer
                analyzed_news_df = analyze_news_dataframe(raw_news_df)

                st.success(" AI Sentiment Analysis Complete!")

                # Step D: Display the table on the web dashboard!
                st.subheader(f"Latest News & AI Sentiment for: {company_name}")

                # Count the sentiment distribution
                counts = analyzed_news_df['sentiment'].value_counts()
                total_articles = len(analyzed_news_df)

                # Create 4 side-by-side metric cards
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(label="Total Articles Analyzed", value=total_articles)
                with col2:
                    st.metric(label="🟢 Positive Articles", value=counts.get('positive', 0))
                with col3:
                    st.metric(label="⚪ Neutral Articles", value=counts.get('neutral', 0))
                with col4:
                    st.metric(label="🔴 Negative Articles", value=counts.get('negative', 0))

                st.divider() # for adding a clean visual horizontal line

                # CREATE TWO TABS FOR VISUAL ANALYTICS AND FULL NEWS SPREADSHEET
                tab1, tab2 = st.tabs(["Visual Analytics", "Full news spreadsheet"])

                with tab1:
                    st.markdown("### Sentiment Proportions & Percentage Breakdown")
                    
                    # 1. Convert our sentiment counts into a clean mini-table for Plotly
                    pie_data = pd.DataFrame({
                        'Sentiment': counts.index,
                        'Articles': counts.values
                    })
                    
                    # 2. Define your explicit custom color map!
                    custom_colors = {
                        'positive': '#2ecc71',  # Bright Emerald Green
                        'neutral': '#bdc3c7',   # Silver / Clean Grey-White
                        'negative': '#e74c3c'   # Bright Crimson Red
                    }
                    
                    # 3. Build an interactive Pie/Donut Chart
                    fig = px.pie(
                        pie_data, 
                        values='Articles', 
                        names='Sentiment', 
                        color='Sentiment',
                        color_discrete_map=custom_colors,
                        hole=0.35  # A 35% hole creates a sleek professional Donut Chart!
                    )
                    
                    # 4. Force explicit percentage AND category labels inside the pie slices
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        textfont_size=15
                    )
                    
                    # 5. Render the Plotly chart inside Streamlit
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    st.markdown("### Detailed AI Analysis Table")
                    st.dataframe(analyzed_news_df[['title', 'sentiment', 'confidence', 'description']], use_container_width=True)

        