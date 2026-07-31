from newsapi import NewsApiClient
import pandas as pd
import streamlit as st  # <--- Import Streamlit so we can use its caching tool!

# Add the cache shield!
@st.cache_data(ttl=3600) #  This saves the API result in memory for 3600 seconds (1 hour).
def get_company_news(company_name: str, api_key: str):
    """
    Fetches news for a given company with defensive error handling.
    """
    try:

        # it searches for whatever 'company_name' is passed to it!
        newsapi = NewsApiClient(api_key=api_key)
        response = newsapi.get_everything(q=company_name, language='en')

        # --- DIAMOND 1: Network / Status Check ---
        if response.get('status') != 'ok':
            print(f"API Error: Could not connect or retrieve data from NewsAPI.")
            return pd.DataFrame()  # Return an empty DataFrame in case of network error
        
        # --- DIAMOND 2: Article Presence Check ---
        if not response.get('articles'):
            print(f"No articles found for: {company_name}")
            return pd.DataFrame()  # Return an empty DataFrame if no articles are found

        # create an empty list before the loop starts
        cleaned_articles = []

        for article in response['articles']:
            title = article.get('title')
            description = article.get('description')

            # If the title is missing, or it says "[removed]", or if the description is missing, skip this article
            if not title or title == "[removed]" or not description:
                continue

            # We combine title and description into 'full_text' so the AI reads the whole story
            cleaned_articles.append({
                'title': title,
                'description': description,
                'full_text': f"{title} {description}"
            })

        # Return the cleaned articles as a DataFrame
        df = pd.DataFrame(cleaned_articles)
        return df
    except Exception as error:
        print(f"An error occurred while fetching news for {company_name}: {error}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error 

# Test our function with a sample company name
if __name__ == "__main__":
    test_table = get_company_news("Microsoft", api_key="MY_API_KEY") 
    print("\n--- FUNCTION TEST WITH MICROSOFT ---")
    print(test_table.head())