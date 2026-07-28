from transformers import pipeline
import pandas as pd

# Load a BERT model specifically trained on financial news to detect Positive, Neutral, or Negative sentiment!
print("Downloading and loading AI model into memory... (This takes a few seconds on the first run)")
ai_brain = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")

def analyze_text(text: str) -> dict:
    """
    Takes a string of news text, passes it to the BERT model, and returns a clean dictionary with sentiment and confidence score
    """
    try: 
        # Pass text to the AI model
        prediction = ai_brain(text)[0]

        # Extract the sentiment label and round off the confidence score to 4 decimal places
        label = prediction['label']
        confidence = round(prediction['score'], 4)

        return {
            'sentiment': label,
            'confidence': confidence
        }
    except Exception as error:
        print(f"Error analyzing text: {error}")
        return {'sentiment': 'Error', 'confidence': 0.0}

def analyze_news_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a clean Pandas Dataframe from news_fetcher, runs BERT sentiment analysis on each article, 
    and adds sentiment and confidence columns to the table.
    """
    # Check if the DataFrame is empty (due to network error or no articles found)
    if df.empty:
        return df  # Return the empty DataFrame as is

    print(f"Running BERT AI analysis on {len(df)} articles...")

    sentiments = []
    confidences = []

    # Loop through every article's full_text in the table
    for text in df['full_text']:
        result = analyze_text(text)
        sentiments.append(result['sentiment'])
        confidences.append(result['confidence'])

    # attach our two new lists as brand new columns in the table
    df['sentiment'] = sentiments
    df['confidence'] = confidences

    return df


# --- TEST BLOCK ---    
if __name__ == "__main__":
    # We import our Extractor module directly into our Brain module to test them together!
    from news_fetcher import get_company_news

    TEST_API_KEY = "d92869d6b57b4b93af51994a0bb887c2"
    test_company = "Nvidia"
    print(f"\n1. Fetching live news for {test_company}...")
    raw_news_table = get_company_news(test_company, api_key=TEST_API_KEY)

    if not raw_news_table.empty:
        print("2. News fetched! Passing table to AI brain")
        analyzed_table = analyze_news_dataframe(raw_news_table)

        print("\n--- FINAL AI-ANALYZED NEWS TABLE ---")
        # print the title, sentiment, and confidence columns for the first five rows
        print(analyzed_table[['title', 'sentiment', 'confidence']].head())
    else:
        print("Could not retrieve news articles for analysis.")