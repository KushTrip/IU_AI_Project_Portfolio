# Corporate News Sentiment Analyzer

An end-to-end Natural Language Processing (NLP) pipeline that dynamically fetches global corporate news and performs real-time sentiment analysis using a fine-tuned Hugging Face BERT model. 

Designed as a portfolio project for the B.Sc. Data Science curriculum, this application demonstrates modular software architecture, API integration, and advanced machine learning deployment.

## System Architecture

This project is built using a three-tier modular architecture:

1. **The Extractor (`news_fetcher.py`):** Integrates with [NewsAPI](https://newsapi.org/) to ingest live, global headlines and article descriptions based on corporate keyword queries. Built with defensive dictionary parsing and error handling to ensure pipeline stability.
2. **The NLP Brain (`sentiment_analyzer.py`):** Utilizes the `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` model via [Hugging Face Transformers](https://huggingface.co/). This specialized RoBERTa neural network evaluates financial context and classifies articles across three mandatory valence levels: **Positive, Neutral, and Negative**.
3. **The Executive Dashboard (`app.py`):** A [Streamlit](https://streamlit.io/) graphical user interface (GUI) featuring interactive metric KPI cards, data caching for API rate-limit protection, and custom [Plotly](https://plotly.com/) donut charts for executive-level visual analytics.

## Key Features

* **Real-Time Data Ingestion:** Fetches up to 100 live articles per query.
* **Financial NLP Classification:** Infers sentiment context specifically tuned for corporate news, ignoring fluff and identifying genuine market-moving sentiment.
* **Confidence Scoring:** Outputs mathematical probability scores for every AI prediction.
* **API Shielding (Caching):** Utilizes Streamlit's `@st.cache_data` to memorize API responses for 1 hour, minimizing redundant internet calls and protecting developer rate limits.
* **Interactive Visualizations:** Renders custom-colored Plotly donut charts for immediate percentage breakdowns of market sentiment.

## Installation & Local Setup

**1. Clone the repository**
```
git clone [https://github.com/YOUR_GITHUB_USERNAME/IU_AI_Project_Portfolio.git](https://github.com/YOUR_GITHUB_USERNAME/IU_AI_Project_Portfolio.git)
cd IU_AI_Project_Portfolio
```

**2. Create and activate a virtual environment**
```
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```
pip install -r requirements.txt
```

## Usage
To launch the web application, execute the following command in your terminal:
```
streamlit run app.py
```
1. Open your browser to http://localhost:8501.
2. Enter your free NewsAPI Key in the secure sidebar authentication box.
3. Type any company name (e.g., Nvidia, Tesla, Apple) and click Analyze News.

## License & Academic Integrity
This project was developed for academic evaluation. The AI model utilized is open-source via Hugging Face. Do not use this application for actual financial trading or automated investment decisions.

