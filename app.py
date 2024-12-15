import streamlit as st
import pandas as pd
import asyncpraw
import nest_asyncio
import asyncio
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import seaborn as sns
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Enable nested event loop for Streamlit
nest_asyncio.apply()

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# PRAW API credentials
REDDIT_CLIENT_ID = "5fAjWkEjNuV3IS0bDT1eFw"
REDDIT_CLIENT_SECRET = "I230bFaJWJHy58dnb3nBDvmiWdsDIg"
REDDIT_USER_AGENT = "windows:SiblingsDataApp:v1.0 (by /u/Proper-Leading-4091)"

# Define disability and sibling terms
disability_terms = [
    "22q11.2 Deletion Syndrome", "ADHD", "Angelman Syndrome", "Autism", "Asperger",
    "CDKL5 Deficiency Disorder", "Cerebral Palsy", "Cornelia de Lange Syndrome",
    "Developmental Delay", "Developmental Disability", "Down Syndrome", "Epilepsy"
]

sibling_terms = [
    "Brother", "Sister", "Bro", "Sis", "Sibling", "Sib", "Carer", "Guardian"
]

# Function for sentiment analysis
def analyze_sentiment(text):
    if pd.isna(text) or not text.strip():
        return "Neutral"
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound = sentiment_scores["compound"]
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Async function to fetch posts using PRAW
async def fetch_praw_data(query, limit=50):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    data = []
    subreddit = await reddit.subreddit("all")
    async for submission in subreddit.search(query, limit=limit):
        sentiment = analyze_sentiment(submission.title + " " + submission.selftext)
        data.append({
            "Post ID": submission.id,
            "Title": submission.title,
            "Body": submission.selftext,
            "Upvotes": submission.score,
            "Subreddit": submission.subreddit.display_name,
            "Author": str(submission.author),
            "Created_UTC": datetime.utcfromtimestamp(submission.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Sentiment": sentiment
        })
    return pd.DataFrame(data)

# Generate word cloud
def create_wordcloud(text, title):
    if not text.strip():
        st.warning("No text data available for Word Cloud!")
        return
    wordcloud = WordCloud(background_color="white").generate(text)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    st.pyplot(plt)

# Main Streamlit app
def main():
    st.title("Simplified Sibling Project: Reddit Data Analysis Dashboard")

    # Sidebar filter inputs
    st.sidebar.header("Filters and Configuration")
    selected_disabilities = st.sidebar.multiselect("Select Disability Terms", disability_terms)
    selected_siblings = st.sidebar.multiselect("Select Sibling Terms", sibling_terms)

    # Initialize session states
    if "post_data" not in st.session_state:
        st.session_state.post_data = pd.DataFrame()

    # Fetch Data
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data... Please wait."):
            all_posts_df = pd.DataFrame()
            for disability in selected_disabilities:
                for sibling in selected_siblings:
                    query = f"({disability}) AND ({sibling})"
                    praw_df = asyncio.run(fetch_praw_data(query, limit=50))
                    all_posts_df = pd.concat([all_posts_df, praw_df], ignore_index=True)
            
            if all_posts_df.empty:
                st.warning("No posts found for the selected filters.")
            else:
                # Save and display all posts
                st.session_state.post_data = all_posts_df
                st.write(f"Total fetched records: {len(all_posts_df)}")
                st.subheader("All Posts")
                st.dataframe(all_posts_df)

                # Word Cloud
                st.subheader("Word Cloud of Post Titles")
                create_wordcloud(" ".join(all_posts_df["Title"].dropna()), "Post Titles Word Cloud")

    # Download Relevant Posts
    if not st.session_state.post_data.empty:
        st.sidebar.download_button(
            "Download Posts",
            st.session_state.post_data.to_csv(index=False),
            file_name="posts.csv"
        )

if __name__ == "__main__":
    main()
