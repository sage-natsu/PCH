from datetime import datetime, timezone
import os
import pandas as pd
import asyncpraw
import nest_asyncio
import streamlit as st
import torch
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import asyncio

# ✅ Download NLTK dependencies
nltk.download('vader_lexicon')

# ✅ Enable nested asyncio for Streamlit
nest_asyncio.apply()

# ✅ Load Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# ✅ Load Zero-Shot Model Efficiently
@st.cache_resource
def load_zsl_model():
    return pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=0 if torch.cuda.is_available() else -1
    )

# ✅ Cached Model
zsl_classifier = load_zsl_model()

# ✅ Zero-Shot Labels for Filtering
zsl_labels = [
    "Growing up with a disabled sibling",
    "Supporting a disabled sibling",
    "Challenges of having a neurodivergent sibling",
    "Struggles of being a sibling to a special needs child",
    "Caring for a sibling with autism or Down syndrome",
    "Emotional impact of sibling disability"
]

# ✅ Function to Check if a Post is Relevant
def is_sibling_experience(text):
    if not text.strip():
        return False
    result = zsl_classifier(text, zsl_labels, multi_label=True)
    return any(score > 0.35 for score in result["scores"])

# ✅ Function for Sentiment & Emotion Analysis
def analyze_sentiment_and_emotion(text):
    if pd.isna(text) or not text.strip():
        return "Neutral", "Neutral"

    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound = sentiment_scores["compound"]

    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    emotion = (
        "Happy" if "happy" in text.lower() else
        "Angry" if "angry" in text.lower() else
        "Sad" if "sad" in text.lower() else
        "Fearful" if "fear" in text.lower() else
        "Neutral"
    )
    return sentiment, emotion

# ✅ Reddit API Credentials
REDDIT_CLIENT_ID = "5fAjWkEjNuV3IS0bDT1eFw"
REDDIT_CLIENT_SECRET = "I230bFaJWJHy58dnb3nBDvmiWdsDIg"
REDDIT_USER_AGENT = "windows:SiblingsDataApp:v1.0 (by /u/Proper-Leading-4091)"

# ✅ Function to Construct Optimized Queries
def generate_queries(disability_terms, sibling_terms, batch_size=3):
    """Generate optimized Reddit search queries by batching terms."""
    disability_batches = [disability_terms[i:i + batch_size] for i in range(0, len(disability_terms), batch_size)]
    sibling_batches = [sibling_terms[i:i + batch_size] for i in range(0, len(sibling_terms), batch_size)]
    
    queries = []
    for disability_group in disability_batches:
        for sibling_group in sibling_batches:
            query = f"({' OR '.join(disability_group)}) AND ({' OR '.join(sibling_group)})"
            queries.append(query)
    
    return queries

# ✅ Async Function to Fetch Posts Efficiently
async def fetch_praw_data(queries, start_date_utc, end_date_utc, limit=50, subreddit="all"):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    data = []
    subreddit_instance = await reddit.subreddit(subreddit)

    async def fetch_single_query(query):
        """Fetches posts for a single query asynchronously."""
        query_data = []
        async for submission in subreddit_instance.search(query, limit=limit):
            created_date = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)
            if not (start_date_utc <= created_date <= end_date_utc):
                continue
            
            post_text = f"{submission.title} {submission.selftext}"
            if not is_sibling_experience(post_text):
                continue
            
            sentiment, emotion = analyze_sentiment_and_emotion(post_text)
            
            post_data = {
                "Post ID": submission.id,
                "Title": submission.title,
                "Body": submission.selftext,
                "Upvotes": submission.score,
                "Subreddit": submission.subreddit.display_name,
                "Author": str(submission.author),
                "Created_UTC": created_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Sentiment": sentiment,
                "Emotion": emotion,
                "Num_Comments": getattr(submission, "num_comments", None),
                "Permalink": f"https://www.reddit.com{submission.permalink}" if hasattr(submission, "permalink") else None
            }
            query_data.append(post_data)
        return query_data

    # ✅ Fetch all queries in parallel
    results = await asyncio.gather(*[fetch_single_query(q) for q in queries])
    for res in results:
        data.extend(res)

    return pd.DataFrame(data)

# ✅ Async Function to Fetch Comments & Append to Posts
async def fetch_comments_and_append(posts_df):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    async def fetch_single_post_comments(post_id):
        """Fetches comments for a single post."""
        try:
            post = await reddit.submission(id=post_id)
            post.comment_sort = "top"
            comments_data = []
            for comment in post.comments:
                if hasattr(comment, "body"):
                    sentiment, emotion = analyze_sentiment_and_emotion(comment.body)
                    comments_data.append({
                        "Post ID": post_id,
                        "Comment ID": comment.id,
                        "Body": comment.body,
                        "Score": comment.score,
                        "Author": str(comment.author),
                        "Created_UTC": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                        "Sentiment": sentiment,
                        "Emotion": emotion
                    })
            return comments_data
        except Exception:
            return []

    # ✅ Fetch comments for all posts in parallel
    post_ids = posts_df["Post ID"].tolist()
    comment_results = await asyncio.gather(*[fetch_single_post_comments(pid) for pid in post_ids])

    # ✅ Flatten list & convert to DataFrame
    comments_df = pd.DataFrame([item for sublist in comment_results for item in sublist])

    # ✅ Merge comments with posts
    final_df = posts_df.merge(comments_df, on="Post ID", how="left")

    return final_df

# ✅ Main Function
def main():
    st.title("The Sibling Project: Reddit Data Analysis Dashboard")
    
    # Select Terms
    selected_disabilities = st.sidebar.multiselect("Select Disability Terms", ["Autism", "ADHD", "Down Syndrome"])
    selected_siblings = st.sidebar.multiselect("Select Sibling Terms", ["Brother", "Sister", "Sibling"])
    
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")
    
    subreddit_filter = st.sidebar.text_input("Subreddit (default: all)", value="all").strip()
    
    # Fetch Data Button
    if st.sidebar.button("Fetch Data"):
        queries = generate_queries(selected_disabilities, selected_siblings)
        
        with st.spinner("Fetching posts..."):
            posts_df = asyncio.run(fetch_praw_data(queries, start_date, end_date, 50, subreddit_filter))

        if not posts_df.empty:
            with st.spinner("Fetching comments..."):
                final_df = asyncio.run(fetch_comments_and_append(posts_df))

            st.dataframe(final_df)
            st.sidebar.download_button("Download CSV", final_df.to_csv(index=False), "reddit_data.csv")

if __name__ == "__main__":
    main()
