from datetime import datetime, timezone
import subprocess
import time  # Import the time module
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import asyncpraw
    print("asyncpraw imported successfully!")
except ModuleNotFoundError as e:
    print(f"Error importing asyncpraw: {e}")

import streamlit as st
import pandas as pd
import asyncpraw
import nest_asyncio
import asyncio
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import seaborn as sns
import nltk
nltk.download('wordnet')

# Download VADER lexicon
nltk.download('vader_lexicon')

from nltk.corpus import wordnet

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
    "22q11.2", "ADHD", "Attention Deficit Hyperactivity Disorder", "ADD", "Angelman", "Attention Deficit Disorder"
    "Autism", "Autistic", "Asperger", "Aspergers", "Asperger's", "Aspie", "ASD", "CDKL5",
    "Cerebral Palsy", "Cognitive delay", "Cognitively delayed", "Cornelia de Lange", "CP" , "Developmental Condition", "Developmental Delay","Developmental Disorder",
    "Developmental Disability", "Disabled", "Disability", "Delayed", "Down Syndrome", "Down's Syndrome", "Downs Syndrome", "Epilepsy", "Epileptic",
    "Fetal Alcohol", "Foetal Alcohol", "FAS", "FASD", "Fragile X", "Genetic condition", "Genetic Disorder", 
     "Global Developmental Delay", "Intellectual Disability", "Intellectually disabled",
     "Intellectual impairment", "MECP2", "Mental delay", "Mentally delayed", "Mentally impaired",
     "Neurodivergent", "Neurodiverse", "Neurospicy", "Neurodevelopmental Condition", "NDC", "Neurodevelopmental Disorder", "Prader Willi",
     "Prader-Willi", "Rett Syndrome", "Rett's Syndrome", "Retts Syndrome" , "Tourette’s", "Tourette", "Tourettes", 
     "Tic Disorder", "Tics", "Williams Syndrome", "William Syndrome", "NDIS", "Hereditary"

]

sibling_terms = [
    "Brother","Brothers", "Brother’s", "Sister", "Sisters", "Sister’s","Bro", "Sis", "Sibling", "Sib", "Carer", "Guardian", "Siblings", "Sibs", "Twin"
]

# Additional keywords for struggles
struggle_keywords = [
    "struggle", "challenge", "hardship", "difficulty", "burden", "overlooked",
    "stress", "mental health", "guilt", "responsibility", "support", "compassion",
    "caring", "caregiver", "overwhelmed", "anxious", "anxiety", "isolation", "loneliness",
    "balance", "pressure", "burnout", "neglect"
]


expanded_sibling_phrases = [
    "I have a brother", "I have a sister", "I have siblings", "my twin", "my bro", "my sis", 
    "my siblings", "my younger sibling", "my older sibling", "growing up with my sibling",
    "responsibility for my sibling", "helping my sibling", "caring for my brother", 
    "caring for my sister", "my brother cares", "my sister cares", 
    "my sibling needs support"]

# Function to discover sibling-related subreddits dynamically
async def discover_sibling_related_subreddits():
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    sibling_keywords = ["siblingsupport", "glasschildren", "specialneedsiblings"]	
    sibling_related_subreddits = set()
    keywords = ["siblingsupport", "glasschildren", "specialneedsiblings"]

    for keyword in keywords:
        subreddit_search = await reddit.subreddits.search_by_name(keyword, include_nsfw=False)
        for subreddit in subreddit_search:
            sibling_related_subreddits.add(subreddit.display_name)
    return list(sibling_related_subreddits)

# Function for sentiment and emotion analysis
def analyze_sentiment_and_emotion(text):
    if pd.isna(text) or not text.strip():
        return "Neutral", "Neutral"
    sentiment_scores = sentiment_analyzer.polarity_scores(text)
    compound = sentiment_scores["compound"]
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    # Simple emotion logic
    emotion = (
        "Happy" if "happy" in text.lower() else
        "Angry" if "angry" in text.lower() else
        "Sad" if "sad" in text.lower() else
        "Fearful" if "fear" in text.lower() else
        "Neutral"
    )
    return sentiment, emotion
async def fetch_all_queries_parallel(queries, start_date_utc, end_date_utc, limit=50, subreddits=["all"]):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

async def fetch_single_query(reddit, query, start_date_utc, end_date_utc, limit, subreddits):
    data = []
    for subreddit in subreddits:
        subreddit_instance = await reddit.subreddit(subreddit)
        
        async for submission in subreddit_instance.search(query, limit=limit):
            try:
                # Ensure submission is valid before accessing attributes
                if submission is None:
                    continue

                created_date = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)
                if not (start_date_utc <= created_date <= end_date_utc):
                    continue

                sentiment, emotion = analyze_sentiment_and_emotion(submission.title + " " + submission.selftext)
                 # Collect post data
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
                }

                # Optional attributes
                optional_attributes = {
                    "Num_Comments": getattr(submission, "num_comments", None),
                    "Over_18": getattr(submission, "over_18", None),
                    "URL": getattr(submission, "url", None),
                    "Permalink": f"https://www.reddit.com{submission.permalink}" if hasattr(submission, "permalink") else None,
                    "Upvote_Ratio": getattr(submission, "upvote_ratio", None),
                    "Pinned": getattr(submission, "stickied", None),
                    "Subreddit_Subscribers": getattr(submission.subreddit, "subscribers", None),
                    "Subreddit_Type": getattr(submission.subreddit, "subreddit_type", None),
                    "Total_Awards_Received": getattr(submission, "total_awards_received", None),
                    "Gilded": getattr(submission, "gilded", None),
                    "Edited": submission.edited if submission.edited else None
                }

                for key, value in optional_attributes.items():
                    if value is not None:
                        post_data[key] = value

                data.append(post_data)
            except Exception as e:
                # Log the exception and continue
                print(f"Error processing submission: {e}")

    return data
    tasks = [fetch_single_query(query, subreddit) for query in queries for subreddit in subreddits]
    results = await asyncio.gather(*tasks)
    combined_df = pd.concat(results, ignore_index=True)
    return combined_df



def group_terms(terms, group_size=3):
    return [terms[i:i + group_size] for i in range(0, len(terms), group_size)]

# Async function to fetch comments for a specific post
async def fetch_comments(post_id, limit=100):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )
    try:
        post = await reddit.submission(id=post_id)
        post.comment_sort = "top"
        comments = []
        for comment in post.comments:
            if hasattr(comment, "body"):
                sentiment, emotion = analyze_sentiment_and_emotion(comment.body)
                comments.append({
                    "Comment ID": comment.id,
                    "Body": comment.body,
                    "Score": comment.score,
                    "Author": str(comment.author),
                    "Created_UTC": datetime.utcfromtimestamp(comment.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "Sentiment": sentiment,
                    "Emotion": emotion
                })
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"Error fetching comments: {str(e)}")
        return pd.DataFrame()

# Generate heatmap
def generate_heatmap(df):
    if "Sentiment" in df.columns and "Emotion" in df.columns:
        heatmap_data = df.pivot_table(index="Sentiment", columns="Emotion", aggfunc="size", fill_value=0)
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="coolwarm")
        plt.title("Heatmap of Sentiment vs Emotion")
        st.pyplot(plt)

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



def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").lower())
    return list(synonyms)

# Extend struggle_keywords with synonyms
#def expand_struggle_keywords(keywords):
#    expanded_keywords = set(keywords)
#    for word in keywords:
 #       expanded_keywords.update(get_synonyms(word))
#    return list(expanded_keywords)
    
    

# Initial struggle keywords
#struggle_keywords = [
#    "struggle", "challenge", "hardship", "difficulty", "burden", "overlooked"
#]

# Dynamically expand struggle_keywords
#struggle_keywords = expand_struggle_keywords(struggle_keywords)
#print(f"Expanded struggle keywords: {struggle_keywords}")


# filter_relevant_posts function
# def filter_relevant_posts(df):
#    expanded_emotions = ["Sad", "Angry", "Fearful", "Neutral", "Confused", "Overwhelmed", "Stressed"]

    # Check for sibling context
#    df["Sibling_Context"] = df["Body"].str.contains("|".join(sibling_terms), case=False, na=False)

    # Apply filters
#    filtered_df = df[
 #       df["Sibling_Context"] & 
 #       (df["Sentiment"].isin(["Negative", "Neutral", "Positive"])) & 
  #      (df["Emotion"].isin(expanded_emotions)) & 
  #      (df["Body"].str.contains("|".join(struggle_keywords), case=False, na=False))
#    ]
 #   return filtered_df

    
def plot_emotion_radar(df):
    if df.empty:
        st.warning("No data available for emotion radar chart.")
        return

    emotion_counts = df['Emotion'].value_counts(normalize=True)
    categories = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']
    values = [emotion_counts.get(cat, 0) for cat in categories]

    fig = px.line_polar(data_frame=df, theta=categories, r=values, title="Emotion Radar Chart", line_close=True)
    st.plotly_chart(fig)


def plot_sentiment_by_subreddit(df):
    if df.empty:
        st.warning("No data available for sentiment analysis by subreddit.")
        return

    sentiment_subreddit = df.groupby(['Subreddit', 'Sentiment']).size().reset_index(name='Count')
    fig = px.bar(sentiment_subreddit, x='Subreddit', y='Count', color='Sentiment',
                 title="Sentiment Distribution by Subreddit", barmode='group')
    st.plotly_chart(fig)
    
   

# Main Streamlit app
def main():
    st.title("The Sibling Project: Reddit Data Analysis Dashboard")

    # Sidebar filter inputs
    st.sidebar.header("Filters and Configuration")
    selected_disabilities = st.sidebar.multiselect("Select Disability Terms", disability_terms)
    selected_siblings = st.sidebar.multiselect("Select Sibling Terms", sibling_terms)
    start_date = st.sidebar.date_input("Start Date")
    end_date = st.sidebar.date_input("End Date")

    # Subreddit filter
    subreddit_filter = st.sidebar.text_input("Subreddit (default: all)", value="all").strip()	
    st.sidebar.write("Specify subreddits or leave 'all' for general search across Reddit.")
    st.sidebar.subheader("Exclusion Filter")
    exclusion_input = st.sidebar.text_input("Exclude posts containing these words (comma-separated):", value="")
    st.sidebar.write("Provide a comma-separated list of words to exclude posts containing them in the title or body.")
    exclusion_words = [word.strip().lower() for word in exclusion_input.split(",") if word.strip()]
    st.sidebar.write(f"Exclusion Words: {exclusion_words if exclusion_words else 'None'}")
	
    # Session states for fetched data
    if "post_data" not in st.session_state:
        st.session_state.post_data = pd.DataFrame()
    if "comments_data" not in st.session_state:
        st.session_state.comments_data = pd.DataFrame()
	
    if start_date > end_date:
        st.error("Start Date must be before End Date!")
        

    # Convert selected dates to UTC
    start_date_utc = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_date_utc = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    st.write(f"Filtering data from {start_date_utc} to {end_date_utc}.")
    st.write(f"Fetching data from subreddit: `{subreddit_filter}`.")

    disability_batches = group_terms(selected_disabilities)
    sibling_batches = group_terms(selected_siblings)
    sibling_subreddits = []  # Initialize empty
    if st.sidebar.button("Discover Sibling Subreddits"):
        sibling_subreddits = asyncio.run(discover_sibling_related_subreddits())
        st.write(f"Discovered sibling-related subreddits: {', '.join(sibling_subreddits)}")

    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data... Please wait."):
            queries = [
                f"({' OR '.join(disability)}) AND ({' OR '.join(sibling)}) OR ({phrase})"
                for disability in disability_batches
                for sibling in sibling_batches
                for phrase in expanded_sibling_phrases
            ]
        # Split queries into smaller batches to avoid overloading asyncio
        batch_size = 10  # Adjust this based on performance
        query_batches = [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
        
        subreddits_to_search = ["all"] + sibling_subreddits if sibling_subreddits else ["all"]  # Search globally + sibling subreddits

        all_posts_df = pd.DataFrame()  # Initialize the result dataframe

        # Fetch data batch by batch
        for batch in query_batches:
            try:
                batch_results = asyncio.run(fetch_all_queries_parallel(batch, start_date_utc, end_date_utc, limit=50, subreddits=subreddits_to_search))
                all_posts_df = pd.concat([all_posts_df, batch_results], ignore_index=True)
            except Exception as e:
                st.error(f"Error fetching batch: {e}")

        if all_posts_df.empty:
            st.warning("No posts found for the selected filters.")
        else:
            st.write(f"Total fetched records: {len(all_posts_df)}")
            st.dataframe(all_posts_df)
            # Visualizations
            if not all_posts_df.empty:
                st.subheader("Word Cloud")
                create_wordcloud(" ".join(all_posts_df["Title"].dropna()), "Post Titles")

                st.subheader("Heatmap of Subreddit Activity")
                generate_heatmap(all_posts_df)

    # Fetch comments for specific post
    st.subheader("Fetch Comments for a Specific Post")
    post_id = st.text_input("Enter Post ID")
    if st.button("Fetch Comments"):
        with st.spinner("Fetching comments..."):
            comments_df = asyncio.run(fetch_comments(post_id))
            st.session_state.comments_data = comments_df
            st.success(f"Fetched {len(comments_df)} comments.")
            st.write(comments_df)

    # Download buttons for fetched data
    if not st.session_state.post_data.empty:
        st.sidebar.download_button("Download Posts Data", st.session_state.post_data.to_csv(index=False), "posts_data.csv")
    if not st.session_state.comments_data.empty:
        st.sidebar.download_button("Download Comments Data", st.session_state.comments_data.to_csv(index=False), "comments_data.csv")
if __name__ == "__main__":
    main()
