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
import torch
from transformers import pipeline
import aiohttp  #  Import for faster parallel HTTP requests

# Disable parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import nltk
nltk.download('wordnet')

# Download VADER lexicon
nltk.download('vader_lexicon')

from nltk.corpus import wordnet

# Enable nested event loop for Streamlit
nest_asyncio.apply()

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize Zero-Shot Learning (ZSL) Model
device = "cuda" if torch.cuda.is_available() else "cpu"
zsl_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)

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

# Generate query combinations
def generate_queries(disability_terms, sibling_terms, batch_size=3):
    queries = [f'"{d} {s}"' for d in disability_terms for s in sibling_terms]
    return [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]

query_batches = generate_queries(disability_terms, sibling_terms, batch_size=5)

# ZSL Labels (Categories for Filtering)
zsl_labels = [
    "Growing up with a disabled sibling",
    "Supporting a disabled sibling",
    "Challenges of having a neurodivergent sibling",
    "Positive experiences with a sibling with disabilities",
    "Struggles of being a sibling to a special needs child",
    "Caring for a sibling with autism or Down syndrome",
    "Emotional impact of sibling disability"
]

# Additional keywords for struggles
struggle_keywords = [
    "struggle", "challenge", "hardship", "difficulty", "burden", "overlooked",
    "stress", "mental health", "guilt", "responsibility", "support", "compassion",
    "caring", "caregiver", "overwhelmed", "anxious", "anxiety", "isolation", "loneliness",
    "balance", "pressure", "burnout", "neglect"
]


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

# Zero-Shot Learning Function to Filter Relevant Posts
def is_sibling_experience(text):
    result = zsl_classifier(text, zsl_labels, multi_label=True)
    return any(score > 0.35 for score in result["scores"])  # Keep only posts with strong relevance

# Async Function to Fetch Posts Using PRAW
async def fetch_praw_data(query_batches, start_date_utc, end_date_utc, limit=50, subreddit="all"):
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    data = []
    seen_post_ids = set()  # Prevent duplicates    
    subreddit_instance = await reddit.subreddit(subreddit)

	
    for batch in query_batches:  #  Iterate over query_batches correctly
        query = " OR ".join(batch)
        async for submission in subreddit_instance.search(query, limit=limit):
            if submission.id in seen_post_ids:
                continue
            seen_post_ids.add(submission.id)

            created_date = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)
            if not (start_date_utc <= created_date <= end_date_utc):
                continue

            post_text = f"{submission.title} {submission.selftext}"
            if not is_sibling_experience(post_text):  # Apply ZSL Filtering
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
            }

            # Add extra data only if available
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

            # Add optional attributes if they are not None
            for key, value in optional_attributes.items():
                if value is not None:
                    post_data[key] = value

            data.append(post_data)
        await asyncio.sleep(0.3)  # 🔹 Avoid rate limits
      # Ensure this return statement is aligned properly within the function
    	
    return pd.DataFrame(data)
    
def group_terms(terms, group_size=3):
   
    return [terms[i:i + group_size] for i in range(0, len(terms), group_size)]
        
async def fetch_and_process(query_batches, start_date_utc, end_date_utc, subreddit_filter):
    return await fetch_praw_data(query_batches, start_date_utc, end_date_utc, limit=200, subreddit=subreddit_filter)

#  Fix: Use asyncio.run() inside Streamlit function
def fetch_data_wrapper(query_batches, start_date_utc, end_date_utc, subreddit_filter):
    return asyncio.run(fetch_and_process(query_batches, start_date_utc, end_date_utc, subreddit_filter))

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
	
	
    if start_date > end_date:
        st.error("Start Date must be before End Date!")
        return

    start_date_utc = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_date_utc = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    st.write(f"Filtering data from {start_date_utc} to {end_date_utc}.")
    st.write(f"Fetching data from subreddit: `{subreddit_filter}`.")

#    disability_batches = group_terms(selected_disabilities)
#    sibling_batches = group_terms(selected_siblings)	

    # Ensure session state variables are initialized
    if "praw_df" not in st.session_state:
        st.session_state.praw_df = pd.DataFrame()  # Initialize an empty DataFrame
#    if "post_data" not in st.session_state:
#        st.session_state.post_data = pd.DataFrame()
#    if "all_posts" not in st.session_state:
#        st.session_state.all_posts = pd.DataFrame()
    if "comments_data" not in st.session_state:
        st.session_state.comments_data = pd.DataFrame()
    all_posts_df = pd.DataFrame()
    # Fetch Data
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching and Classifying Posts..."):
            praw_df = fetch_data_wrapper(query_batches, start_date_utc, end_date_utc, subreddit_filter)


            if praw_df.empty:
                st.warning("No relevant sibling experience posts found.")
            else:
                st.success(f"Fetched {len(praw_df)} posts related to sibling experiences.")
                st.session_state.praw_df = praw_df  # 🔹 Store in session state    
                st.dataframe(praw_df)

                # Filter and display relevant posts
#                relevant_posts = filter_relevant_posts(praw_df)
#                st.session_state.post_data = relevant_posts
#                st.write(f"Total relevant records: {len(relevant_posts)}")
#                st.subheader("Relevant Posts")
#                st.dataframe(relevant_posts)

                # Top 5 Subreddits
                st.subheader("Top 5 Popular Subreddits")
                top_subreddits = praw_df["Subreddit"].value_counts().head(5)
                st.bar_chart(top_subreddits)

                # Word Cloud
                st.subheader("Word Cloud of Post Titles")
                create_wordcloud(" ".join(all_posts_df["Title"].dropna()), "Post Titles Word Cloud")


                # Post Highlights
                st.subheader("Post Highlights")
                st.write("**Most Upvoted Post:**", praw_df.loc[all_posts_df["Upvotes"].idxmax()])
                st.write("**Latest Post:**", praw_df.loc[all_posts_df["Created_UTC"].idxmax()])
                st.write("**Oldest Post:**", praw_df.loc[all_posts_df["Created_UTC"].idxmin()])

                # Heatmap of Sentiment vs Emotion
                st.subheader("Heatmap of Sentiment vs Emotion")
                generate_heatmap(st.session_state.post_data)

                # 1. Sentiment and Emotion Distribution by Topic
                st.subheader("Sentiment and Emotion Distribution by Topic")
                if not all_posts_df.empty:
                    sentiment_emotion_dist = praw_df.groupby(["Sentiment", "Emotion"]).size().reset_index(name="Count")
                    fig = px.bar(sentiment_emotion_dist, x="Sentiment", y="Count", color="Emotion", title="Sentiment and Emotion Distribution by Topic")
                    st.plotly_chart(fig, use_container_width=True)

                # 2. Struggles Word Cloud for Siblings
                if not st.session_state.praw_df.empty:
                    st.subheader("Struggles Word Cloud")
                    relevant_text = " ".join(
                        st.session_state.praw_df["Body"].dropna().tolist()
                    )
                    struggle_words_only = " ".join([word for word in relevant_text.split() if word.lower() in struggle_keywords])
                    create_wordcloud(struggle_words_only, "Struggles Word Cloud")


                # 4. Most Discussed Subreddits
                st.subheader("Most Discussed Subreddits")
                if not all_posts_df.empty:
                    subreddit_count = all_posts_df["Subreddit"].value_counts().head(10).reset_index()
                    subreddit_count.columns = ["Subreddit", "Count"]
                    fig = px.bar(subreddit_count, x="Subreddit", y="Count", title="Most Discussed Subreddits")
                    st.plotly_chart(fig, use_container_width=True)
           
           
                st.subheader("Sentiment Distribution by Subreddit")       
                if not st.session_state.praw_df.empty:
                    plot_sentiment_by_subreddit(st.session_state.all_posts)
                   
                st.subheader("Emotion Radar Chart")               
                if not st.session_state.praw_df.empty:
                    plot_emotion_radar(st.session_state.all_posts)




                

    # Download buttons
    # Download Relevant Posts
    if not st.session_state.post_data.empty:
        st.sidebar.download_button(
            "Download Relevant Posts",
            st.session_state.post_data.to_csv(index=False),
            file_name="relevant_posts.csv",
            key="download_relevant_posts"  # Unique key for relevant posts
        )
    # Download All Posts
    if not st.session_state.praw_df.empty:
        st.sidebar.download_button(
            "Download All Posts",
            st.session_state.praw_df.to_csv(index=False),
            file_name="all_posts.csv",
            key="download_all_posts"  # Unique key for all posts
        )

    # Fetch comments and summaries
    st.subheader("Enter Post ID for Comments and Summarization Analysis")
    post_id = st.text_input("Post ID")
    if st.button("Fetch Comments and Summarize"):

        with st.spinner("Fetching comments... Please wait."):
            comments_data = asyncio.run(fetch_comments(post_id))

        if not comments_data.empty:
            st.success(f"Fetched {len(comments_data)} comments for Post ID: {post_id}")
            st.session_state.comments_data = comments_data  # Persist comments data
            st.dataframe(st.session_state.comments_data)


       

             # Visualizations for Comments
            st.subheader("Visualizations for Comments")
            st.bar_chart(st.session_state.comments_data["Score"].value_counts().head(10), use_container_width=True)

            st.subheader("Word Cloud for Comments")
            create_wordcloud(" ".join(st.session_state.comments_data["Body"].dropna()), "Comments Word Cloud")

            # Overall Sentiment and Emotion
            overall_sentiment = st.session_state.comments_data["Sentiment"].value_counts()
            overall_emotion = st.session_state.comments_data["Emotion"].value_counts()
            st.write("**Overall Sentiment in Comments:**", overall_sentiment)
            st.write("**Overall Emotion in Comments:**", overall_emotion)

            # Heatmap for Comments
            st.subheader("Heatmap of Sentiments and Emotions in Comments")
            generate_heatmap(st.session_state.comments_data)
            
            
            # 5. Sentiment Comparison Between Posts and Comments
            st.subheader("Sentiment Comparison Between Posts and Comments")
            if not st.session_state.comments_data.empty and not all_posts_df.empty:
                post_sentiments = all_posts_df["Sentiment"].value_counts(normalize=True).reset_index()
                post_sentiments.columns = ["Sentiment", "Percentage"]
                post_sentiments["Source"] = "Posts"

                comment_sentiments = st.session_state.comments_data["Sentiment"].value_counts(normalize=True).reset_index()
                comment_sentiments.columns = ["Sentiment", "Percentage"]
                comment_sentiments["Source"] = "Comments"

                combined_sentiments = pd.concat([post_sentiments, comment_sentiments])
                fig = px.bar(combined_sentiments, x="Sentiment", y="Percentage", color="Source", barmode="group",
                             title="Sentiment Comparison Between Posts and Comments")
                st.plotly_chart(fig, use_container_width=True)
    
    # Download Comments
    if not st.session_state.comments_data.empty:
        st.sidebar.download_button(
            "Download Comments Data",
            st.session_state.comments_data.to_csv(index=False),
            file_name="comments_data.csv",
            key="download_comments"  # Unique key for comments
        )


if __name__ == "__main__":
    main()
