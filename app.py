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
from transformers import pipeline
import torch
import numpy as np
nltk.download('wordnet')

# Download VADER lexicon
nltk.download('vader_lexicon')

from nltk.corpus import wordnet

# Enable nested event loop for Streamlit
nest_asyncio.apply()

# Initialize Sentiment Analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# ✅ Load Zero-Shot Model Efficiently & Handle Errors
@st.cache_resource
def load_zsl_model():
    try:
        return pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",  # ✅ Use BART instead of DeBERTa
	    device=-1  # ✅ Force CPU execution (avoids slow GPU fallback issues)	
           # device=0 if torch.cuda.is_available() else -1  # ✅ Runs on CPU if no GPU
        )
    except ImportError as e:
        st.error("Error: Numpy is missing. Try installing with `pip install numpy`.")	    	    
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None

# ✅ Load model once & handle failures
zsl_classifier = load_zsl_model()

# ✅ Ensure model is available before using
if zsl_classifier is None:
    st.error("Zero-shot model failed to load. Posts filtering will be skipped.")


# PRAW API credentials
REDDIT_CLIENT_ID = "5fAjWkEjNuV3IS0bDT1eFw"
REDDIT_CLIENT_SECRET = "I230bFaJWJHy58dnb3nBDvmiWdsDIg"
REDDIT_USER_AGENT = "windows:SiblingsDataApp:v1.0 (by /u/Proper-Leading-4091)"

# Define disability and sibling terms
disability_terms = [
    "22q11.2", "ADHD", "Attention Deficit Hyperactivity Disorder", "ADD", "Angelman", "Attention Deficit Disorder",
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


# ✅ Categories for Zero-Shot Filtering
zsl_labels = [
    # ✅ Existing labels
    "Growing up with a disabled sibling",
    "Supporting a disabled sibling",
    "Challenges of having a neurodivergent sibling",
    "Struggles of being a sibling to a special needs child",
    "Emotional impact of sibling disability",
    "Feeling neglected as a sibling of a special needs child"

]

# ✅ Function to Apply ZSL Filtering **AFTER** Fetching
def filter_relevant_posts(df, batch_size=20):
    """Batch process Zero-Shot Classification for CPU efficiency."""
    if df.empty:
        st.warning("Skipping Zero-Shot filtering: No posts.")
        return df
    
    if not zsl_labels:
        st.error("Zero-shot labels missing. Skipping filtering.")
        return df  

    df = df.dropna(subset=["Body"]).copy()  # Remove NaN texts
    df = df[df["Body"].str.strip() != ""]  # Remove empty strings
    relevance_scores = []

    try:
        body_texts = df["Body"].tolist()

        # ✅ Ensure we have valid inputs
        if not body_texts:
            st.warning("No valid posts to process for Zero-Shot classification.")
            return df

        # ✅ Process texts in small batches to prevent memory overload
        for i in range(0, len(body_texts), batch_size):
            batch = body_texts[i:i + batch_size]  # Get batch
            
            # ✅ Ensure batch is not empty before calling model
            if not batch:
                continue

            batch_results = zsl_classifier(batch, zsl_labels, multi_label=True)  # Process batch
            
            # ✅ Extract highest score per post in batch
            batch_scores = [max(result["scores"]) if result["scores"] else 0 for result in batch_results]
            relevance_scores.extend(batch_scores)

        # ✅ Ensure we have as many scores as rows in df
        df["Relevance_Score"] = relevance_scores[:len(df)]
        df = df[df["Relevance_Score"] > 0.35]  # Apply threshold

    except Exception as e:
        st.error(f"Zero-shot filtering failed: {e}")
    df = df.astype({col: str for col in df.select_dtypes(include=["object"]).columns})  # Ensure object columns are strings
    df = df.astype({col: int for col in df.select_dtypes(include=["int64"]).columns})  # Convert int64 to Python int

    return df


async def fetch_full_post_data(post_id):
    """Fetch the full post content, ensuring no truncation and include all attributes."""
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    try:
        post = await reddit.submission(id=post_id)

        # If the post is too long and is truncated, load the full content
        post_data = {
            "Post ID": post.id,
            "Title": post.title,
            "Body": post.selftext if len(post.selftext) < 1000 else "Text too long to display fully",
            "Upvotes": post.score,
            "Subreddit": post.subreddit.display_name,
            "Author": str(post.author),
            "Created_UTC": datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Fetch the full content if the body is large
        if len(post.selftext) >= 1000:  # Check if it's truncated
            await post.load()  # Fetch the full content
            post_data["Body"] = post.selftext  # Update with full content

        # Add optional attributes if available
        optional_attributes = {
            "Num_Comments": getattr(post, "num_comments", None),
            "Over_18": getattr(post, "over_18", None),
            "URL": getattr(post, "url", None),
            "Permalink": f"https://www.reddit.com{submission.permalink}" if hasattr(post, "permalink") else None,
            "Upvote_Ratio": getattr(post, "upvote_ratio", None),
            "Pinned": getattr(post, "stickied", None),
            "Subreddit_Subscribers": getattr(post.subreddit, "subscribers", None),
            "Subreddit_Type": getattr(post.subreddit, "subreddit_type", None),
            "Total_Awards_Received": getattr(post, "total_awards_received", None),
            "Gilded": getattr(post, "gilded", None),
            "Edited": post.edited if post.edited else None
        }

        # Add optional attributes only if they are not None
        for key, value in optional_attributes.items():
            if value is not None:
                post_data[key] = value

        return post_data

    except Exception as e:
        st.error(f"Error fetching full post data: {e}")
        return None


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
    
    
# ✅ Function to Construct Queries Properly
def generate_queries(disability_terms, sibling_terms, batch_size=5):
    """Generate optimized Reddit search queries by batching terms to reduce API calls."""
    disability_batches = [disability_terms[i:i + batch_size] for i in range(0, len(disability_terms), batch_size)]
    sibling_batches = [sibling_terms[i:i + batch_size] for i in range(0, len(sibling_terms), batch_size)]
    
    queries = []
    for disability_group in disability_batches:
        for sibling_group in sibling_batches:
            query = f"({' OR '.join(disability_group + sibling_group)})"
            queries.append(query)

    # ✅ Also fetch posts from specific sibling support subreddits
    sibling_support_subreddits = [
        "GlassChildren", "AutisticSiblings", "SiblingSupport","SpecialNeedsSiblings"
    ]
    for sub in sibling_support_subreddits:
        queries.append(f"subreddit:{sub}")  # Fetch all posts from these subs

    return queries

   
async def is_subreddit_valid(reddit, subreddit_name):
    """Check if a subreddit exists and is accessible."""
    try:
        subreddit = await reddit.subreddit(subreddit_name, fetch=True)
        if subreddit.over18:
            st.warning(f"⚠️ Skipping r/{subreddit_name} (NSFW content).")
            return False
        return True
    except Exception as e:  # ✅ Catch all exceptions instead of `asyncpraw.exceptions.NotFound`
        st.warning(f"❌ Error checking r/{subreddit_name}: {e}")
        return False


# Async function to fetch posts using PRAW
# ✅ Async Function to Fetch Posts Efficiently (No ZSL Filtering Here)
async def fetch_praw_data(queries, start_date_utc, end_date_utc, limit=50, subreddit="all"):
    """Fetch Reddit posts asynchronously for given keyword queries AND sibling support subreddits."""
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    data = []
    # ✅ Fetch sibling support subreddits in parallel with keyword queries
    sibling_posts = await fetch_sibling_subreddits()
 # ✅ Add posts from sibling subreddits
    if sibling_posts:
        data.extend(sibling_posts)	
    async def fetch_single_query(query):
        """Fetches posts for a single query asynchronously."""
        query_data = []
        try:
            subreddit_instance = await reddit.subreddit(subreddit)
            async for submission in subreddit_instance.search(query, limit=limit):
                created_date = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)

                if not (start_date_utc <= created_date <= end_date_utc):
                    continue
                sentiment, emotion = analyze_sentiment_and_emotion(submission.title + " " +  submission.selftext)


                # Fetch the full post content using the updated method
                post_data = await fetch_full_post_data(submission.id)
                if post_data:
                    query_data.append(post_data)


		    
                # ✅ Prepare the post data with mandatory fields
                post_data = {
                    "Post ID": submission.id,
                    "Title": submission.title,
                    "Body":  submission.selftext,
                    "Upvotes": submission.score,
                    "Subreddit": submission.subreddit.display_name,
                    "Author": str(submission.author),
                    "Created_UTC": created_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Sentiment": sentiment,
                    "Emotion": emotion,
                }

                # ✅ Add optional attributes if available
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

                # ✅ Only add optional attributes if they are not None
                for key, value in optional_attributes.items():
                    if value is not None:
                        post_data[key] = value

                query_data.append(post_data)

        except Exception as e:
            st.error(f"Error fetching query `{query}`: {e}")

        return query_data



    results = await asyncio.gather(
        *[fetch_single_query(q) for q in queries],  # Keyword-based search
        return_exceptions=True
    )

    # ✅ Collect valid results
    for res in results:
        if isinstance(res, list):
            data.extend(res)

    # ✅ Add posts from sibling subreddits
    if sibling_posts:
        data.extend(sibling_posts)

    # ✅ Ensure at least an empty DataFrame is returned
    return pd.DataFrame(data) if data else pd.DataFrame(columns=["Post ID", "Title", "Body", "Upvotes", "Subreddit", "Created_UTC", "Sentiment", "Emotion", "Num_Comments", "Over_18", "URL", "Permalink", "Upvote_Ratio", "Pinned", "Subreddit_Subscribers", "Subreddit_Type", "Total_Awards_Received", "Gilded", "Edited"])

async def fetch_sibling_subreddits(limit=50):
    """Fetch latest posts from valid sibling support subreddits."""
    subreddit_posts = []
    sibling_support_subreddits = [
        "GlassChildren", "AutisticSiblings", "SiblingSupport", "SpecialNeedsSiblings","DisabledSiblings"
    ]
    
    reddit = asyncpraw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    # ✅ Check which subreddits are valid before fetching
    valid_subreddits = []
    for sub in sibling_support_subreddits:
        if await is_subreddit_valid(reddit, sub):
            valid_subreddits.append(sub)

    if not valid_subreddits:
        st.warning("⚠️ No valid sibling support subreddits found.")
        return []

    for sub in valid_subreddits:
        try:
            subreddit_instance = await reddit.subreddit(sub)
            async for submission in subreddit_instance.hot(limit=limit):
                created_date = datetime.utcfromtimestamp(submission.created_utc).replace(tzinfo=timezone.utc)

                sentiment, emotion = analyze_sentiment_and_emotion(submission.title + " " + submission.selftext)

                # ✅ Prepare the post data
                post_data = {
                    "Post ID": submission.id,
                    "Title": submission.title,
                    "Body": submission.selftext,
                    "Upvotes": submission.score,
                    "Subreddit": sub,
                    "Author": str(submission.author),
                    "Created_UTC": created_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "Sentiment": sentiment,
                    "Emotion": emotion,
                }

                # ✅ Add optional attributes if available
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

                # ✅ Only add optional attributes if they are not None
                for key, value in optional_attributes.items():
                    if value is not None:
                        post_data[key] = value

                subreddit_posts.append(post_data)

        except asyncpraw.exceptions.RedditAPIException as e:
            st.warning(f"⚠️ API Error fetching r/{sub}: {e}")
        except Exception as e:
            st.error(f"⚠️ Error fetching r/{sub}: {e}")

    return subreddit_posts




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
    
@st.cache_data
def cached_fetch_data(queries, start_date_utc, end_date_utc, limit_per_query, subreddit):
    """Cache fetched Reddit posts to avoid duplicate fetching."""
    return asyncio.run(fetch_praw_data(queries, start_date_utc, end_date_utc, limit_per_query, subreddit))
# Initialize session state variables if not present
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = pd.DataFrame()


# Main Streamlit app
def main():
    st.title("The Sibling Project: Reddit Data Analysis Dashboard")
    df_cleaned = pd.DataFrame()

# Initialize session state variables if not present
    if "cleaned_data" not in st.session_state:
        st.session_state.cleaned_data = pd.DataFrame()	
 # Ensure session state variables exist
    if "data_uploaded" not in st.session_state:
        st.session_state.data_uploaded = False  # Default to False
	    
	

    # Sidebar filter inputs
    st.sidebar.header("Filters and Configuration")
    # Select All option
    select_all_disability = st.sidebar.checkbox('Select All Disabilities', value=False)
    if select_all_disability:
        selected_disabilities = st.sidebar.multiselect("Select Disability Terms", disability_terms,default=disability_terms)
    else:
        selected_disabilities = st.sidebar.multiselect("Select Disability Terms", disability_terms)

    select_all_sibling = st.sidebar.checkbox('Select All Siblings', value=False)
    if select_all_sibling:
        selected_siblings = st.sidebar.multiselect("Select Sibling Terms", sibling_terms,default=sibling_terms)
    else:
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
        
     
	
    # Convert selected dates to UTC
    start_date_utc = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
    end_date_utc = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

    st.write(f"Filtering data from {start_date_utc} to {end_date_utc}.")
    st.write(f"Fetching data from subreddit: `{subreddit_filter}`.")


    # Initialize session states
    if "post_data" not in st.session_state:
        st.session_state.post_data = pd.DataFrame()
    if "all_posts" not in st.session_state:
        st.session_state.all_posts = pd.DataFrame()
    if "comments_data" not in st.session_state:
        st.session_state.comments_data = pd.DataFrame()
    all_posts_df = pd.DataFrame()
	# Ensure session state variables exist
    if "all_posts_df" not in st.session_state:
        st.session_state.all_posts_df = pd.DataFrame()




    # Fetch Data
    if st.sidebar.button("Fetch Data"):
        queries = generate_queries(selected_disabilities, selected_siblings)

        if not queries:
            st.error("Please select at least one disability term and one sibling term.")
            return
        with st.spinner("Fetching data... Please wait."):
            start_time = time.time()  # Start the timer	   
            praw_df = cached_fetch_data(queries, start_date_utc, end_date_utc, 50, subreddit_filter)
        
            all_posts_df = praw_df

            end_time = time.time()  # End the timer
            elapsed_time = end_time - start_time  # Calculate the elapsed time	
            if exclusion_words:
                all_posts_df = all_posts_df[
                    ~all_posts_df["Title"].str.lower().str.contains("|".join(exclusion_words), na=False)
                    & ~all_posts_df["Body"].str.lower().str.contains("|".join(exclusion_words), na=False)
                ]

            if all_posts_df.empty:
                st.warning("No posts found for the selected filters.")
            else:
                # Save and display all posts
                st.session_state.all_posts = all_posts_df
                st.write(f"Total fetched records: {len(all_posts_df)}")
                st.write(f"Time taken to fetch records: {elapsed_time:.2f} seconds")  # Display the elapsed time    
                st.subheader("All Posts")
                st.dataframe(all_posts_df)
                st.sidebar.download_button("Download Raw Data", st.session_state.all_posts.to_csv(index=False), "raw_reddit_data.csv")
                colab_url = "https://colab.research.google.com/drive/1GMpH4iE0l54fIEchsM50EVb0pJJFdOy8"
                st.markdown(f"**[Process Data in Google Colab]({colab_url})**", unsafe_allow_html=True)



		# 🔹 Upload Processed CSV from Google Colab
                st.subheader("Upload Processed Data from Colab")
                uploaded_file = st.file_uploader("Upload Processed CSV", type=["csv"])
		
		# ✅ Ensure uploaded file is processed
                if uploaded_file is not None:
                    try:
                        df_cleaned = pd.read_csv(uploaded_file)
		        
                        if df_cleaned.empty:
                            st.error("❌ Uploaded CSV is empty! Please check your file.")
                        else:
                            st.session_state.cleaned_data = df_cleaned  # ✅ Store in session state
                            st.session_state.data_uploaded = True


           		    # ✅ Debug: Print Column Names and Sample Rows
                            st.write("🔍 Debugging: CSV Columns Detected:")
                            st.write(df_cleaned.columns.tolist())

				
                            st.success("✅ Processed data successfully uploaded!")
		
		            # ✅ Display the dataframe correctly
                            st.write("### Processed Data from Colab:")
                            st.dataframe(cleaned_data)
		            
		            # ✅ Add Download Button for Processed Data
                            st.sidebar.download_button(
                                "Download Processed Data",
                                df_cleaned.to_csv(index=False),
                                "final_filtered_reddit_data.csv",
                                key="download_cleaned_data"
		            )
		            # ✅ Force UI refresh to display updated data
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"⚠️ Error processing the uploaded file: {e}")


    
                # Word Cloud for Titles & Bodies
                st.subheader("Word Cloud of Post Titles")
                wordcloud = WordCloud(background_color="white").generate(" ".join(df_cleaned["Title"].dropna()))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                st.pyplot(plt)

               

                # Sentiment Over Time
                st.subheader("Sentiment Trends Over Time")
                df_cleaned["Created_UTC"] = pd.to_datetime(df_cleaned["Created_UTC"])
                sentiment_trends = df_cleaned.groupby([df_cleaned["Created_UTC"].dt.to_period("M"), "Sentiment"]).size().unstack()
                sentiment_trends.plot(kind="line", figsize=(10, 5))
                plt.title("Sentiment Trends Over Time")
                plt.xlabel("Month")
                plt.ylabel("Count")
                st.pyplot(plt)

   

                         

    # Download buttons

    # Download All Posts
    if not st.session_state.cleaned_data.empty:
        st.sidebar.download_button(
            "Download Cleaned data",
            st.session_state.all_posts.to_csv(index=False),
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
