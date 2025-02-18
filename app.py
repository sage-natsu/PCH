def filter_relevant_posts(df, batch_size=10):
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
    
    return df
