import streamlit as st
import pandas as pd
import numpy as np
import os
from neo4j import GraphDatabase


# --- Neo4j Configuration ---
NEO4J_URI = "neo4j+s://3b31837b.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "D4W3Zfi44nAJfStBuxSE2DpKhlk_nMP6ybEjvOX5qxw"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# --- Function to Extract `slope_last_500_cycles` ---
def extract_slope_from_data(file_content):
    """
    Extracts the `slope_last_500_cycles` feature from the uploaded file data.
    
    Args:
        file_content (str): Raw text content from the uploaded file.

    Returns:
        float: The extracted slope value.
    """
    # Compute the mean gradient for different last-k cycles
    last_k_th_cycles_list = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    slope_results = {}

    for k in last_k_th_cycles_list:
        if len(file_content) >= k:
            slope = np.gradient(file_content[-k:], 1)  # Compute gradient
            mean_slope = np.mean(slope)  # Take the mean
        else:
            mean_slope = np.nan  # Not enough data

        slope_results[f'mean_grad_last_{k}_cycles'] = mean_slope
    
    
    return slope_results

# --- Neo4j Query Function ---
# @tool
def query_similar_batteries(test_slope: float, threshold: int = 10, top_k: int = 3, scale_factor: int = 1e6):
    """
    Query Neo4j graph DB to find batteries similar to a given test battery based on a feature.
    
    Args:
        test_slope (float): The extracted slope value from uploaded data.
        threshold (int): The max similarity difference allowed (default: 10).
        top_k (int): The number of closest matches to return (default: 3).
    Returns:
        List[Dict]: A list of the closest matching batteries and their charging policies.
    """
    with driver.session() as session:
        query = """
        MATCH (cp:ChargingPolicy)-[:USED_BY]->(b:Battery)
        WHERE abs(b.slope_last_500_cycles - $test_slope) < $threshold
        RETURN b.battery_id AS battery_id, 
               b.slope_last_500_cycles AS feature_value, 
               abs(b.slope_last_500_cycles - $test_slope) AS similarity,
               cp.charging_policy AS charging_policy
        ORDER BY similarity ASC
        LIMIT $top_k
        """

        results = session.run(query, test_slope=test_slope, threshold=threshold, top_k=top_k)
        
        output = []
        for record in results:
            output.append({
                "battery_id": record["battery_id"],
                "feature_value": record["feature_value"],
                "similarity": record["similarity"] * scale_factor,
                "charging_policy": record["charging_policy"] or "Unknown"
            })
        
        return output

# --- Streamlit UI ---
st.title("🔋 Battery Similarity Finder")

# File Upload Section
uploaded_file = st.file_uploader("Upload a battery data file (.txt)", type=["txt"])


selected_slope_window = st.selectbox(
    "Select slope window (cycles):",
    [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    index=6  # Default to 500
)


# User sets similarity threshold
scaled_threshold = st.number_input(
    "Set Similarity Threshold (scaled):", 
    min_value=1, 
    max_value=5000, 
    value=10, 
    step=10,  # Allows stepping in increments of 100
    format="%d"  # Ensures integer input
)

# Scale it back for querying
actual_threshold = scaled_threshold / 1e6  # Convert back to small range

# User sets number of top results
top_k = st.slider("Number of top similar batteries to return:", min_value=1, max_value=10, value=3)



if uploaded_file:
    # Read file and decode if necessary
    file_content = uploaded_file.read().decode("utf-8")
    
    # Ensure file_content is a string
    if isinstance(file_content, list):  
        file_content = "\n".join(file_content)  # Convert list to string

    # Convert to a list of floats
    try:
        data_list = [float(line.strip().replace(',', '')) for line in file_content.split("\n") if line.strip()]
        st.success("✅ File uploaded successfully!")
        st.write(f"📊 First 10 values: {data_list[:10]}")  # Show a preview
    except ValueError as e:
        st.error(f"⚠️ Error reading file: {e}")
        
    # Extract feature
    slope_value = extract_slope_from_data(data_list)
    slope_value = slope_value[f"mean_grad_last_{selected_slope_window}_cycles"]
    
    st.write(f"📊 Extracted Feature: **slope_last_{selected_slope_window}_cycles = {slope_value:.5f}**")
    
    # Query Neo4j
    results = query_similar_batteries(slope_value, actual_threshold, top_k)
    
    # Display Results
    if results:
        st.subheader(f"🔍 Closest Matches for uploaded data:")
        df = pd.DataFrame(results)
        st.dataframe(df)
    else:
        st.warning("No similar batteries found in the database.")
