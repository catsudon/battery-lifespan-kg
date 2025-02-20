# evaluate.py

import os
import pandas as pd
from io import StringIO

from pipeline import run_pipeline

# Configurable parameter: top_k (must be less than 15)
TOP_K = 5  # Adjust this as needed

# Paths
TESTSET_CSV_PATH = "/home/jaf/battery-lifespan-kg/eval/evaluator_question.csv"  # Update with your CSV file path
BATTERY_FILES_DIR = "/home/jaf/battery-lifespan-kg/resources/testset"

def parse_response(response, top_k):
    """
    Parse the LLM response to extract a list of battery IDs.
    Expects a comma-separated list of battery IDs.
    
    :param response: The LLM response string.
    :param top_k: Number of battery IDs expected.
    :return: List of extracted battery IDs (up to top_k items).
    """
    try:
        # Split by comma and trim whitespace
        candidates = [item.strip() for item in response.split(",")]
        # Filter out any empty strings
        candidates = [cand for cand in candidates if cand]
        return candidates[:top_k]
    except Exception as e:
        print(f"Error parsing response: {e}. Response was: {response}")
        return []

def normalize_str(s):
    return s.strip().lower()

def evaluate_testset():
    # Read the testset CSV
    df = pd.read_csv(TESTSET_CSV_PATH)
    
    total_comparisons = 0
    total_correct = 0
    
    # Iterate over each test case (row)
    for idx, row in df.iterrows():
        test_battery_id = row["TEST_BATTERY_ID"]
        battery_file_path = os.path.join(BATTERY_FILES_DIR, f"{test_battery_id}.txt")
        
        try:
            with open(battery_file_path, "r") as f:
                file_content = f.read()
        except Exception as e:
            print(f"Error reading file for battery {test_battery_id}: {e}")
            continue
        
        # Get expected battery IDs for positions 1 to TOP_K
        expected_ids = []
        for i in range(1, TOP_K+1):
            col_name = f"{i}_Most_Similar_Battery_ID"
            if col_name in row:
                expected_ids.append(str(row[col_name]).strip().lower())
            else:
                expected_ids.append("")  # In case the column is missing
        
        # Identify all sample question columns (e.g., SAMPLE_QUESTION_1, SAMPLE_QUESTION_2, etc.)
        sample_question_cols = [col for col in df.columns if col.startswith("SAMPLE_QUESTION")]
        
        # Evaluate each sample question as an individual data point
        for question_col in sample_question_cols:
            base_question = str(row[question_col]).strip()
            # Append instruction to force the list format with top-K results
            modified_question = (
                f"{base_question} return the top-{TOP_K} results from the most similar first. "
                "Please respond with a comma-separated list of battery IDs only."
            )
            
            # Create a file-like object from the file content
            from io import StringIO
            uploaded_file = StringIO(file_content)
            
            # Get LLM response
            response = run_pipeline(modified_question, uploaded_file)
            # Parse the response into a list of battery IDs
            extracted_ids = parse_response(response, TOP_K)
            
            # Initialize score for this sample question
            sample_correct = 0
            for rank in range(TOP_K):
                total_comparisons += 1
                # If there is no candidate for this rank, count as not matched
                if rank >= len(extracted_ids):
                    print(f"Test case {test_battery_id}, question '{base_question}': Missing candidate at rank {rank+1}. Expected: {expected_ids[rank]}")
                    continue
                
                if normalize_str(extracted_ids[rank]) == normalize_str(expected_ids[rank]):
                    sample_correct += 1
                    total_correct += 1
                else:
                    print(f"Test case {test_battery_id}, question '{base_question}': Mismatch at rank {rank+1}. Expected: {expected_ids[rank]}, Got: {extracted_ids[rank]}")
            
            # Report score for this sample question
            accuracy = (sample_correct / TOP_K) * 100
            print(f"Test case {test_battery_id}, question '{base_question}' accuracy: {accuracy:.2f}%")
    
    overall_accuracy = (total_correct / total_comparisons) * 100 if total_comparisons > 0 else 0
    print(f"\nOverall accuracy across all sample questions: {overall_accuracy:.2f}%")

evaluate_testset()
