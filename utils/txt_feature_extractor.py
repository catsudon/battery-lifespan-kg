import numpy as np

def extract_features_from_sample_battery(file_path):
    """
    Reads a sample battery file and extracts features from the q_d_n values.
    
    Parameters:
        file_path (str): Path to the text file containing q_d_n values (one per line).
        
    Returns:
        dict: A dictionary containing:
            - slope_last_k_cycles: Slope over the last k cycles for each k in the list.
            - mean_grad_last_k_cycles: Mean gradient (via np.gradient) over the last k cycles.
            - trimmed_q_d_n_avg: Average value of the trimmed q_d_n array.
            - total_cycles: Total number of cycles computed as the length of the trimmed q_d_n array.
    """
    # Load the q_d_n values from file (one value per line)
    with open(file_path, 'r') as f:
        # Convert each non-empty line to a float
        q_d_n = [float(line.strip()) for line in f if line.strip()]
    
    # Convert the list to a numpy array
    q_d_n_array = np.array(q_d_n)
    
    # Trim trailing zeros from the q_d_n array (assumes zeros at the end indicate no data)
    trimmed_q_d_n = np.trim_zeros(q_d_n_array, 'b')
    
    # Compute total cycles as the length of the trimmed array
    total_cycles = len(trimmed_q_d_n)
    
    # Compute the average of the trimmed q_d_n values (if available)
    trimmed_q_d_n_avg = float(np.mean(trimmed_q_d_n)) if total_cycles > 0 else np.nan

    # Define k values for which features are computed
    k_values = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # Initialize the dictionary to hold features
    features = {}
    
    # For each k, compute the slope over the last k cycles and the mean gradient
    for k in k_values:
        if total_cycles > k:
            # Slope: (last value - value k cycles ago) divided by k
            slope = (trimmed_q_d_n[-1] - trimmed_q_d_n[-k]) / k
            
            # Mean gradient over the last k cycles using numpy.gradient
            grad = np.gradient(trimmed_q_d_n[-k:], 1)
            mean_grad = float(np.mean(grad))
        else:
            slope = np.nan
            mean_grad = np.nan
        
        features[f'slope_last_{k}_cycles'] = slope
        features[f'mean_grad_last_{k}_cycles'] = mean_grad

    # Add average of trimmed_q_d_n and total cycles to the feature set
    features['trimmed_q_d_n_avg'] = trimmed_q_d_n_avg
    features['total_cycles'] = total_cycles

    return features