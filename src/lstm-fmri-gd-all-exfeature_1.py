import tensorflow as tf
import numpy as np
import joblib
import os
import scipy.io
import pandas as pd
import h5py

def extract_lstm_features(model_path: str, scaler_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Loads a pre-trained Keras LSTM model and a scaler to extract features
    from an intermediate dense layer.

    Args:
        model_path (str): The file path to the saved Keras model (.h5 file).
        scaler_path (str): The file path to the saved scikit-learn scaler (.joblib file).
        input_data (np.ndarray): The input data with the shape (samples, timesteps, features/voxels).
                                 For your case, this is (samples, 5, voxels).

    Returns:
        np.ndarray: The extracted features with the shape (samples, 128).
    """
    # --- 1. Input Validation ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")
    if input_data.ndim != 3:
        raise ValueError(f"Input data must be 3D (samples, timesteps, features), but got shape {input_data.shape}")

    print("Loading model and scaler...")
    # --- 2. Load the Model and Scaler ---
    # Load the entire pre-trained model
    original_model = tf.keras.models.load_model(model_path)
    # Load the scaler that was fitted on the training data
    scaler = joblib.load(scaler_path)

    # --- 3. Preprocess the Input Data ---
    # The data must be scaled exactly as it was during training
    num_samples, num_timesteps, num_voxels = input_data.shape
    
    # a. Flatten for the scaler: (samples, timesteps * voxels)
    input_data_flatten = input_data.reshape(num_samples, -1)
    
    # b. Apply the loaded scaler
    input_data_scaled = scaler.transform(input_data_flatten)
    
    # c. Reshape back to LSTM format: (samples, timesteps, voxels)
    input_data_lstm = input_data_scaled.reshape(num_samples, num_timesteps, num_voxels)
    print(f"Preprocessed data shape: {input_data_lstm.shape}")

    # --- 4. Create the Feature Extraction Model ---
    # We create a new model that takes the same input as the original
    # but outputs the activations from our desired layer ('feature_extractor').

    feature_extractor_model = tf.keras.Model(
        inputs=original_model.input,
        outputs=original_model.get_layer('feature_extractor').output
    )
    
    # --- 5. Extract Features ---
    print("Extracting features...")
    extracted_features = feature_extractor_model.predict(input_data_lstm)
    print(f"Extraction complete. Feature shape: {extracted_features.shape}")

    return extracted_features, original_model, input_data_lstm

split_to_use = 3
Subject = '01'
model_dir = rf'../../interior-design/LSTM/sub-01_ses-01-02_roi-ALL_fmri-gd_4_fold_balanced/1031'
# output_dir = rf'D:/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-{str(Subject).zfill(2)}/'
output_dir = model_dir

MODEL_PATH = os.path.join(model_dir, f"best_model_LSTM_split_{split_to_use}.h5")
SCALER_PATH = os.path.join(model_dir, f"scaler_split_{split_to_use}.joblib")

# Adjust these paths as needed to point to the correct locations of the files
MAT_PATH = f'D:/IDWB/Video-Emotion/Neural_data/video_order_rt_{split_to_use}.mat'  # Path to video_order_rt_1.mat
CSV_PATH = 'D:/IDWB/Video-Emotion/BrainGuided/video_id_rt.csv'       # Path to video_id_rt.csv

input_data = np.load(rf'D:/IDWB/Video-Emotion/Neural_data/emotion_encoding_results/sub-{str(Subject).zfill(2)}/voxel_select_remain_time_{split_to_use}.npy')

print(f"Shape of input data: {input_data.shape}")

# This is the list where you will store the results
neural_response = []

try:
    # --- Run the function to get the features, model, and preprocessed data ---
    features, original_model, input_data_lstm = extract_lstm_features(
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        input_data=input_data
    )
    
    # --- Append the result as requested ---
    # The 'features' variable is a NumPy array of shape (200, 128)
    # save features to a npy file
    np.save(os.path.join(output_dir, f'voxel_select_lstm_feature_{split_to_use}.npy'), features)
    neural_response.append(features)

    # --- Verify the result ---
    # Note: `neural_response` is a list containing one item: the (200, 128) array.
    # If you process multiple batches and want a single large array, you might use `extend` or `vstack`.
    print(f"\nSuccessfully appended features.")
    print(f"Number of items in neural_response list: {len(neural_response)}")
    print(f"Shape of the appended item: {neural_response[0].shape}")

    # --- Additional code to check model accuracy ---
    print("\nChecking model accuracy...")

    # Load video order from .mat file
    # mat_data = scipy.io.loadmat(MAT_PATH)
    # Assuming the key is 'video_order'; adjust if the key is different
    video_order = np.array(h5py.File(MAT_PATH)['video_order'])
    if video_order is None:
        raise KeyError("Key 'video_order' not found in .mat file. Check the file contents.")
    video_ids = np.squeeze(video_order).astype(int)  # Squeeze to 1D array and ensure int type

    # Load video ID to name mapping from .csv
    df = pd.read_csv(CSV_PATH, sep=',')
    id_to_name = dict(zip(df['Video ID'], df['Video Name and Directory']))

    # Extract labels from video names (directory part before '/')
    labels = []
    for vid in video_ids:
        if vid not in id_to_name:
            raise ValueError(f"Video ID {vid} not found in video_id_rt.csv")
        name = id_to_name[vid]
        label = name.split('/')[0]
        labels.append(label)

    # Use the fixed design_to_number mapping
    design_to_number = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}
    # Map string labels to integers using the fixed mapping
    true_labels = np.array([design_to_number[lab] for lab in labels if lab in design_to_number])
    
    # Check for unknown labels
    unknown_labels = set(labels) - set(design_to_number.keys())
    if unknown_labels:
        print(f"Warning: Unknown labels found: {unknown_labels}. These will be ignored in accuracy calculation.")

    # Ensure the number of labels matches the input samples
    if len(true_labels) != input_data.shape[0]:
        raise ValueError(f"Number of labels ({len(true_labels)}) does not match input samples ({input_data.shape[0]}). Possibly due to unknown labels.")

    # Get model predictions
    predictions = original_model.predict(input_data_lstm)
    predicted_classes = np.argmax(predictions, axis=1)

    # Compute accuracy
    accuracy = np.mean(predicted_classes == true_labels)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Label mapping used: {design_to_number}")

except (FileNotFoundError, ValueError, KeyError) as e:
    print(f"An error occurred: {e}")