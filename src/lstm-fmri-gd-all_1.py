import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
import joblib

# TensorFlow/Keras imports for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import sys

# --- Configuration ---
sub = '01'
sessions = ['01', '02'] # <-- MODIFIED: Use a list of sessions
roi = 'ALL'
print('------', roi, '------')

# For reproducibility
random_state = 42
np.random.seed(random_state)
tf.random.set_seed(random_state)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# List available GPUs
print(f"TensorFlow Version: {tf.__version__}")
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"Found {len(gpu_devices)} GPU(s):")
    for gpu in gpu_devices:
        print(f"- {gpu.name}")
else:
    print("❌ ERROR: No GPU found. Please check your installation.")

# --- Update output directory for combined sessions ---
ses_str = "-".join(sessions)
output_dir = f'../../interior-design/LSTM/sub-{sub}_ses-{ses_str}_roi-{roi}_fmri-gd_4_fold_balanced/1031'
os.makedirs(output_dir, exist_ok=True)

n_splits = 4

# --- Metadata ---
metadata = {
    "subject": f"sub-{sub}",
    "sessions": f"ses-{ses_str}", # <-- MODIFIED
    "roi": roi,
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "n_splits": n_splits,
    "model_type": "LSTM (Keras/TensorFlow)"
}

# --- Data Loading ---
data_by_video = defaultdict(list)
video_design_mapping = {}
video_session_mapping = {} # <-- NEW: To track the session for each video
design_to_number = {'MODN': 0, 'MUJI': 1, 'SCAN': 2, 'WABI': 3}
number_to_design = {v: k for k, v in design_to_number.items()}

# <-- MODIFIED: Loop over both sessions to load all fMRI data
for session in sessions:
    print(f"\n--- Loading data for sub-{sub} ses-{session} ---")
    directory = f"../../../data/fmri-clip-gd/sub-{sub}/ses-{session}"
    
    if not os.path.exists(directory):
        print(f"Warning: Directory not found, skipping: {directory}")
        continue

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            parts = os.path.splitext(filename)[0].split('_')
            file_roi = parts[2]
            video_id = f"{parts[3]}_{parts[4]}_{parts[5]}"
            design_name = parts[-2]
            if file_roi == roi:
                if design_name not in design_to_number:
                    raise ValueError(f"Unknown design '{design_name}' encountered.")
                features = np.load(os.path.join(directory, filename))
                data_by_video[video_id].append(features.T)
                video_design_mapping[video_id] = design_name
                video_session_mapping[video_id] = session # <-- NEW

num_classes = len(design_to_number)
if num_classes < 2:
    raise ValueError(f"LSTM model requires at least 2 classes, but found {num_classes}.")
print(f"Found {num_classes} unique design categories.")

# --- Data Structuring for Cross-Validation ---
all_accuracy = []
all_f1 = []
all_confusion_matrices = []
unique_designs = list(design_to_number.keys())

# <-- NEW: Group videos by design AND session
videos_by_design_and_session = defaultdict(lambda: defaultdict(list))
for video_id, design in video_design_mapping.items():
    session = video_session_mapping[video_id]
    videos_by_design_and_session[design][session].append(video_id)

# <-- NEW: Prerequisite Check
print("\nChecking prerequisites for cross-validation...")
for design in unique_designs:
    if '01' not in videos_by_design_and_session[design] or not videos_by_design_and_session[design]['01']:
        raise ValueError(f"Design '{design}' has no videos from ses-01.")
    if '02' not in videos_by_design_and_session[design] or not videos_by_design_and_session[design]['02']:
        raise ValueError(f"Design '{design}' has no videos from ses-02.")
print("Prerequisite check passed.")

# --- Perform the 4 splits with the session-balanced strategy ---
for i in range(n_splits):
    print(f"\n--- Split {i+1}/{n_splits} ---")
    train_features, train_labels = [], []
    test_features, test_labels = [], []
    test_videos_in_split = defaultdict(list)

    # <-- MODIFIED: This is the session-balanced splitting logic
    for design in unique_designs:
        ses01_videos = videos_by_design_and_session[design]['01']
        ses02_videos = videos_by_design_and_session[design]['02']
        # sort the videos to ensure a consistent split
        ses01_videos.sort()
        ses02_videos.sort()
        test_video_s1 = ses01_videos[i % len(ses01_videos)]
        test_video_s2 = ses02_videos[i % len(ses02_videos)]
        test_videos_in_split[design].extend([test_video_s1, test_video_s2])

    all_test_videos_this_split = {video for video_list in test_videos_in_split.values() for video in video_list}
    print(f"Test videos for this split: {all_test_videos_this_split}")

    for video_id, features_list in data_by_video.items():
        label = design_to_number[video_design_mapping[video_id]]
        if video_id in all_test_videos_this_split:
            test_features.extend(features_list)
            test_labels.extend([label] * len(features_list))
        else:
            train_features.extend(features_list)
            train_labels.extend([label] * len(features_list))

    X_train = np.array(train_features)
    print(X_train.shape)
    y_train = np.array(train_labels)
    X_test = np.array(test_features)
    print(X_test.shape)
    y_test = np.array(test_labels)

    # --- Model Training and Hyperparameter Search ---
    best_f1_fold = -1.0
    best_params_fold = {}
    best_y_pred_fold = None

    # Scale features
    scaler = StandardScaler()
    # flatten the 3D array to 2D (samples, features)
    X_train_flatten = X_train.reshape(X_train.shape[0], -1)
    X_test_flatten = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flatten)
    X_test_scaled = scaler.transform(X_test_flatten)
    print("X_train_scaled.shape:", X_train_scaled.shape)
    print("X_test_scaled.shape:", X_test_scaled.shape)

    # Reshape data for LSTM: (samples, timesteps, features)
    X_train_lstm = X_train_scaled.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test_lstm = X_test_scaled.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    print("X_train_lstm.shape:", X_train_lstm.shape)
    print("X_test_lstm.shape:", X_test_lstm.shape)
    # sys.exit()
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=num_classes)

    print(f"Starting hyperparameter tuning for Split {i+1}...")
    
    model = Sequential([
        Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
        # This layer needs to return sequences for the next LSTM layer
        LSTM(int(X_train_lstm.shape[2]/4), return_sequences=True),
        # This is the final LSTM layer, so it should only return the last output
        LSTM(int(X_train_lstm.shape[2]/16), return_sequences=False), # ✅ CORRECTED
        Dropout(0.3),
        Dense(128, activation='relu', name='feature_extractor'),
        # Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    try:
        history = model.fit(X_train_lstm, y_train_cat,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=0)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"loss_split_{i+1}_LSTM.png"))
        plt.close()
        # plot accuracy
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"accuracy_split_{i+1}_LSTM.png"))
        plt.close()

        y_pred_proba = model.predict(X_test_lstm)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        if f1 > best_f1_fold:
            best_f1_fold = f1
            best_y_pred_fold = y_pred
        # save the model
        model.save(os.path.join(output_dir, f"best_model_LSTM_split_{i+1}.h5"))
        scaler_filename = os.path.join(output_dir, f"scaler_split_{i+1}.joblib")
        joblib.dump(scaler, scaler_filename)

        # # --- Extract features from the training set using the saved model ---
        # print(f"Loading saved model for feature extraction on training set (Split {i+1})...")
        # loaded_model = tf.keras.models.load_model(os.path.join(output_dir, f"best_model_LSTM_split_{i+1}.h5"))
        
        # feature_extractor_model = tf.keras.Model(
        #     inputs=loaded_model.input,
        #     outputs=loaded_model.get_layer('feature_extractor').output
        # )
        
        # print("Extracting features from training set...")
        # train_features = feature_extractor_model.predict(X_train_lstm)
        # print(f"Extracted training features shape: {train_features.shape}")
        
        # Save the extracted training features
        np.save(os.path.join(output_dir, f"train_lstm_features_split_{i+1}.npy"), train_features)
    
    except Exception as e:
        print(f"Error training: {e}")
    finally:
        tf.keras.backend.clear_session()

    # --- Evaluation for the best model of the split ---
    if best_y_pred_fold is None:
        print(f"Warning: No model was successfully trained for Split {i+1}.")
        all_accuracy.append(np.nan)
        all_f1.append(np.nan)
        all_confusion_matrices.append(np.full((num_classes, num_classes), np.nan))
        continue
    
    accuracy = accuracy_score(y_test, best_y_pred_fold)
    cm = confusion_matrix(y_test, best_y_pred_fold, labels=list(range(num_classes)))

    all_accuracy.append(accuracy)
    all_f1.append(best_f1_fold)
    all_confusion_matrices.append(cm)

    print(f"\n--- Results for Split {i+1} ---")
    print(f"Test Set Videos: {dict(test_videos_in_split)}")
    print(f"Best LSTM Parameters: {best_params_fold}")
    print(f"Test Accuracy: {accuracy:.2f}")
    print(f"Test F1 Score: {best_f1_fold:.2f}")
    print("Confusion Matrix:\n", cm)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    heatmap_labels = [number_to_design[l_idx] for l_idx in sorted(number_to_design.keys())]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=heatmap_labels, yticklabels=heatmap_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Split {i+1} - Acc: {accuracy:.2f}, F1: {best_f1_fold:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"heatmap_split_{i+1}_LSTM.png"))
    plt.close()

# --- Final Summary ---
valid_accuracies = [acc for acc in all_accuracy if not np.isnan(acc)]
valid_f1_scores = [f1 for f1 in all_f1 if not np.isnan(f1)]
avg_accuracy = np.mean(valid_accuracies) if valid_accuracies else np.nan
avg_f1 = np.mean(valid_f1_scores) if valid_f1_scores else np.nan

print(f"\n--- Summary Across All Valid Splits (Keras LSTM) ---")
print(f"Average Test Accuracy: {avg_accuracy:.2f}")
print(f"Average Test F1 Score: {avg_f1:.2f}")

# --- Save Logs ---
log_filename = os.path.join(output_dir, "model_metadata_log_LSTM.txt")
with open(log_filename, "w") as log_file:
    for key, value in metadata.items():
        log_file.write(f"{key}: {value}\n")
    log_file.write(f"Average Test Accuracy: {avg_accuracy:.2f}\n")
    log_file.write(f"Average Test F1 Score: {avg_f1:.2f}\n")
    log_file.write("\nResults for Each Split:\n")
    for i in range(len(all_accuracy)):
        log_file.write(f"\nSplit {i+1}:\n")
        log_file.write(f"Test Accuracy: {all_accuracy[i]:.2f}\n")
        log_file.write(f"Test F1 Score: {all_f1[i]:.2f}\n")
        if i < len(all_confusion_matrices) and not np.all(np.isnan(all_confusion_matrices[i])):
            log_file.write(f"Confusion Matrix (Split {i+1}):\n{all_confusion_matrices[i]}\n")

print(f"\nAnalysis complete. Results saved to {output_dir}")