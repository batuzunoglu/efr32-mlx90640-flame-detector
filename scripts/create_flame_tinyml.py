#!/usr/bin/env python3
"""
create_flame_tinyml.py

Loads flame/noflame CSV data, trains a simple Keras model,
converts it to a float32 TFLite format, and saves it as a C array
for microcontroller deployment.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import argparse # For command-line arguments

# --- Configuration ---
FLAME_CSV = "flame.csv"
NOFLAME_CSV = "noflame.csv"
OUTPUT_DIR = "output_model" # Directory to save model files
MODEL_BASE_NAME = "flame_detector" # Base name for output files

# Training Parameters
TEST_SIZE = 0.2
RANDOM_SEED = 42
EPOCHS = 30 # Adjust as needed
BATCH_SIZE = 16

# Model Architecture Parameters
INPUT_SHAPE = (768,) # 24 * 32 pixels flattened
DENSE1_UNITS = 32    # Reduced complexity for TinyML
DENSE2_UNITS = 16    # Reduced complexity for TinyML

# --- Helper Functions ---

def load_data(flame_csv, noflame_csv):
    """Loads and prepares data from CSV files."""
    print(f"Loading data from {flame_csv} and {noflame_csv}...")
    try:
        df_f = pd.read_csv(flame_csv)
        df_nf = pd.read_csv(noflame_csv)
    except FileNotFoundError as e:
        print(f"Error loading CSV: {e}. Make sure files exist.")
        return None, None, None, None

    # Combine data and create labels
    X = np.vstack([df_f.values, df_nf.values]).astype(np.float32)
    y = np.concatenate([
        np.ones(len(df_f), dtype=np.int32),
        np.zeros(len(df_nf), dtype=np.int32),
    ])

    # Normalize raw pixel counts [0...65535] -> [0...1]
    # Important: Use float32 for initial processing
    X /= 65535.0
    print(f"Data range after normalization: min={X.min()}, max={X.max()}") # Verify normalization

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y # Ensure balanced split
    )
    print(f"Data loaded: {len(X_train)} train samples, {len(X_val)} validation samples.")
    return X_train, X_val, y_train, y_val

def build_model(input_shape):
    """Builds a simple Keras sequential model."""
    print("Building Keras model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="input_layer", dtype=tf.float32), # Explicitly float32
        tf.keras.layers.Dense(DENSE1_UNITS, activation="relu", name="dense_1"),
        tf.keras.layers.Dense(DENSE2_UNITS, activation="relu", name="dense_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output_layer"), # Binary classification
    ], name="FlameDetector")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("Model built successfully.")
    model.summary()
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    """Trains the Keras model."""
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=2 # Show progress per epoch
    )
    print("Training complete.")

    print("\nEvaluating on validation set:")
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  Validation Loss: {loss:.4f}")
    print(f"  Validation Accuracy: {acc:.4%}")
    return history

def convert_to_tflite_float32(model, output_tflite_path):
    """Converts the Keras model to a standard float32 TFLite model."""
    print(f"\nConverting model to TFLite (float32)... -> {output_tflite_path}")

    # Set up converter for float32
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # No quantization options needed for standard float32 conversion
    # converter.optimizations = [tf.lite.Optimize.DEFAULT] # Optional: May apply some float optimizations

    # Convert the model
    try:
        tflite_model_float = converter.convert()
    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        return None

    # Save the TFLite model
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model_float)

    print(f"Float32 TFLite model saved successfully ({len(tflite_model_float)} bytes).")
    return tflite_model_float

def convert_to_c_array(tflite_model_bytes, output_cc_path, model_var_name):
    """Converts the TFLite model bytes to a C array in a .cc file."""
    if tflite_model_bytes is None:
        print("Skipping C array conversion due to previous errors.")
        return

    print(f"\nConverting TFLite model to C array... -> {output_cc_path}")
    try:
        # Create the C array string
        hex_array = [f"0x{b:02x}" for b in tflite_model_bytes]
        c_array_str = ", ".join(hex_array)

        # Create the C source file content
        c_file_content = f"""
/* Auto-generated C array from TFLite model */
/* Contains the float32 model data for {model_var_name} */

// Ensure alignment for better performance on some platforms
alignas(16) const unsigned char {model_var_name}[] = {{
  {c_array_str}
}};

const unsigned int {model_var_name}_len = {len(tflite_model_bytes)};
"""
        # Write the C file
        with open(output_cc_path, "w") as f:
            f.write(c_file_content)

        print(f"C array saved successfully.")

        # Suggest creating a header file
        print("\n--------------------------------------------------")
        print(f"ACTION NEEDED: Create a header file (e.g., {model_var_name}.h) with:")
        print(f"""
#ifndef {model_var_name.upper()}_H
#define {model_var_name.upper()}_H

extern const unsigned char {model_var_name}[];
extern const unsigned int {model_var_name}_len;

#endif // {model_var_name.upper()}_H
""")
        print("--------------------------------------------------")

    except Exception as e:
        print(f"Error converting to C array: {e}")


# --- Main Execution ---
def main(args):
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data
    X_train, X_val, y_train, y_val = load_data(args.flame_csv, args.noflame_csv)
    if X_train is None:
        return # Exit if data loading failed

    # 2. Build Model
    model = build_model(input_shape=INPUT_SHAPE)

    # 3. Train Model
    train_model(model, X_train, y_train, X_val, y_val)

    # Define output paths
    base_path = os.path.join(args.output_dir, args.model_base_name)
    output_tflite_path = f"{base_path}_float.tflite" # Changed suffix
    output_cc_path = f"{base_path}_float_model.cc"  # Changed suffix
    model_var_name = f"{args.model_base_name}_float_tflite" # Changed suffix for C variable

    # 4. Convert to TFLite (Float32)
    tflite_model_bytes = convert_to_tflite_float32(model, output_tflite_path)

    # 5. Convert to C Array
    convert_to_c_array(tflite_model_bytes, output_cc_path, model_var_name)

    print("\nTinyML model creation pipeline finished!")

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train, convert (float32), and save a flame detection model for TinyML.")
    parser.add_argument("--flame_csv", type=str, default=FLAME_CSV, help="Path to the flame data CSV file.")
    parser.add_argument("--noflame_csv", type=str, default=NOFLAME_CSV, help="Path to the no-flame data CSV file.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save output model files.")
    parser.add_argument("--model_base_name", type=str, default=MODEL_BASE_NAME, help="Base name for output files (e.g., 'flame_detector').")
    # Add more arguments if needed (e.g., epochs, batch_size)

    args = parser.parse_args()
    main(args)
