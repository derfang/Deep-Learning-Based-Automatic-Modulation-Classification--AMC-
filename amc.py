import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

# ==========================================
# 1. CONFIGURATION (UPDATED FOR BIG DATA)
# ==========================================
DATA_FILE = 'my_synthetic_dataset.mat'

# Model Filenames
FILE_BASELINE  = 'amc_model_baseline.h5'
FILE_OVERFIT   = 'amc_model_overfit.h5'
FILE_OPTIMIZED = 'amc_model_optimized.h5'
FILE_HIGHSNR   = 'amc_model_highsnr_overfit.h5'

# Hyperparameters (Tuned for ~90,000 samples)
BATCH_STD   = 64   # Standard batch size
BATCH_LARGE = 256  # Larger batch for the optimized model (speed + stability)
BATCH_SMALL = 32   # Small batch to encourage overfitting in the "bad" model
EPOCHS      = 30   # More epochs because we have more data to learn from

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def load_dataset(filename):
    """Loads .mat file and fixes dimensions/labels."""
    print(f"Loading data from {filename}...")
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found! Please run the MATLAB script first.")
        return None, None, None, None

    data = scipy.io.loadmat(filename)
    X = data['X'] 
    Y = data['Y'] 
    Z = data['Z'] # SNR
    
    # Flatten classes list
    classes = data['mods'].ravel() 
    class_names = [c[0] if c.size > 0 else "Unknown" for c in classes]
    
    print(f"Data shape: {X.shape}")
    print(f"Classes: {class_names}")
    return X, Y, Z, class_names

def get_snr_accuracy(model, X, Y, Z, snr_vals):
    """Calculates accuracy for specific SNR levels."""
    acc_list = []
    for snr in snr_vals:
        mask = (Z.flatten() == snr)
        X_sub = X[mask]
        Y_sub = Y[mask]
        if len(X_sub) > 0:
            _, acc = model.evaluate(X_sub, Y_sub, verbose=0)
            acc_list.append(acc)
        else:
            acc_list.append(0)
    return acc_list

# ==========================================
# 3. MODEL ARCHITECTURES
# ==========================================

def build_baseline_cnn(input_shape, num_classes):
    """The 'Goldilocks' model - Simple but effective."""
    model = models.Sequential([
        layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_overfit_cnn(input_shape, num_classes):
    """Too complex, no batch norm - prone to overfitting."""
    model = models.Sequential([
        layers.Conv1D(128, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def build_optimized_cnn(input_shape, num_classes):
    """
    UPDATED V2: Optimized for LARGER DATASET
    - Deeper (3 blocks) to capture more features
    - Batch Norm used heavily
    - Reduced Dropout (0.2) since more data provides natural regularization
    """
    model = models.Sequential([
        # Block 1
        layers.Conv1D(64, 3, padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        
        # Block 2
        layers.Conv1D(64, 3, padding='same'), # Increased filters
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        
        # Block 3 (New)
        layers.Conv1D(32, 3, padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling1D(2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2), # Lower dropout for big data
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. MANAGER LOGIC
# ==========================================

def manage_model(name, filename, builder_func, input_shape, num_classes, X_train, Y_train, X_val, Y_val, batch_size):
    """Handles Loading vs Training logic."""
    print(f"\n--- Managing {name} ---")
    model = None
    should_train = True
    
    if os.path.exists(filename):
        choice = input(f"Found saved '{name}'. Load it? (y/n): ").strip().lower()
        if choice == 'y':
            print(f"Loading {name} from disk...")
            model = load_model(filename)
            should_train = False
    
    if should_train:
        print(f"Training {name}...")
        model = builder_func(input_shape, num_classes)
        model.fit(X_train, Y_train, 
                  epochs=EPOCHS, 
                  batch_size=batch_size, 
                  validation_data=(X_val, Y_val),
                  verbose=1)
        model.save(filename)
        print(f"Saved {filename}")
        
    return model

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    
    # A. Load Data
    X, Y, Z, class_names = load_dataset(DATA_FILE)
    if X is None:
        exit() 
        
    INPUT_SHAPE = (128, 2)
    NUM_CLASSES = len(class_names)

    # B. Create Data Splits
    # 1. Standard Split (80/20)
    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=0.2, random_state=42, shuffle=True, stratify=Y
    )
    
    # 2. "Clean Only" Subset (SNR >= 5dB)
    idx_clean = np.where(Z_train >= 5)[0]
    X_train_clean = X_train[idx_clean]
    Y_train_clean = Y_train[idx_clean]
    
    # 3. "Positive Only" Subset (SNR > 0dB)
    idx_pos = np.where(Z_train > 0)[0]
    X_train_pos = X_train[idx_pos]
    Y_train_pos = Y_train[idx_pos]
    
    print(f"\nData Prepared (Big Data Mode):")
    print(f" - Standard Train Size: {len(X_train)}")
    print(f" - Clean Train Size:    {len(X_train_clean)}")
    print(f" - >0dB Train Size:     {len(X_train_pos)}")

    # C. Run Experiments
    
    # Experiment 1: Baseline (The Standard)
    model_base = manage_model(
        "Baseline Model", FILE_BASELINE, build_baseline_cnn,
        INPUT_SHAPE, NUM_CLASSES, X_train, Y_train, X_test, Y_test, BATCH_STD
    )
    
    # Experiment 2: Overfit Arch + Clean Data
    model_over = manage_model(
        "Overfit (Clean Data)", FILE_OVERFIT, build_overfit_cnn,
        INPUT_SHAPE, NUM_CLASSES, X_train_clean, Y_train_clean, X_test, Y_test, BATCH_SMALL
    )
    
    # Experiment 3: Optimized Arch + All Data
    model_opt = manage_model(
        "Optimized Model (V2)", FILE_OPTIMIZED, build_optimized_cnn,
        INPUT_SHAPE, NUM_CLASSES, X_train, Y_train, X_test, Y_test, BATCH_LARGE
    )
    
    # Experiment 4: Overfit Arch + >0dB Data (The "Fair Weather" Test)
    model_highsnr = manage_model(
        "Overfit (>0dB Data)", FILE_HIGHSNR, build_overfit_cnn, 
        INPUT_SHAPE, NUM_CLASSES, X_train_pos, Y_train_pos, X_test, Y_test, BATCH_SMALL
    )

    # D. Evaluation & Plotting
    print("\nCalculating Performance Metrics...")
    snr_values = np.unique(Z)
    
    acc_base = get_snr_accuracy(model_base, X_test, Y_test, Z_test, snr_values)
    acc_over = get_snr_accuracy(model_over, X_test, Y_test, Z_test, snr_values)
    acc_opt  = get_snr_accuracy(model_opt, X_test, Y_test, Z_test, snr_values)
    acc_high = get_snr_accuracy(model_highsnr, X_test, Y_test, Z_test, snr_values)

    # Plot 1: The Grand Comparison
    plt.figure(figsize=(12, 8))
    
    # Blue: Baseline
    plt.plot(snr_values, acc_base, 'b--', marker='o', label='Baseline (All Data)')
    
    # Red: Overfit (Clean Data Only)
    plt.plot(snr_values, acc_over, 'r:', marker='x', linewidth=2, label='Overfit Arch (Clean Data Only)')
    
    # Green: Optimized V2 (All Data)
    plt.plot(snr_values, acc_opt,  'g-', marker='s', linewidth=2, alpha=0.8, label='Optimized V2 (All Data)')
    
    # Purple: Overfit (>0dB Only)
    plt.plot(snr_values, acc_high, color='purple', marker='D', linestyle='-.', label='Overfit Arch (>0dB Data Only)')
    
    plt.title('Impact of Big Data & Architecture on AMC Accuracy')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim([0, 1.05])
    plt.show()
    
    # Plot 2: Confusion Matrix (For Optimized V2)
    print("Generating Confusion Matrix for Optimized Model...")
    y_pred = model_opt.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Optimized V2)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()