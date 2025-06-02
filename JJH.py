import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import shutil # For splitting dataset

# --- Configuration ---
# TASK: User needs to set this path to the extracted 'caltech-101' directory
# It should contain '101_ObjectCategories' and 'BACKGROUND_Google'
BASE_DATASET_PATH = './caltech-101' # Example path

# Processed dataset directory
PROCESSED_DATA_PATH = './caltech_101_processed'
TRAIN_DIR = os.path.join(PROCESSED_DATA_PATH, 'train')
VALIDATION_DIR = os.path.join(PROCESSED_DATA_PATH, 'validation')
TEST_DIR = os.path.join(PROCESSED_DATA_PATH, 'test')

# For image processing
IMG_WIDTH = 128  # Reduced from 224x224 for faster training on a custom model
IMG_HEIGHT = 128
CHANNELS = 3
IMAGE_SIZE = (IMG_HEIGHT, IMG_WIDTH)
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Training parameters
BATCH_SIZE = 32 # Describe why you chose this [cite: 21]
EPOCHS = 50    # Describe why you chose this [cite: 21] (Early stopping will be used)
NUM_CLASSES = 102 # 101 object categories + 1 background category

# --- 1. Dataset Preparation ---
def create_processed_dirs():
    # This function creates train/validation/test directories.
    # It's important for organizing data for ImageDataGenerator.
    if os.path.exists(PROCESSED_DATA_PATH):
        print(f"Processed data path {PROCESSED_DATA_PATH} already exists. Skipping creation or cleaning up.")
        # You might want to add logic here to either use existing or delete and recreate
        # For safety in this example, if it exists, we assume it's correctly populated.
        # return
        shutil.rmtree(PROCESSED_DATA_PATH) # Clean up for a fresh run
        print(f"Cleaned up existing processed data path: {PROCESSED_DATA_PATH}")

    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    all_categories = []
    # Add object categories
    obj_cat_path = os.path.join(BASE_DATASET_PATH, '101_ObjectCategories')
    if not os.path.exists(obj_cat_path):
        print(f"ERROR: Dataset not found at {obj_cat_path}. Please download and place it correctly.")
        return False
        
    for category in os.listdir(obj_cat_path):
        if os.path.isdir(os.path.join(obj_cat_path, category)):
            all_categories.append(os.path.join(obj_cat_path, category))

    # Add background category
    bg_cat_path = os.path.join(BASE_DATASET_PATH, 'BACKGROUND_Google')
    if os.path.exists(bg_cat_path) and os.path.isdir(bg_cat_path):
         all_categories.append(bg_cat_path)
    else:
        print(f"Warning: BACKGROUND_Google category not found at {bg_cat_path}")


    if not all_categories:
        print("No image categories found. Please check BASE_DATASET_PATH.")
        return False

    for category_path in all_categories:
        category_name = os.path.basename(category_path)

        # Create subdirectories in train, validation, test
        os.makedirs(os.path.join(TRAIN_DIR, category_name), exist_ok=True)
        os.makedirs(os.path.join(VALIDATION_DIR, category_name), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, category_name), exist_ok=True)

        images = [img for img in os.listdir(category_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        # Splitting: 80% train, 10% validation, 10% test [cite: 23]
        train_split = int(0.8 * len(images))
        val_split = int(0.9 * len(images)) # 0.8 to 0.9 is 10% for validation

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for img in train_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(TRAIN_DIR, category_name, img))
        for img in val_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(VALIDATION_DIR, category_name, img))
        for img in test_images:
            shutil.copy(os.path.join(category_path, img), os.path.join(TEST_DIR, category_name, img))
    
    print("Dataset successfully split into train, validation, and test sets.")
    return True


def get_data_generators():
    # Data augmentation for training to improve robustness
    # Normalization is crucial
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Validation and test generators should only rescale
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, # Can be 1 or BATCH_SIZE for evaluation
        class_mode='categorical',
        shuffle=False
    )
    
    # Verify number of classes picked up by generator
    # global NUM_CLASSES # Ensure NUM_CLASSES is updated based on what generator found.
    # NUM_CLASSES = train_generator.num_classes
    # print(f"Number of classes detected by ImageDataGenerator: {NUM_CLASSES}")
    # Actually, NUM_CLASSES should be set based on the folders prepared.
    # The check above is a good sanity check.
    
    # Example of image transformation for the report [cite: 21]
    # x_batch, y_batch = next(train_generator)
    # first_image_transformed = x_batch[0]
    # plt.imshow(first_image_transformed)
    # plt.title("Example of a Transformed Training Image")
    # plt.show() # You would save this plot for the report.

    return train_generator, validation_generator, test_generator

# --- 2. Model Definition ---
# TASK: Design your CNN. Explain each layer choice in your report. [cite: 18, 20]
# This is a basic example. You might want more layers, different filter sizes, dropout, batch normalization etc.
def create_cnn_model(input_shape, num_classes):
    # This is where the "forward propagation" path is defined. [cite: 21]
    # TensorFlow will automatically handle "backward propagation" during training.
    model = Sequential([
        # Convolutional Block 1
        # Explain filter size, kernel size, strides, padding, activation. [cite: 19, 20]
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        # BatchNormalization(), # Optional: Can help stabilize and speed up training
        MaxPooling2D((2, 2)),
        # Dropout(0.25), # Optional: Regularization

        # Convolutional Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        # BatchNormalization(),
        MaxPooling2D((2, 2)),
        # Dropout(0.25),

        # Convolutional Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        # BatchNormalization(),
        MaxPooling2D((2, 2)),
        # Dropout(0.25),

        # Flattening the 3D output to 1D for Dense layers
        Flatten(),

        # Fully Connected Layer
        Dense(512, activation='relu'),
        # BatchNormalization(),
        # Dropout(0.5), # Dropout before the final classification layer

        # Output Layer
        # Softmax for multi-class classification
        Dense(num_classes, activation='softmax')
    ])

    # Optimizer choice: Adam is a common default. Explain why you chose it. [cite: 21]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Experiment with learning rates [cite: 22]

    # Loss function: Categorical crossentropy for multi-class. Explain. [cite: 21]
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.F1Score(average='macro', name='f1_score')]) # F1 score for TF 2.6+
                  # For older TF, you might need to calculate F1 manually using sklearn after prediction

    model.summary() # Prints the model structure, useful for the report's table [cite: 19]
    return model

# --- 3. Training ---
def train_model(model, train_generator, validation_generator, epochs):
    # Callbacks for better training control
    # EarlyStopping: Stops training if no improvement. Explain use. [cite: 21]
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # ReduceLROnPlateau: Reduces learning rate if no improvement. Explain use. [cite: 21]
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )
    return history

# --- 4. Evaluation ---
def evaluate_model(model, test_generator):
    results = model.evaluate(test_generator, verbose=1)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")
    if len(results) > 2: # If F1 score metric was compiled in model
        print(f"Test F1-score (macro): {results[2]}") # Index might change based on metrics order

    # For confusion matrix and detailed F1 score using sklearn
    y_pred_probabilities = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_probabilities, axis=1)
    y_true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Calculate F1 score using scikit-learn (often more reliable for detailed reporting)
    f1 = f1_score(y_true_classes, y_pred_classes, average='macro')
    print(f"Test F1-score (macro sklearn): {f1}")
    
    accuracy_sk = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Test Accuracy (sklearn): {accuracy_sk}")


    # Confusion Matrix [cite: 22]
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(15, 12)) # Adjust size depending on NUM_CLASSES
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues') # Annot=True can be slow for many classes
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Classes')
    plt.xlabel('Predicted Classes')
    # plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=90)
    # plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png') # Save for the report [cite: 8]
    print("Confusion matrix saved as confusion_matrix.png")
    # plt.show() # Display plot

    return accuracy_sk, f1 # Return sklearn calculated metrics

# --- 5. Plotting Training History ---
def plot_history(history):
    # Plot for accuracy and loss curves [cite: 22]
    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history.png') # Save for the report [cite: 8]
    print("Training history plot saved as training_history.png")
    # plt.show() # Display plot


# --- Main Execution ---
if __name__ == '__main__':
    # Prepare dataset structure (run once initially)
    if not create_processed_dirs():
        print("Failed to prepare dataset directories. Exiting.")
        exit()

    # Get data generators
    train_generator, validation_generator, test_generator = get_data_generators()
    
    # Make sure NUM_CLASSES is correct (it's hardcoded above, but generator can verify)
    if train_generator.num_classes != NUM_CLASSES:
        print(f"Warning: Number of classes from generator ({train_generator.num_classes}) "
              f"does not match configured NUM_CLASSES ({NUM_CLASSES}).")
        # Potentially update NUM_CLASSES = train_generator.num_classes here if dynamic adjustment is desired.

    # For averaging results over 10 trials [cite: 23]
    num_trials = 1 # Set to 10 for the final assignment submission
    all_accuracies = []
    all_f1_scores = []

    for i in range(num_trials):
        print(f"\n--- Starting Trial {i+1}/{num_trials} ---")
        
        # Create and compile the model (re-create for each trial for independence)
        # This ensures weights are re-initialized.
        model = create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
        
        history = train_model(model, train_generator, validation_generator, EPOCHS)
        
        acc, f1 = evaluate_model(model, test_generator)
        all_accuracies.append(acc)
        all_f1_scores.append(f1)
        
        if i == 0: # Plot history only for the first trial or save all if needed
            plot_history(history)

    print("\n--- Overall Results ---")
    if num_trials > 1:
        print(f"Average Accuracy over {num_trials} trials: {np.mean(all_accuracies):.4f} (+/- {np.std(all_accuracies):.4f})")
        print(f"Average F1-score over {num_trials} trials: {np.mean(all_f1_scores):.4f} (+/- {np.std(all_f1_scores):.4f})")
    else:
        print(f"Accuracy (1 trial): {all_accuracies[0]:.4f}")
        print(f"F1-score (1 trial): {all_f1_scores[0]:.4f}")

    # Remember to document hyperparameter choices and experiments in the report [cite: 21, 22]
    print("\nEnd of script. Don't forget to complete your report!")