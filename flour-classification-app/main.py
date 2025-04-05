import numpy as np
import tensorflow as tf
import gdown
import zipfile
import os
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input
from urllib.parse import urlparse



# 2. Function to load and prepare datasets
def prepare_data(input_size=150, batch_size=32):
    """
    Loads the training, validation, and test datasets using ImageDataGenerator.
    """
    try:
        # Assuming datasets are already downloaded and extracted in respective folders
        train_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        train_ds = train_gen.flow_from_directory(
            './train_dataset',
            target_size=(input_size, input_size),
            batch_size=batch_size
        )

        val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_ds = val_gen.flow_from_directory(
            './validation_dataset',
            target_size=(input_size, input_size),
            batch_size=batch_size,
            shuffle=False
        )

        test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        test_ds = test_gen.flow_from_directory(
            './test_dataset',
            target_size=(input_size, input_size),
            batch_size=batch_size,
            shuffle=False
        )
        
        return train_ds, val_ds, test_ds
    except Exception as e:
        print(f"Error while loading data: {e}")
        return None, None, None

# 3. Function to create the model
def create_model(input_size=150, learning_rate=0.001, size_inner=100, droprate=0.5):
    """
    Builds the Xception-based model with customizable parameters for inner layer size and dropout rate.
    """
    try:
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
        base_model.trainable = False

        inputs = keras.Input(shape=(input_size, input_size, 3))
        base = base_model(inputs, training=False)
        vectors = keras.layers.GlobalAveragePooling2D()(base)
        inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
        drop = keras.layers.Dropout(droprate)(inner)

        outputs = keras.layers.Dense(5, activation='softmax')(drop)  # Assuming 5 classes

        model = keras.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        loss = keras.losses.CategoricalCrossentropy()

        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        return model
    except Exception as e:
        print(f"Error while creating the model: {e}")
        return None

# 4. Function to plot training history
def plot_history(history, epochs=10):
    """
    Plots the training and validation accuracy over epochs.
    """
    try:
        if 'accuracy' in history.history and 'val_accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_accuracy'], label='val')
            plt.xticks(np.arange(epochs))
            plt.legend()
            plt.show()
        else:
            print("No accuracy data found in training history.")
    except Exception as e:
        print(f"Error while plotting history: {e}")

# 5. Function to evaluate the model
def evaluate_model(model, test_data):
    """
    Evaluates the model on test data and returns the evaluation result.
    """
    try:
        return model.evaluate(test_data)
    except Exception as e:
        print(f"Error while evaluating the model: {e}")
        return None

# 6. Function to make predictions on a single image
def predict_image(model, image_path, input_size=299):
    """
    Makes a prediction on a single image.
    """
    try:
        img = load_img(image_path, target_size=(input_size, input_size))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        pred = model.predict(img_array)

        classes = [
            "contaminated_maize_flour",
            "maize_flour_grade_one",
            "maize_flour_grade_two",
            "sorghum_flour",
            "yellow_flour"
        ]

        return dict(zip(classes, pred[0]))
    except Exception as e:
        print(f"Error while predicting image: {e}")
        return None

# 7. Function to set up and train the model
def train_model(model, train_ds, val_ds, epochs=10, checkpoint_filepath='model_checkpoint.h5'):
    """
    Trains the model and saves the best weights using a checkpoint.
    """
    try:
        checkpoint = keras.callbacks.ModelCheckpoint(
            checkpoint_filepath, 
            save_best_only=True, 
            monitor='val_accuracy', 
            mode='max'
        )

        history = model.fit(
            train_ds, 
            epochs=epochs, 
            validation_data=val_ds,
            callbacks=[checkpoint]
        )

        return history
    except Exception as e:
        print(f"Error while training the model: {e}")
        return None

# 8. Main Execution Flow
def main():
    try:
       
        # Step 2: Load data
        train_ds, val_ds, test_ds = prepare_data(input_size=299, batch_size=32)
        if train_ds is None or val_ds is None or test_ds is None:
            raise ValueError("Error in loading datasets. Aborting training.")

        # Step 3: Create model
        model = create_model(input_size=299, learning_rate=0.001, size_inner=100, droprate=0.2)
        if model is None:
            raise ValueError("Error in creating model. Aborting training.")

        # Step 4: Train the model
        history = train_model(model, train_ds, val_ds, epochs=10)
        if history is None:
            raise ValueError("Error in training the model. Aborting further steps.")

        # Step 5: Plot training history
        plot_history(history, epochs=10)

        # Step 6: Save the final model
        model.save('final_model.h5')

        # Step 7: Evaluate model on test data
        test_score = evaluate_model(model, test_ds)
        if test_score is None:
            raise ValueError("Error in model evaluation.")

        print("Test Evaluation:", test_score)

       
        # Step 8: Predict a single image
        image_path = "test_dataset/contaminated_maize_flour/Contaminated _1742985210041.jpg"

        predictions = predict_image(model, image_path, input_size=299)
        if predictions is None:
            raise ValueError("Error in making predictions.")
        
        print("Predictions:", predictions)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
