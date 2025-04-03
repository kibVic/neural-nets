import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.xception import Xception, preprocess_input
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# 1. Function to load and prepare datasets with augmentation
def prepare_data(input_size=150, batch_size=32):
    """
    Loads the training and validation datasets using ImageDataGenerator with augmentation.
    """
    train_gen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30, 
        width_shift_range=0.2, 
        height_shift_range=0.2, 
        shear_range=0.2, 
        zoom_range=0.2, 
        horizontal_flip=True
    )
    train_ds = train_gen.flow_from_directory(
        './train_dataset',
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
    val_ds = val_gen.flow_from_directory(
        './validation_dataset',
        target_size=(input_size, input_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_ds, val_ds

# 2. Function to create the model with fine-tuning
def create_model(input_size=150, learning_rate=1e-5, size_inner=100, droprate=0.3):
    """
    Builds the Xception-based model with fine-tuning enabled.
    """
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
    base_model.trainable = True  # Enable fine-tuning

    # Freeze first 100 layers to retain pre-trained features
    for layer in base_model.layers[:100]:
        layer.trainable = False  

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    outputs = keras.layers.Dense(5, activation='softmax')(drop)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

# 3. Function to compute class weights for imbalanced data
def compute_weights(train_ds):
    labels = train_ds.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return dict(enumerate(class_weights))

# 4. Function to train the model with early stopping
def train_model(model, train_ds, val_ds, epochs=50, checkpoint_filepath='model_checkpoint.h5'):
    """
    Trains the model with early stopping and saves the best weights.
    """
    class_weight_dict = compute_weights(train_ds)
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath, save_best_only=True, monitor='val_accuracy', mode='max'
    )
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        train_ds, 
        epochs=epochs, 
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping],
        class_weight=class_weight_dict
    )
    return history

# 5. Main Execution Flow
def main():
    train_ds, val_ds = prepare_data(input_size=299, batch_size=32)
    model = create_model(input_size=299, learning_rate=1e-5, size_inner=100, droprate=0.3)
    history = train_model(model, train_ds, val_ds, epochs=50)
    model.save('final_model.h5')

if __name__ == "__main__":
    main()
