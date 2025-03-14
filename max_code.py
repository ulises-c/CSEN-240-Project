# %%
import time
import numpy as np
import pandas as pd
import os
import tensorflow as tf 

import seaborn as sns
import matplotlib.pyplot as plt

import cv2

start_time = time.time()

def import_data(): 
    """
    Assumes that we have a directory of the images in the root of the project. 
    """
    data_path = os.curdir + "/Knee_Osteoarthritis_Classification"
    categories = ["Normal","Osteopenia", "Osteoporosis"]

    image_paths = []
    labels = []

    for category in categories:
        category_path = os.path.join(data_path, "train", category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            image_paths.append(image_path)
            labels.append(category)

    return image_paths, labels, categories

image_paths, labels, categories = import_data()
df = pd.DataFrame({"image_path": image_paths, "label": labels})

print(df.shape)
print(df.duplicated().sum())
print(df.isnull().sum())
print(df.info())
print("Unique labels: {}".format(df['label'].unique()))
print("Label counts: {}".format(df['label'].value_counts()))

# %%
# =============================================================================
def show_tumor_distribution_chart(df):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x="label", palette="viridis")
    plt.title("Distribution of Tumor Types", fontsize=14, fontweight='bold')
    plt.xlabel("Tumor Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, color='black',
            xytext=(0, 5), textcoords='offset points')

    plt.show(block=False)
    plt.pause(5)
    plt.close()

show_tumor_distribution_chart(df)

# %%
# =============================================================================
def show_tumor_distribution_pie(df):    
    label_counts = df["label"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(label_counts))
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
    startangle=140, colors=colors, textprops={'fontsize': 12, 'weight':'bold'},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    ax.set_title("Distribution of Tumor Types - Pie Chart", fontsize=14,
    fontweight='bold')

    plt.show(block=False)
    plt.pause(5)
    plt.close()

show_tumor_distribution_pie(df)

# %%
# =============================================================================
def show_example_images(df, categories): 
    num_images = 5
    plt.figure(figsize=(15, 12))
    for i, category in enumerate(categories):
        category_images = df[df['label'] == category]['image_path'].iloc[:num_images]
        for j, img_path in enumerate(category_images):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(category)

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    plt.close()

show_example_images(df, categories)

# %%
# =============================================================================
from sklearn.preprocessing import LabelEncoder

def add_encoded_column(df):
    label_encoder = LabelEncoder()
    df['category_encoded'] = label_encoder.fit_transform(df['label'])
    df = df[['image_path', 'category_encoded']]
    return df 

df = add_encoded_column(df)

# %%
# =============================================================================
from imblearn.over_sampling import RandomOverSampler

def over_sample_data(df): 
    """
    Oversample the minority classes to balance the class distribution.
    """
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(df[['image_path']],
    df['category_encoded'])
    df_resampled = pd.DataFrame(X_resampled, columns=['image_path'])
    df_resampled['category_encoded'] = y_resampled
    return df_resampled

df_resampled = over_sample_data(df)

print("\nClass distribution after oversampling:")
print(df_resampled['category_encoded'].value_counts())
print(df_resampled)

# %%
# =============================================================================

def convert_encoded_category_to_string(df): 
    df['category_encoded'] = df['category_encoded'].astype(str)
    return df

df_resampled = convert_encoded_category_to_string(df_resampled)

# %%
# =============================================================================
import time
import shutil
import pathlib
import itertools
from PIL import Image
import cv2
import seaborn as sns
sns.set_style('darkgrid')

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras import Sequential
from keras.api.optimizers import Adam
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras.api.preprocessing import image
from keras import regularizers

import warnings
warnings.filterwarnings("ignore")
print ('check')

# %%
# =============================================================================
train_df_new, temp_df_new = train_test_split(
    df_resampled,
    train_size=0.8,
    shuffle=True,
    random_state=42,
    stratify=df_resampled['category_encoded']
)
print(train_df_new.shape)
print(temp_df_new.shape)

valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=0.5,
    shuffle=True,
    random_state=42,
    stratify=temp_df_new['category_encoded']
)
print(valid_df_new.shape)
print(test_df_new.shape)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
# =============================================================================

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def create_classification_model(input_shape=(224, 224, 3), num_classes=3, learning_rate=5e-4):
    inputs = layers.Input(shape=input_shape, name="Input_Layer")
    
    # EfficientNetV2-S (no filters) like the sobel we had before...
    base_model = EfficientNetV2S(weights="imagenet", include_top=False, input_tensor=inputs, pooling=None)
    base_model.trainable = False
    
    x = base_model.output  # Shape: (None, 7, 7, 1280)
    x = layers.GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)  # Shape: (None, 1280)
    
    # Enhanced head with three dense layers
    x = layers.Dense(1024, activation="relu", name="FC_1024", 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(name="Head_BN1")(x)
    x = layers.Dropout(0.5, name="Dropout_1")(x)  # Increased to 0.5
    
    x = layers.Dense(512, activation="relu", name="FC_512", 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(name="Head_BN2")(x)
    x = layers.Dropout(0.5, name="Dropout_2")(x)
    
    x = layers.Dense(256, activation="relu", name="FC_256", 
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(name="Head_BN3")(x)
    x = layers.Dropout(0.5, name="Dropout_3")(x)
    
    outputs = layers.Dense(num_classes, activation="softmax", name="Output_Layer")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="EfficientNetV2S_Osteoarthritis_Best")
    model.compile(optimizer=Adam(learning_rate=learning_rate), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    return model, base_model

def train_model(model, base_model, train_generator, valid_generator, epochs=300, batch_size=16):
    frozen_epochs = 150  # 150 frozen epochs
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, min_delta=0.002, verbose=1, mode="max", restore_best_weights=True)
    history = model.fit(train_generator, validation_data=valid_generator, epochs=frozen_epochs, 
                        steps_per_epoch=len(train_generator), validation_steps=len(valid_generator), 
                        callbacks=[reduce_lr, early_stopping])
    
    # Selective fine-tuning: unfreeze top ~30% of layers
    base_model.trainable = True
    total_layers = len(base_model.layers)
    fine_tune_from = int(total_layers * 0.7)  # ~30% of layers (e.g., ~70 out of ~237)
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=2e-5), 
                  loss="sparse_categorical_crossentropy", 
                  metrics=["accuracy"])
    
    history_fine = model.fit(train_generator, validation_data=valid_generator, epochs=epochs, 
                             initial_epoch=frozen_epochs, steps_per_epoch=len(train_generator), 
                             validation_steps=len(valid_generator), 
                             callbacks=[reduce_lr, early_stopping])
    return model, history, history_fine

# Monte Carlo Dropout inference function
def mc_dropout_predict(model, test_generator, num_samples=10):
    predictions = []
    model.trainable = False  # Ensure dropout is active during inference
    for _ in range(num_samples):
        preds = model.predict(test_generator, steps=len(test_generator))
        predictions.append(preds)
    predictions = np.stack(predictions, axis=0)
    mean_preds = np.mean(predictions, axis=0)
    return mean_preds

# %%
# =============================================================================

tr_gen = ImageDataGenerator(
    rotation_range=30,  # Rotation up to 30 degrees
    preprocessing_function=lambda x: x + tf.random.normal(tf.shape(x), mean=0, stddev=0.05)  # Noise
)
ts_gen = ImageDataGenerator()  # No augmentation for validation/test
batch_size = 32
img_size = (224, 224)

train_gen_new = tr_gen.flow_from_dataframe(train_df_new, x_col='image_path', y_col='category_encoded', 
                                          target_size=img_size, class_mode='sparse', color_mode='rgb', 
                                          shuffle=True, batch_size=batch_size)
valid_gen_new = ts_gen.flow_from_dataframe(valid_df_new, x_col='image_path', y_col='category_encoded', 
                                          target_size=img_size, class_mode='sparse', color_mode='rgb', 
                                          shuffle=True, batch_size=batch_size)
test_gen_new = ts_gen.flow_from_dataframe(test_df_new, x_col='image_path', y_col='category_encoded', 
                                         target_size=img_size, class_mode='sparse', color_mode='rgb', 
                                         shuffle=False, batch_size=batch_size)

# %%
# =============================================================================

cnn_model, base_model = create_classification_model(input_shape=(224, 224, 3), num_classes=3)
cnn_model, history, history_fine = train_model(cnn_model, base_model, train_gen_new, valid_gen_new, epochs=250, batch_size=batch_size)

test_loss, test_accuracy = cnn_model.evaluate(test_gen_new, steps=len(test_gen_new))
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

cnn_model.save('/home/max/Documents/csen-240-project/knee_osteo_model_v11.keras')
print("Model saved as knee_osteo_model_v11.keras")

# %%
# =============================================================================

y_pred = cnn_model.predict(valid_gen_new)
y_true = valid_gen_new.labels

def ppo_loss(y_true, y_pred):
    epsilon = 0.2
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    selected_probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
    old_selected_probs = tf.reduce_sum(tf.stop_gradient(y_pred) * y_true_one_hot, axis=-1)
    ratio = selected_probs / (old_selected_probs + 1e-10)
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio))
    return loss

ppo_loss_value = ppo_loss(y_true, y_pred)
print("\nPPO Loss on Validation Data:", ppo_loss_value.numpy())

# %%
# =============================================================================

# Plotting
train_acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
train_loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']
epochs_range = range(1, len(train_acc) + 1)

# Plot 1: Accuracy
plt.figure(figsize=(6, 5))  # Single plot size
plt.plot(epochs_range, train_acc, label='Train')
plt.plot(epochs_range, val_acc, label='Validation')
plt.axvline(x=80, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('/home/max/Documents/csen-240-project/accuracy_plot-11.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Loss
plt.figure(figsize=(6, 5))
plt.plot(epochs_range, train_loss, label='Train')
plt.plot(epochs_range, val_loss, label='Validation')
plt.axvline(x=80, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('/home/max/Documents/csen-240-project/loss_plot-11.png', dpi=300, bbox_inches='tight')
plt.close()

# Optional: Display combined plot (if you still want to see it)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_acc, label='Train')
plt.plot(epochs_range, val_acc, label='Validation')
plt.axvline(x=80, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_loss, label='Train')
plt.plot(epochs_range, val_loss, label='Validation')
plt.axvline(x=80, color='gray', linestyle='--', label='Fine-Tuning Start')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()

# %%
# =============================================================================

test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(test_labels, predicted_classes,
target_names=list(test_gen_new.class_indices.keys()))
print(report)

conf_matrix = confusion_matrix(test_labels, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=list(test_gen_new.class_indices.keys()),
yticklabels=list(test_gen_new.class_indices.keys()))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show(block=False)
plt.pause(5)
plt.close()

end_time = time.time()
print("Total Execution Time: {:.2f} seconds".format(end_time - start_time))