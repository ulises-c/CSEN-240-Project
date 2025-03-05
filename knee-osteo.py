import numpy as np
import pandas as pd
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import time
import shutil
import pathlib
import itertools
import tensorflow as tf
import warnings
import platform
import logging
import plot_utils # Custom module that has plotting functions
import json # To load hyperparameters from a JSON file, also used to save model history
import random # To set random seed for reproducibility

from datetime import datetime
from imblearn.over_sampling import RandomOverSampler
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.applications import Xception
# from tensorflow.keras.mixed_precision import experimental as mixed_precision # may be useful for training on Apple Silicon or Nvidia GPUs with less VRAM
from tensorflow.keras import mixed_precision # may be useful for training on Apple Silicon or Nvidia GPUs with less VRAM
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    Input,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    GaussianNoise,
    MultiHeadAttention,
    Reshape,
)

# TODO: Add support for mixed precision training
# TODO: Add support for sklearn.utils shuffle
# TODO: Add support for sklearn.utils resample
# TODO: Add support for sklearn.metrics accuracy_score
# TODO: Add support for logging
# TODO: Checkout Tensorflow Logging
# TODO: Checkout Tensorboard

start_time = time.perf_counter()

# Load hyperparameters from a JSON file
with open("hyperparameters.json", "r") as f:
    hyperparameters = json.load(f)

ENABLE_PLOTS = hyperparameters["enable_plots"]
SAVE_PLOTS = hyperparameters["save_plots"]
ENABLE_TF_DETERMINISM = hyperparameters["enable_tf_determinism"]
SAVE_BEST_MODEL = hyperparameters["save_best_model"]
CONVERT_TO_COREML = hyperparameters["convert_to_coreml"]
RANDOM_SEED = hyperparameters["random_seed"]
BATCH_SIZE = hyperparameters["batch_size"] # reducing may help with VRAM issues, but may also reduce accuracy, original was 16
IMG_SIZE = tuple(hyperparameters["img_size"])
CHANNELS = hyperparameters["channels"]
LEARNING_RATE = hyperparameters["learning_rate"]
NUM_CLASSES = hyperparameters["num_classes"]
EPOCHS = hyperparameters["epochs"]
EARLY_STOPPING_PATIENCE = hyperparameters["early_stopping_patience"]
DROPOUT_RATE = hyperparameters["dropout_rate"]
GAUSSIAN_NOISE_STDDEV = hyperparameters["gaussian_noise_stddev"]
NUM_ATTENTION_HEADS = hyperparameters["num_attention_heads"]
ATTENTION_KEY_DIM = hyperparameters["attention_key_dim"] # Based on the number of channels in the input (3 in this case for RGB)
TRAIN_SPLIT = hyperparameters["train_split"]
VALID_TEST_SPLIT = hyperparameters["valid_test_split"]
ENABLE_MIXED_PRECISION = hyperparameters["enable_mixed_precision"]
USE_EARLY_STOPPING = hyperparameters["use_early_stopping"]
OPTIMIZER = hyperparameters["optimizer"]
LOSS_FUNCTION = hyperparameters["loss_function"]
METRICS = hyperparameters["metrics"]
AUGMENTATION = hyperparameters["augmentation"]


# Set random seed for reproducibility, requires ENABLE_TF_DETERMINISM to be set to True
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.keras.utils.set_random_seed(RANDOM_SEED)
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

# Setup  Tensorflow determinism
if ENABLE_TF_DETERMINISM:
    tf.config.experimental.enable_op_determinism()

# Set up mixed precision training if enabled
if ENABLE_MIXED_PRECISION:
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

# Check if logs directory exists, create if not
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Generate log file name with ISO format date-time and system platform
current_time = datetime.now().isoformat(timespec='seconds')
system_platform = platform.system()
IDENTIFIER = f"{current_time}_{system_platform}"
log_file_name = f"{log_dir}/knee_osteo_{IDENTIFIER}.log"

# Configure logging to write to the generated log file
logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

### Log important information
gpus = tf.config.list_physical_devices("GPU")
logger.info("--- START ---")
logger.info(f"Start Time: {current_time}")
logger.info(f"System Platform: {system_platform}")
logger.info(f"GPU Available: {gpus}")
logger.info(f"--- Hyperparameters Start ---")
for key, value in hyperparameters.items():
    logger.info(f"{key}: {value}")
logger.info(f"--- Hyperparameters End ---")

# Enable GPU acceleration if available
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if system_platform == "Linux":
            logger.info("Using CUDA for acceleration on Nvidia GPU")
        elif system_platform == "Darwin": # macOS
            logger.info("Using Metal backend for acceleration on Apple Silicon")

    except RuntimeError as e:
        logger.error(e)
else:
    logger.info("No GPU found. Running on CPU.")

# data_path = "images/Knee_Osteoarthritis_Classification_Original" # Causing issues currently
# data_path = "images/Knee_Osteoarthritis_Classification"  # Extracted zip file
data_path = "images/Knee_Osteoarthritis_Classification_Camino"  # Extracted zip file
categories = ["Normal", "Osteopenia", "Osteoporosis"]
image_paths = []
labels = []

for category in categories:
    category_path = os.path.join(data_path, "train", category)
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        image_paths.append(image_path)
        labels.append(category)


df = pd.DataFrame({"image_path": image_paths, "label": labels})
# print(df.shape)
# print(df.duplicated().sum())
# print(df.isnull().sum())
# print(df.info())
# logger.info("Unique labels: {}".format(df["label"].unique()))
# logger.info("Label counts: {}".format(df["label"].value_counts()))


sns.set_style("whitegrid")

if ENABLE_PLOTS:
    plot_utils.plot_label_distribution(df, identifier=IDENTIFIER, save=SAVE_PLOTS)
    plot_utils.plot_sample_images(df, categories, num_images=5, identifier=IDENTIFIER, save=SAVE_PLOTS)

label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["label"])
df = df[["image_path", "category_encoded"]]


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[["image_path"]], df["category_encoded"])
df_resampled = pd.DataFrame(X_resampled, columns=["image_path"])
df_resampled["category_encoded"] = y_resampled
# logger.info("\nClass distribution after oversampling:")
# print(df_resampled["category_encoded"].value_counts())
# print(df_resampled)


df_resampled["category_encoded"] = df_resampled["category_encoded"].astype(str)


sns.set_style("darkgrid")

warnings.filterwarnings("ignore")

train_df_new, temp_df_new = train_test_split(
    df_resampled,
    train_size=0.8,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=df_resampled["category_encoded"],
)
# print(train_df_new.shape)
# print(temp_df_new.shape)

valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=0.5,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=temp_df_new["category_encoded"],
)
# print(valid_df_new.shape)
# print(test_df_new.shape)

# Train data generator (with augmentation)
### NOTE: This may not be allowed within project constraints ###
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=AUGMENTATION["rotation_range"],
    width_shift_range=AUGMENTATION["width_shift_range"],
    height_shift_range=AUGMENTATION["height_shift_range"],
    shear_range=AUGMENTATION["shear_range"],
    zoom_range=AUGMENTATION["zoom_range"],
    horizontal_flip=AUGMENTATION["horizontal_flip"],
    vertical_flip=AUGMENTATION["vertical_flip"],
    fill_mode=AUGMENTATION["fill_mode"],
)

# Validation and test data generators (no augmentation)
valid_test_data_gen = ImageDataGenerator(rescale=1.0 / 255)

# Data generators for training, validation, and testing
train_gen_new = train_data_gen.flow_from_dataframe(
    train_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=IMG_SIZE,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE,
)
valid_gen_new = valid_test_data_gen.flow_from_dataframe(
    valid_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=IMG_SIZE,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=True,
    batch_size=BATCH_SIZE,
)
test_gen_new = valid_test_data_gen.flow_from_dataframe(
    test_df_new,
    x_col="image_path",
    y_col="category_encoded",
    target_size=IMG_SIZE,
    class_mode="sparse",
    color_mode="rgb",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

def log_epoch_data(epoch, logs):
    logger.info(f"Epoch {epoch + 1:3.0} | Loss: {logs['loss']:.4f} | Accuracy: {logs['accuracy']:.4f} | Val Loss: {logs['val_loss']:.4f} | Val Accuracy: {logs['val_accuracy']:.4f}")

# Set up Lambda callback to log epoch data
log_epoch_callback = LambdaCallback(on_epoch_end=log_epoch_data)

# Set up early stopping callback, higher patience may result in better accuracy
early_stopping = EarlyStopping(
    monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True
)

# Setup callbacks
callbacks = [log_epoch_callback]
if USE_EARLY_STOPPING:
    callbacks.append(early_stopping)

def create_xception_model(input_shape, num_classes=8, learning_rate=1e-4):
    inputs = Input(shape=input_shape, name="Input_Layer")
    base_model = Xception(weights="imagenet", input_tensor=inputs, include_top=False)
    base_model.trainable = False
    x = base_model.output
    height, width, channels = x.shape[1], x.shape[2], x.shape[3]
    x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
    x = MultiHeadAttention(num_heads=NUM_ATTENTION_HEADS, key_dim=ATTENTION_KEY_DIM, name="Multi_Head_Attention")(
        x, x
    )
    x = Reshape((height, width, channels), name="Reshape_to_Spatial")(x)
    x = GaussianNoise(GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(x)
    x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
    x = Dense(512, activation="relu", name="FC_512")(x)
    x = BatchNormalization(name="Batch_Normalization")(x)
    x = Dropout(DROPOUT_RATE, name="Dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="Output_Layer")(x)
    model = Model(inputs=inputs, outputs=outputs, name="Xception_with_Attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=LOSS_FUNCTION,
        metrics=METRICS,
    )
    return model

img_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
cnn_model = create_xception_model(img_shape, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

history = cnn_model.fit(
    train_gen_new,
    validation_data=valid_gen_new,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if SAVE_BEST_MODEL:
    model_save_path = f"models/knee_osteo_model_{current_time}.keras"
    cnn_model.save(model_save_path)
    logger.info(f"Model saved as {model_save_path}")
    # Convert the model to Core ML format if on macOS
    if CONVERT_TO_COREML and system_platform == "Darwin":
        from coreml_util import convert_to_coreml # Custom module that has CoreML conversion function
        convert_to_coreml(cnn_model, logger)
else:
    logger.info("Model not saved. Set SAVE_BEST_MODEL to True to save the model.")

y_pred = cnn_model.predict(valid_gen_new)
y_true = valid_gen_new.labels


def ppo_loss(y_true, y_pred):
    epsilon = 0.2
    # Ensure the types of both tensors are compatible, casting to float32
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
    
    if ENABLE_MIXED_PRECISION:
        # Ensure both tensors are of the same type
        y_pred = tf.cast(y_pred, tf.float32)
        y_true_one_hot = tf.cast(y_true_one_hot, tf.float32)
    
    selected_probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
    old_selected_probs = tf.reduce_sum(
        tf.stop_gradient(y_pred) * y_true_one_hot, axis=-1
    )
    ratio = selected_probs / (old_selected_probs + 1e-10)
    clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
    loss = -tf.reduce_mean(tf.minimum(ratio, clipped_ratio))
    return loss


ppo_loss_value = ppo_loss(y_true, y_pred)
print(f"\nPPO Loss on Validation Data: {ppo_loss_value.numpy()}")

test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(
    test_labels, predicted_classes, target_names=list(test_gen_new.class_indices.keys())
)
print(report)

conf_matrix = confusion_matrix(test_labels, predicted_classes)

if ENABLE_PLOTS:
    plot_utils.plot_training_history(history, identifier=IDENTIFIER, save=SAVE_PLOTS)
    plot_utils.plot_confusion_matrix(conf_matrix, list(test_gen_new.class_indices.keys()), identifier=IDENTIFIER, save=SAVE_PLOTS)

end_time = time.perf_counter()
execution_time = end_time - start_time
logger.info(f"Execution time: {execution_time:.2f} seconds")

logger.info("--- END ---")
