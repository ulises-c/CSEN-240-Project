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
import utils.plot_utils as plot_utils  # Custom module that has plotting functions
import json  # To load hyperparameters from a JSON file, also used to save model history
import random  # To set random seed for reproducibility

from utils.system_specs_util import (
    log_system_specs,
)  # Custom module that has system specs logging function
from utils.create_model_util import (
    ModelCreator,
)  # Custom module that has model creation function
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
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers, layers, models
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LambdaCallback,
    ReduceLROnPlateau,
)

# TODO: Test with other models such as ResNet50, InceptionV3, MobileNetV2, DenseNet121, VGG16, VGG19

# from tensorflow.keras.mixed_precision import experimental as mixed_precision # may be useful for training on Apple Silicon or Nvidia GPUs with less VRAM
from tensorflow.keras import (
    mixed_precision,
)  # may be useful for training on Apple Silicon or Nvidia GPUs with less VRAM
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
    GaussianNoise,
    MultiHeadAttention,
    Reshape,
)

# Potential TODOs
# TODO: Add support for sklearn.utils shuffle
# TODO: Add support for sklearn.utils resample
# TODO: Add support for sklearn.metrics accuracy_score
# TODO: Checkout Tensorflow Logging
# TODO: Checkout Tensorboard
# TODO: Add support for saving model history to a JSON file
# TODO: Add support for saving model checkpoints

# Actual TODOs

# TODO: Dynamic batch sizing based on hardware
# Example: JSON will have fixed batch based on hardware (fine tuned by user)
# Then the code will check which hardware it is running on and choose the batch size accordingly
# For example on an RTX 3070, batch size with mixed precision may be able to go as high as 256
# But on a M4 Mac Mini, it may only be able to go as high as 32
# And without mixed precision, it may only be able to go as high as 8

start_time = time.perf_counter()

# Load hyperparameters from a JSON file
with open("config.json", "r") as f:
    config = json.load(f)

### Config section of the JSON file
CONFIG = config["config"]
ENABLE_PLOTS = config["config"]["enable_plots"]
SAVE_PLOTS = config["config"]["save_plots"]
SHOW_PLOTS = config["config"]["show_plots"]
ENABLE_TF_DETERMINISM = config["config"]["enable_tf_determinism"]
SAVE_BEST_MODEL = config["config"]["save_best_model"]
CONVERT_TO_COREML = config["config"]["convert_to_coreml"]
UNFREEZE_LAYERS = config["config"]["unfreeze_layers"]
RANDOM_SEED = config["config"]["random_seed"]
ENABLE_MIXED_PRECISION = config["config"]["enable_mixed_precision"]
USE_EARLY_STOPPING = config["config"]["use_early_stopping"]

### Hyperparameters section of the JSON file
HYPERPARAMETERS = config["hyperparameters"]
# Reducing may help with VRAM issues, but may also reduce accuracy, original was 16
BATCH_SIZE = config["hyperparameters"]["batch_size"]
IMG_SIZE = tuple(config["hyperparameters"]["img_size"])
CHANNELS = config["hyperparameters"]["channels"]
LEARNING_RATE = config["hyperparameters"]["learning_rate"]
EPOCHS = config["hyperparameters"]["epochs"]
EARLY_STOPPING_PATIENCE = config["hyperparameters"]["early_stopping_patience"]
TRAIN_SPLIT = config["hyperparameters"]["train_split"]
VALID_TEST_SPLIT = config["hyperparameters"]["valid_test_split"]

### Image augmentation section of the JSON file
ROTATION_RANGE = config["augmentation"]["rotation_range"]
WIDTH_SHIFT_RANGE = config["augmentation"]["width_shift_range"]
HEIGHT_SHIFT_RANGE = config["augmentation"]["height_shift_range"]
SHEAR_RANGE = config["augmentation"]["shear_range"]
ZOOM_RANGE = config["augmentation"]["zoom_range"]
HORIZONTAL_FLIP = config["augmentation"]["horizontal_flip"]
VERTICAL_FLIP = config["augmentation"]["vertical_flip"]
FILL_MODE = config["augmentation"]["fill_mode"]
# TODO: Implement other image augmentation techniques such as brightness, contrast, etc.

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

# Generate log file name with ISO format date-time and system platform
current_time = datetime.now().isoformat(timespec="seconds")
system_platform = platform.system()
IDENTIFIER = f"{current_time}_{system_platform}"
out_dir = "out"
OUT_FULL_PATH = os.path.join(out_dir, IDENTIFIER)  # out/{IDENTIFIER}
# Check if logs directory exists, create if not
if not os.path.exists(OUT_FULL_PATH):
    os.makedirs(OUT_FULL_PATH)
log_file_name = f"{OUT_FULL_PATH}/knee_osteo_{IDENTIFIER}.log"

# Configure logging to write to the generated log file
logging.basicConfig(
    level=logging.INFO,
    filename=log_file_name,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(
    logging.INFO
)  # Set to INFO to display only important information, DEBUG to display all information
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# Store copy of config.json in the output directory
json_copy_name = f"{OUT_FULL_PATH}/config_{IDENTIFIER}.json"
try:
    shutil.copy("config.json", json_copy_name)
    logger.info(f"config.json copied to {json_copy_name}")
except Exception as e:
    logger.error(f"Error copying config.json to {json_copy_name}: {e}")

# Store a copy of this script in the output directory for reproducibility and debugging
script_copy_name = f"{OUT_FULL_PATH}/knee-osteo_{IDENTIFIER}.py"
try:
    shutil.copy(__file__, script_copy_name)
    logger.info(f"knee-osteo.py copied to {script_copy_name}")
except Exception as e:
    logger.error(f"Error copying knee-osteo.py to {script_copy_name}: {e}")

### Log important information
gpus = tf.config.list_physical_devices("GPU")
logger.info("--- START ---")
logger.info(f"Start Time: {current_time}")
log_system_specs(logger, gpus)
logger.info(f"--- config.json ---")
for key, value in config.items():
    logger.info(f"{key}: {value}")
logger.info(f"--- config.json ---")

# Enable GPU acceleration if available
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if system_platform == "Linux":
            logger.info("Using CUDA for acceleration on Nvidia GPU")
        elif system_platform == "Darwin":  # macOS
            logger.info("Using Metal backend for acceleration on Apple Silicon")

    except RuntimeError as e:
        logger.error(e)
else:
    logger.info("No GPU found. Running on CPU.")

data_path = "images/Knee_Osteoarthritis_Classification"  # Extracted Camino zip file
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
# logger.info(df.shape)
# logger.info(df.duplicated().sum())
# logger.info(df.isnull().sum())
# logger.info(df.info())
# logger.info("Unique labels: {}".format(df["label"].unique()))
# logger.info("Label counts: {}".format(df["label"].value_counts()))


if ENABLE_PLOTS:
    plotter = plot_utils.PlotUtils(
        logger=logger,
        save_dir=out_dir,
        identifier=IDENTIFIER,
        save=SAVE_PLOTS,
        show=SHOW_PLOTS,
    )
    plotter.plot_label_distribution(df)
    plotter.plot_sample_images(df, categories, num_images=5)

label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["label"])
df = df[["image_path", "category_encoded"]]


ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[["image_path"]], df["category_encoded"])
df_resampled = pd.DataFrame(X_resampled, columns=["image_path"])
df_resampled["category_encoded"] = y_resampled
# logger.info("\nClass distribution after oversampling:")
# logger.info(df_resampled["category_encoded"].value_counts())
# logger.info(df_resampled)


df_resampled["category_encoded"] = df_resampled["category_encoded"].astype(str)

warnings.filterwarnings("ignore")

train_df_new, temp_df_new = train_test_split(
    df_resampled,
    train_size=TRAIN_SPLIT,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=df_resampled["category_encoded"],
)
# logger.info(train_df_new.shape)
# logger.info(temp_df_new.shape)

valid_df_new, test_df_new = train_test_split(
    temp_df_new,
    test_size=VALID_TEST_SPLIT,
    shuffle=True,
    random_state=RANDOM_SEED,
    stratify=temp_df_new["category_encoded"],
)
# logger.info(valid_df_new.shape)
# logger.info(test_df_new.shape)

# Train data generator (with augmentation)
### NOTE: This may not be allowed within project constraints ###
train_data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=ROTATION_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    shear_range=SHEAR_RANGE,
    zoom_range=ZOOM_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    vertical_flip=VERTICAL_FLIP,
    fill_mode=FILL_MODE,
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
    # Temporarily disable the console handler, to prevent duplicate lines in console
    logger.removeHandler(console)
    # Log epoch data
    logger.info(
        f"Epoch {epoch + 1:3} | Accuracy: {logs['accuracy']:.4f} | Loss: {logs['loss']:.4f} | Val Accuracy: {logs['val_accuracy']:.4f} | Val Loss: {logs['val_loss']:.4f}"
    )
    # Re-enable the console handler
    logger.addHandler(console)


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

img_shape = (IMG_SIZE[0], IMG_SIZE[1], CHANNELS)
model_creator = ModelCreator(logger, HYPERPARAMETERS, CONFIG)
cnn_model = model_creator.create_model(img_shape)

# Add learning rate callback if it exists
if model_creator.lr_callback:
    callbacks.append(model_creator.lr_callback)

logger.info(f"Model summary (BEFORE): {cnn_model.summary()}")

history = cnn_model.fit(
    train_gen_new,
    validation_data=valid_gen_new,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,  # Verbose 0 for silent, 1 for progress bar, 2 for one line per epoch
)

logger.info(f"Model summary (AFTER): {cnn_model.summary()}")

if SAVE_BEST_MODEL:
    out_dir = f"out/{IDENTIFIER}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model_save_path = f"{out_dir}/knee_osteo_model_{IDENTIFIER}.keras"
    cnn_model.save(model_save_path)
    logger.info(f"Model saved as {model_save_path}")
    # Convert the model to Core ML format if on macOS
    if CONVERT_TO_COREML and system_platform == "Darwin":
        from utils.coreml_util import (
            convert_to_coreml,
        )  # Custom module that has CoreML conversion function

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
logger.info(f"PPO Loss on Validation Data: {ppo_loss_value.numpy()}")

test_labels = test_gen_new.classes
predictions = cnn_model.predict(test_gen_new)
predicted_classes = np.argmax(predictions, axis=1)

report = classification_report(
    test_labels, predicted_classes, target_names=list(test_gen_new.class_indices.keys())
)
logger.info(f"Classification Report:\n{report}")

conf_matrix = confusion_matrix(test_labels, predicted_classes)

if ENABLE_PLOTS:
    plotter.plot_training_history(history)
    plotter.plot_confusion_matrix(
        conf_matrix, class_names=list(test_gen_new.class_indices.keys())
    )


def exec_time(end_time: float) -> str:
    # Convert performance time to hours, minutes, and seconds
    perf_time = end_time - start_time
    hours = perf_time // 3600
    minutes = (perf_time % 3600) // 60
    seconds = perf_time % 60
    milliseconds = (seconds - int(seconds)) * 1000
    exec_time_str = (
        f"{int(hours):02}H {int(minutes):02}M {int(seconds):02}.{int(milliseconds):03}S"
    )
    logger.info(f"Execution Time: {exec_time_str}")


execution_time = exec_time(time.perf_counter())
logger.info("--- END ---")
