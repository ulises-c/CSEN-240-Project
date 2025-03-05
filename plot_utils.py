import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os

def plot_label_distribution(df):
    """Plots the distribution of labels in the dataset."""
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="label", palette="viridis")
    plt.title("Distribution of Tumor Types", fontsize=14, fontweight="bold")
    plt.xlabel("Tumor Type", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.xticks(rotation=45)
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha="center", va="bottom", fontsize=11, color="black", 
                           xytext=(0, 5), textcoords="offset points")
    plt.show()

def plot_sample_images(df, categories, num_images=5):
    """Displays sample images from each category."""
    plt.figure(figsize=(15, 12))
    for i, category in enumerate(categories):
        category_images = df[df["label"] == category]["image_path"].iloc[:num_images]
        for j, img_path in enumerate(category_images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(len(categories), num_images, i * num_images + j + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(category)
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Plots the accuracy and loss curves after training."""
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

def plot_confusion_matrix(conf_matrix, class_names):
    """Plots the confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
