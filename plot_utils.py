"""Utility functions for plotting data and model performance."""

import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os


class PlotUtils:
    def __init__(self, logger=None, save_dir="plots", identifier=None, save=False, show=False):
        """Initialize with optional parameters for saving, showing plots, and a unique identifier."""
        self.save_dir = save_dir
        self.identifier = identifier
        self.save = save
        self.show = show
        self.plot_show_time = 3
        self.logger = logger

    def save_plot(self, plot_name):
        """Saves the current plot as a PNG file with a unique identifier."""
        if not self.identifier:
            print("Identifier not set. Please provide one during initialization or when calling methods.")
            return
        
        full_path = os.path.join(self.save_dir, self.identifier)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        save_path = os.path.join(full_path, f"{plot_name}_{self.identifier}.png")
        plt.savefig(save_path)
        if self.logger:
            self.logger.info(f"Plot saved at {save_path}")

    def plot_label_distribution(self, df):
        """Plots the distribution of labels in the dataset."""
        sns.set_style("whitegrid")
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
        if self.save and self.identifier:
            self.save_plot("label_distribution")
        if self.show:
            plt.pause(self.plot_show_time) # Pause for a few seconds before closing
            plt.close()

    def plot_sample_images(self, df, categories, num_images=5):
        """Displays sample images from each category."""
        sns.set_style("darkgrid")
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
        if self.save and self.identifier:
            self.save_plot("sample_images")
        plt.tight_layout()
        if self.show:
            plt.pause(self.plot_show_time) # Pause for a few seconds before closing
            plt.close()

    def plot_training_history(self, history):
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

        if self.save and self.identifier:
            self.save_plot("training_history")
        if self.show:
            plt.pause(self.plot_show_time) # Pause for a few seconds before closing
            plt.close()

    def plot_confusion_matrix(self, conf_matrix, class_names):
        """Plots the confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        if self.save and self.identifier:
            self.save_plot("confusion_matrix")
        if self.show:
            plt.pause(self.plot_show_time) # Pause for a few seconds before closing
            plt.close()
