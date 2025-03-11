import coremltools as ct
import coremltools.models


# Saves model, optimized for macOS neural engine
# Convert the model to Core ML format if on macOS
def convert_to_coreml(cnn_model, logger) -> coremltools.models.MLModel:
    logger.info(
        "Converting TensorFlow model to Core ML format for Apple Neural Engine..."
    )

    # Convert the trained model to Core ML format
    model_input = ct.ImageType(shape=(1, 224, 224, 3))  # Adjust input shape as needed

    coreml_model = ct.convert(
        cnn_model,  # Your trained TensorFlow/Keras model
        inputs=[model_input],
        compute_units=ct.ComputeUnit.ALL,  # Use ALL to leverage CPU, GPU, and Neural Engine
    )

    # Save the Core ML model
    coreml_model_path = "model.mlmodel"
    coreml_model.save(coreml_model_path)

    logger.info(f"Core ML model saved as {coreml_model_path}")

    # Optional: Load and test the Core ML model
    logger.info("Loading Core ML model for inference...")

    loaded_model = coremltools.models.MLModel(coreml_model_path)

    logger.info(
        "Core ML model loaded successfully! Ready for Neural Engine acceleration."
    )
    return loaded_model
