import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
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
    Multiply,
    Add,
    UpSampling2D,
    Concatenate,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LambdaCallback,
    ReduceLROnPlateau,
)
from tensorflow.keras.applications import Xception, EfficientNetV2B0, EfficientNetV2S
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.activations import relu, softmax, swish


class ProgressiveUnfreeze(tf.keras.callbacks.Callback):
    def __init__(self, logger, model, unfreeze_schedule: dict, base_model_layers=None):
        """
        Progressive unfreezing for models where base layers are flat (not nested)

        Args:
            model: The full model
            unfreeze_schedule: Dict mapping epoch -> percentage to unfreeze
            base_model_layers: List of layer names that belong to the base model part
                              If None, will try to identify them automatically
        """
        super(ProgressiveUnfreeze, self).__init__()
        self._model = model
        self.unfreeze_schedule = unfreeze_schedule
        self.logger = logger

        # If base model layers not provided, try to identify them
        if base_model_layers is None:
            # For EfficientNetV2S, identify by block prefix
            self.base_model_layers = []
            for layer in model.layers:
                # Identify base model layers by common prefixes
                if any(
                    layer.name.startswith(prefix)
                    for prefix in ["stem_", "block", "top_", "rescaling"]
                ):
                    self.base_model_layers.append(layer.name)
        else:
            self.base_model_layers = base_model_layers

        self.total_layers = len(self.base_model_layers)

        self.logger.info(f"Identified {self.total_layers} base model layers")

    def create_unfreeze_schedule(self, epochs: int) -> dict:
        """
        Unfreeze schedule is a dict mapping epoch -> percentage of layers to unfreeze
        Comes from config.json. A sample unfreeze schedule would look like this:
        "progressive_unfreeze_schedule": {
            "0": 0,
            "10": 0.05,
            "20": 0.15,
            "30": 0.3,
            "40": 0.5,
            "50": 0.75
        }
        Where they key is the epoch and the value is the percentage of layers to unfreeze.

        Returns:
            dict: Unfreeze schedule (epoch: int -> percentage: float)
        """
        # Convert the string keys to integers
        progressive_unfreeze_schedule = {
            int(k): float(v) for k, v in self.PROGRESSIVE_UNFREEZE_SCHEDULE.items()
        }
        return progressive_unfreeze_schedule

    def on_epoch_begin(self, epoch, logs=None):
        if epoch in self.unfreeze_schedule:
            percentage = self.unfreeze_schedule[epoch]
            layers_to_unfreeze = int(self.total_layers * percentage)

            # Freeze all base model layers first
            for layer_name in self.base_model_layers:
                layer = self._model.get_layer(layer_name)
                layer.trainable = False

            # Then unfreeze the specified percentage from the end
            for layer_name in self.base_model_layers[-layers_to_unfreeze:]:
                layer = self._model.get_layer(layer_name)
                layer.trainable = True

            # Recompile the model to apply changes
            self._model.compile(
                optimizer=Adam(learning_rate=self._model.optimizer.learning_rate),
                loss=self._model.loss,
                metrics=self._model.metrics,
            )

            trainable_weights = len(self._model.trainable_weights)
            self.logger.info(
                f"\nEpoch {epoch}: Unfreezing {percentage*100}% of base model "
                f"({layers_to_unfreeze}/{self.total_layers} layers). "
                f"Trainable weights: {trainable_weights}"
            )


class ModelCreator:
    def __init__(self, logger, hyperparameters: dict, config: dict):
        self.logger = logger
        self.hyperparameters = hyperparameters
        self.config = config
        self.BASE_MODEL = hyperparameters["base_model"]
        self.BASE_MODEL_WEIGHTS = hyperparameters["base_model_weights"]
        self.BASE_MODEL_INCLUDE_TOP = hyperparameters["base_model_include_top"]
        self.UNFREEZE_LAYERS = config["unfreeze_layers"]
        self.USE_PROGRESSIVE_UNFREEZING = config["use_progressive_unfreezing"]
        self.UNFREEZE_LAST_N_LAYERS = hyperparameters["unfreeze_last_n_layers"]
        self.GAUSSIAN_NOISE_STDDEV = hyperparameters["gaussian_noise_stddev"]
        self.DENSE_LAYERS = hyperparameters["dense_layers"]
        self.ACTIVATION_FN = hyperparameters["activation_fn"]
        self.LR_SCHEDULE_TYPE = hyperparameters["lr_schedule_type"]
        self.L2_REG_RATE = hyperparameters["l2_reg_rate"]
        self.DROPOUT_RATE = hyperparameters["dropout_rate"]
        self.NUM_CLASSES = hyperparameters["num_classes"]
        self.LEARNING_RATE = hyperparameters["learning_rate"]
        self.LOSS_FUNCTION = hyperparameters["loss_function"]
        self.METRICS = hyperparameters["metrics"]
        self.NUM_ATTENTION_HEADS = hyperparameters["num_attention_heads"]
        # Based on the number of channels in the input (3 in this case for RGB)
        self.ATTENTION_KEY_DIM = hyperparameters["attention_key_dim"]
        self.EARLY_STOPPING_PATIENCE = hyperparameters["early_stopping_patience"]
        self.MIN_LEARNING_RATE = hyperparameters["min_learning_rate"]
        self.LR_PATIENCE = hyperparameters["lr_patience"]
        self.PROGRESSIVE_UNFREEZE_SCHEDULE = hyperparameters[
            "progressive_unfreeze_schedule"
        ]

        # Create Learning Rate Scheduler
        self.lr_schedule, self.lr_callback = self.create_lr_scheduler()

    def create_lr_scheduler(self):
        # TODO: Test other learning rate schedules such as reduce on plateau, cosine decay, etc.
        lr_schedule = None
        lr_callback = None
        self.logger.info(
            f"Creating Learning Rate Scheduler with {self.LR_SCHEDULE_TYPE}"
        )
        if self.LR_SCHEDULE_TYPE == "ExponentialDecay":
            lr_schedule = ExponentialDecay(
                initial_learning_rate=self.LEARNING_RATE,
                decay_steps=10000,
                decay_rate=0.96,
                staircase=True,
            )
        elif self.LR_SCHEDULE_TYPE == "ReduceLROnPlateau":
            lr_callback = ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=self.LR_PATIENCE,
                min_lr=self.MIN_LEARNING_RATE,
                verbose=1,
            )
            lr_schedule = self.LEARNING_RATE  # Use the initial learning rate
        if lr_schedule == None:
            # No scheduler, just basic learning rate
            lr_schedule = self.LEARNING_RATE
        return lr_schedule, lr_callback

    def create_xception_model(self, input_shape):
        inputs = Input(shape=input_shape, name="Input_Layer")
        base_model = Xception(
            weights=self.BASE_MODEL_WEIGHTS,
            input_tensor=inputs,
            include_top=self.BASE_MODEL_INCLUDE_TOP,
        )

        if self.UNFREEZE_LAYERS and not self.USE_PROGRESSIVE_UNFREEZING:
            # Static unfreezing
            base_model.trainable = True
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
            # Progressive unfreezing is handled by a custom callback in the training loop
            base_model.trainable = False

        x = base_model.output
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
        x = MultiHeadAttention(
            num_heads=self.NUM_ATTENTION_HEADS,
            key_dim=self.ATTENTION_KEY_DIM,
            name="Multi_Head_Attention",
        )(x, x)
        x = Reshape((height, width, channels), name="Reshape_to_Spatial")(x)
        x = GaussianNoise(self.GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(x)
        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
        x = Dense(
            self.DENSE_LAYERS,
            activation=self.ACTIVATION_FN,
            kernel_regularizer=regularizers.l2(self.L2_REG_RATE),
            name="FC_512",
        )(x)
        x = BatchNormalization(name="Batch_Normalization")(x)
        x = Dropout(self.DROPOUT_RATE, name="Dropout")(x)
        outputs = Dense(self.NUM_CLASSES, activation="softmax", name="Output_Layer")(x)
        model = Model(inputs=inputs, outputs=outputs, name="Xception_with_Attention")
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss=self.LOSS_FUNCTION,
            metrics=self.METRICS,
        )
        return model

    def create_efficientnetv2_model(self, input_shape):
        inputs = Input(shape=input_shape, name="Input_Layer")
        base_model = EfficientNetV2S(
            weights=self.BASE_MODEL_WEIGHTS,
            input_tensor=inputs,
            include_top=self.BASE_MODEL_INCLUDE_TOP,
        )

        if self.UNFREEZE_LAYERS and not self.USE_PROGRESSIVE_UNFREEZING:
            # Static unfreezing
            base_model.trainable = True
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
            # Progressive unfreezing is handled by a custom callback in the training loop
            base_model.trainable = False

        x = base_model.output
        # Add attention mechanism similar to Xception model
        height, width, channels = x.shape[1], x.shape[2], x.shape[3]
        x_orig = x  # Store original for skip connection
        x = Reshape((height * width, channels), name="Reshape_to_Sequence")(x)
        x = MultiHeadAttention(
            num_heads=self.NUM_ATTENTION_HEADS,
            key_dim=self.ATTENTION_KEY_DIM,
            name="Multi_Head_Attention",
        )(x, x)
        x = Reshape((height, width, channels), name="Reshape_to_Spatial")(x)
        x = Add(name="Skip_Connection")([x, x_orig])  # Skip connection

        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
        x = GaussianNoise(self.GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(x)
        x = Dense(
            self.DENSE_LAYERS,
            kernel_regularizer=regularizers.l2(
                self.L2_REG_RATE
            ),  # Use the config value
            name="FC_512",
        )(x)
        x = BatchNormalization(name="Batch_Normalization")(x)
        x = Activation(swish, name="Activation")(x)  # EfficientNet typically uses swish
        x = Dropout(self.DROPOUT_RATE, name="Dropout")(x)
        outputs = Dense(self.NUM_CLASSES, activation="softmax", name="Output_Layer")(x)
        model = Model(
            inputs=inputs, outputs=outputs, name="EfficientNetV2_with_Attention"
        )
        model.compile(
            optimizer=Adam(learning_rate=self.lr_schedule),
            loss=self.LOSS_FUNCTION,
            metrics=self.METRICS,
        )
        return model

    def create_model(self, input_shape):
        self.logger.info(f"Creating model with base model: {self.BASE_MODEL}")
        if self.BASE_MODEL == "Xception":
            return self.create_xception_model(input_shape)
        elif self.BASE_MODEL == "EfficientNetV2S":
            return self.create_efficientnetv2_model(input_shape)
        else:
            self.logger.error(f"Unsupported base model: {self.BASE_MODEL}")
