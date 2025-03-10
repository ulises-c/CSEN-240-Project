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


class ModelCreator:
    def __init__(self, logger, hyperparameters: dict, config: dict):
        self.logger = logger
        self.hyperparameters = hyperparameters
        self.config = config
        self.BASE_MODEL = hyperparameters["base_model"]
        self.BASE_MODEL_WEIGHTS = hyperparameters["base_model_weights"]
        self.BASE_MODEL_INCLUDE_TOP = hyperparameters["base_model_include_top"]
        self.UNFREEZE_LAYERS = config["unfreeze_layers"]
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

        if self.UNFREEZE_LAYERS:
            base_model.trainable = True
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
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
            drop_connect_rate=0.2,  # Stochastic Depth
        )

        if self.UNFREEZE_LAYERS:
            base_model.trainable = True
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
            base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)
        x = GaussianNoise(self.GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(x)
        x = Dense(
            self.DENSE_LAYERS,
            kernel_regularizer=regularizers.l2(0.001),
            name="FC_512",
        )(x)
        x = BatchNormalization(name="Batch_Normalization")(x)
        x = Activation(swish, name="Activation")(x)
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

    def create_efficientnetv2_model_test(self, input_shape):
        inputs = Input(shape=input_shape, name="Input_Layer")

        # Correct the parameter name (drop_connect_Rate → drop_connect_rate)
        base_model = EfficientNetV2S(
            weights=self.BASE_MODEL_WEIGHTS,
            input_tensor=inputs,
            include_top=self.BASE_MODEL_INCLUDE_TOP,
            drop_connect_rate=0.2,  # Stochastic Depth (fixed parameter name)
        )

        # Implement progressive layer unfreezing for better fine-tuning
        if self.UNFREEZE_LAYERS:
            base_model.trainable = True
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
            base_model.trainable = False

        # Extract intermediate features for multi-scale representation
        high_level_features = base_model.output
        mid_level_features = base_model.get_layer("block4a_expand_activation").output

        # Add attention mechanism - Squeeze and Excitation
        def apply_se_attention(x, ratio=16):
            channels = x.shape[-1]
            # Squeeze operation (global average pooling)
            se = GlobalAveragePooling2D()(x)
            # Excitation operation (two FC layers with bottleneck)
            se = Dense(channels // ratio, activation="swish")(se)
            se = Dense(channels, activation="sigmoid")(se)
            # Reshape for multiplication
            se = Reshape((1, 1, channels))(se)
            # Scale the input
            return Multiply()([x, se])

        # Apply attention to high-level features
        attended_features = apply_se_attention(high_level_features)

        # Feature fusion with skip connection
        # First apply 1x1 convolution to harmonize feature dimensions
        mid_features_processed = Conv2D(
            attended_features.shape[-1] // 2,  # Reduce channels
            kernel_size=1,
            padding="same",
            activation="swish",
            name="Mid_Features_1x1",
        )(mid_level_features)

        # Upsample if needed to match dimensions
        if attended_features.shape[1:3] != mid_features_processed.shape[1:3]:
            attended_features_upsampled = UpSampling2D(
                size=(
                    mid_features_processed.shape[1] // attended_features.shape[1],
                    mid_features_processed.shape[2] // attended_features.shape[2],
                )
            )(attended_features)
            # Concatenate features for multi-scale representation
            fused_features = Concatenate(axis=-1)(
                [mid_features_processed, attended_features_upsampled]
            )
        else:
            fused_features = Concatenate(axis=-1)(
                [mid_features_processed, attended_features]
            )

        # Apply 1x1 convolution to reduce feature dimensions after fusion
        fused_features = Conv2D(
            attended_features.shape[-1],
            kernel_size=1,
            padding="same",
            activation="swish",
            name="Fused_Features_1x1",
        )(fused_features)

        # Global pooling
        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(fused_features)

        # Add light regularization
        x = GaussianNoise(self.GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(x)

        # First fully connected layer with optimized regularization
        x = Dense(
            self.DENSE_LAYERS,
            kernel_regularizer=regularizers.l2(
                self.L2_REG_RATE
            ),  # Use the hyperparameter value
            name="FC_512",
        )(x)
        x = BatchNormalization(momentum=0.9, name="Batch_Normalization")(x)
        x = Activation(swish, name="Activation")(x)

        # Add a second fully connected layer for better feature hierarchy
        x = Dense(
            self.DENSE_LAYERS // 2,  # Smaller layer for dimensionality reduction
            kernel_regularizer=regularizers.l2(self.L2_REG_RATE),
            name="FC_256",
        )(x)
        x = BatchNormalization(name="Batch_Normalization_2")(x)
        x = Activation(swish, name="Activation_2")(x)
        x = Dropout(self.DROPOUT_RATE, name="Dropout")(x)

        # Output layer
        outputs = Dense(self.NUM_CLASSES, activation="softmax", name="Output_Layer")(x)

        # Build model
        model = Model(
            inputs=inputs, outputs=outputs, name="EfficientNetV2_with_Attention"
        )

        # Use a more advanced optimizer for better convergence
        optimizer = Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            clipnorm=1.0,  # Gradient clipping for stability
        )

        model.compile(
            optimizer=optimizer,
            loss=self.LOSS_FUNCTION,
            metrics=self.METRICS,
        )

        return model

    def create_xception_model_test(self, input_shape):
        inputs = Input(shape=input_shape, name="Input_Layer")

        # Initialize base model with pre-trained weights
        base_model = Xception(
            weights=self.BASE_MODEL_WEIGHTS,
            input_tensor=inputs,
            include_top=self.BASE_MODEL_INCLUDE_TOP,
        )

        # Implement strategic layer unfreezing
        if self.UNFREEZE_LAYERS:
            base_model.trainable = True
            # Freeze batch norm layers for stability during fine-tuning
            for layer in base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False
                elif "batch_normalization" in layer.name:
                    layer.trainable = False

            # Freeze early layers and unfreeze later layers
            for layer in base_model.layers[: -self.UNFREEZE_LAST_N_LAYERS]:
                layer.trainable = False
        else:
            base_model.trainable = False

        # Extract features from different depths for multi-scale representation
        high_level_features = base_model.output
        mid_level_features = base_model.get_layer("block13_sepconv1_act").output
        low_level_features = base_model.get_layer("block4_sepconv1_act").output

        # Apply attention mechanism to high-level features
        height, width, channels = (
            high_level_features.shape[1],
            high_level_features.shape[2],
            high_level_features.shape[3],
        )

        # Self-attention on high-level features
        x_reshape = Reshape((height * width, channels), name="Reshape_to_Sequence")(
            high_level_features
        )
        x_attended = MultiHeadAttention(
            num_heads=self.NUM_ATTENTION_HEADS,
            key_dim=self.ATTENTION_KEY_DIM,
            name="Multi_Head_Attention",
        )(x_reshape, x_reshape)
        x_spatial = Reshape((height, width, channels), name="Reshape_to_Spatial")(
            x_attended
        )

        # Add residual connection to preserve gradient flow
        attended_features = Add()([high_level_features, x_spatial])

        # Apply 1x1 convolution to reduce feature dimensions for mid and low-level features
        mid_features_processed = Conv2D(
            256,
            kernel_size=1,
            padding="same",
            activation="relu",
            name="Mid_Features_1x1",
        )(mid_level_features)

        low_features_processed = Conv2D(
            128,
            kernel_size=1,
            padding="same",
            activation="relu",
            name="Low_Features_1x1",
        )(low_level_features)

        # Upsample high-level features to match mid-level dimensions
        high_to_mid = UpSampling2D(
            size=(
                mid_features_processed.shape[1] // attended_features.shape[1],
                mid_features_processed.shape[2] // attended_features.shape[2],
            ),
            name="Upsample_High_to_Mid",
        )(attended_features)

        # Fuse high and mid-level features
        high_mid_fusion = Concatenate(axis=-1, name="High_Mid_Fusion")(
            [mid_features_processed, high_to_mid]
        )

        # Refine fused features with 3x3 convolution
        high_mid_fusion = Conv2D(
            384,
            kernel_size=3,
            padding="same",
            activation="relu",
            name="High_Mid_Fusion_Conv",
        )(high_mid_fusion)

        # Upsample fused features to match low-level dimensions
        fused_to_low = UpSampling2D(
            size=(
                low_features_processed.shape[1] // high_mid_fusion.shape[1],
                low_features_processed.shape[2] // high_mid_fusion.shape[2],
            ),
            name="Upsample_Fused_to_Low",
        )(high_mid_fusion)

        # Final feature fusion across all scales
        final_fusion = Concatenate(axis=-1, name="Final_Fusion")(
            [low_features_processed, fused_to_low]
        )

        # Apply final 1x1 convolution to harmonize feature dimensions
        final_features = Conv2D(
            512,
            kernel_size=1,
            padding="same",
            activation="relu",
            name="Final_Features_Conv",
        )(final_fusion)

        # Add regularization
        x = GaussianNoise(self.GAUSSIAN_NOISE_STDDEV, name="Gaussian_Noise")(
            final_features
        )

        # Global pooling to get feature vector
        x = GlobalAveragePooling2D(name="Global_Avg_Pooling")(x)

        # Classification head with proper regularization
        x = Dense(
            self.DENSE_LAYERS,
            activation=self.ACTIVATION_FN,
            kernel_regularizer=regularizers.l2(self.L2_REG_RATE),
            name="FC_512",
        )(x)
        x = BatchNormalization(name="Batch_Normalization")(x)

        # Add a second FC layer for better feature hierarchy
        x = Dense(
            self.DENSE_LAYERS // 2,
            activation=self.ACTIVATION_FN,
            kernel_regularizer=regularizers.l2(self.L2_REG_RATE),
            name="FC_256",
        )(x)
        x = BatchNormalization(name="Batch_Normalization_2")(x)
        x = Dropout(self.DROPOUT_RATE, name="Dropout")(x)

        # Output layer
        outputs = Dense(self.NUM_CLASSES, activation="softmax", name="Output_Layer")(x)

        # Build model
        model = Model(inputs=inputs, outputs=outputs, name="Xception_Enhanced")

        # Use a sophisticated optimizer configuration
        optimizer = Adam(
            learning_rate=self.lr_schedule,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            clipnorm=1.0,  # Gradient clipping for stability
        )

        model.compile(
            optimizer=optimizer,
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
