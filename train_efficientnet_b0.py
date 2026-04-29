
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os


train_dir = "data/Train"
val_dir = "data/Check"
out_dir = "Models/EfficientNetB0"
os.makedirs(out_dir, exist_ok=True)
weights_path = f"{out_dir}/accident_model_efficientnet_b0.h5"
img_size = (224, 224)
batch_size = 32



train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=8,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.75, 1.25),
    channel_shift_range=25.0,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary",
)

print("Class mapping:", train_generator.class_indices)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)

# --- Phase 1 ---

base_model.trainable = False

model = Sequential(
    [
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

checkpoint = ModelCheckpoint(
    weights_path,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

class_weight = {0: 1.2,
                1: 1.0}

phase1_callbacks = [
    checkpoint,
    EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

print("\n=== Phase 1:  EfficientNetB0, ===\n")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    class_weight=class_weight,
    callbacks=phase1_callbacks,
)


base_model.trainable = True
fine_tune_at = 155
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
print(
    f"\n=== Phase 2: fine-tuning ({trainable_count} trainable base layers) ===\n"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
#Phase 2 Fine-tuning
print("\n=== Phase 2: Fine-tuning ===\n")


phase2_callbacks = [
    checkpoint,
    EarlyStopping(
        monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
    ),
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=55,
    class_weight=class_weight,
    callbacks=phase2_callbacks,
)

print(f"Training complete. Best weights: {weights_path}")
