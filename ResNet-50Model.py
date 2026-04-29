import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


train_dir = "data/Train"
val_dir = "data/Check"
out_dir = "Models/ResNet50"

os.makedirs(out_dir, exist_ok=True)
model_path = f"{out_dir}/accident_model_resnet50.h5"

img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    rescale=1./255

)



val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet50.preprocess_input)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary")



val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size= batch_size,
    class_mode="binary",
    shuffle=False)




base_model=ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3))

base_model.trainable = True


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)



model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']

)



callbacks = [
    ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )
]

training=model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=18,
    
    callbacks=callbacks
)




print("=========="*10)

print("\n=== Phase 1:  ResNet50, ===\n")


final_model_path = f"{out_dir}/accident_model_resnet50_final.h5"
model.save(final_model_path)

print(f"Training complete. Best weights: {model_path}")

print(f"Final model saved to: {final_model_path}")
print(f"Training history: {training.history}")
print("Training complete....")

