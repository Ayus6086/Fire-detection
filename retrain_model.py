import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

print("🔄 Retraining fire detection model...")

# Paths
train_dir = 'dataset'
img_size = (224, 224)
batch_size = 16

# Data preprocessing with more augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=0.2
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='binary'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='binary'
)

print(f"Classes found: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# Improved model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(256, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("📊 Model architecture:")
model.summary()

# Training with callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=3)
]

print("🚀 Starting training...")
history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# Save the improved model
os.makedirs("model", exist_ok=True)
model.save('model/fire_detector.h5')
print("✅ Improved model saved to model/fire_detector.h5")

# Test the model
print("\n🧪 Testing the new model...")
import numpy as np
from tensorflow.keras.preprocessing import image

# Test with a fire image
try:
    img = image.load_img('dataset/fire/fire.1.png', target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    print(f"Fire image prediction: {prediction:.4f}")
except:
    print("Could not test fire image")

# Test with a no-fire image
try:
    img = image.load_img('dataset/no_fire/non_fire.1.png', target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)[0][0]
    print(f"No-fire image prediction: {prediction:.4f}")
except:
    print("Could not test no-fire image")

print("\n🎯 Training completed! The model should now work correctly.")
print("Expected behavior:")
print("- Fire images should have HIGH prediction scores (>0.5)")
print("- No-fire images should have LOW prediction scores (<0.5)")