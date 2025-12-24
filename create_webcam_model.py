import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.keras.preprocessing import image
import os

print("🔄 Creating a simple, reliable fire detection model...")

# Paths
train_dir = 'dataset'
img_size = (224, 224)
batch_size = 32

# Simple data preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
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

# Very simple model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("🚀 Training for 10 epochs...")
history = model.fit(train_gen, validation_data=val_gen, epochs=10, verbose=1)

# Save the model
os.makedirs("model", exist_ok=True)
model.save('model/fire_detector.h5')
print("✅ Simple model saved to model/fire_detector.h5")

# Test the model
print("\n🧪 Testing the simple model...")
test_cases = [
    ('dataset/fire/fire.1.png', 'Fire image'),
    ('dataset/no_fire/non_fire.1.png', 'No-fire image')
]

for file_path, label in test_cases:
    try:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        print(f'{label:15} | Prediction: {prediction:.4f}')
    except Exception as e:
        print(f'{label:15} | Error: {e}')

print("\n🎯 Simple model ready!")
print("This model should have better separation between fire and no-fire predictions.")