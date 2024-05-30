import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = r'E:\IITD_ResearchWork\Database\IITD_FER_Augmented'

# Image dimensions and other parameters
img_height, img_width = 48, 48
batch_size = 32

# Create data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# CNN Model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(20, activation='softmax')  # Assuming 20 classes
])



# Vision Transformer Model
vit_input = layers.Input(shape=(img_height, img_width, 3))
vit_patches = layers.Reshape((-1, 3))(vit_input)
vit_attention = layers.MultiHeadAttention(num_heads=8, key_dim=2)(vit_patches, vit_patches)
vit_normalized = layers.LayerNormalization(epsilon=1e-6)(vit_attention)
vit_flatten = layers.Flatten()(vit_normalized)
vit_dense1 = layers.Dense(128, activation='relu')(vit_flatten)
vit_output = layers.Dense(20, activation='softmax')(vit_dense1)

vit_model = models.Model(inputs=vit_input, outputs=vit_output)




# Compile models
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
vit_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model
cnn_history = cnn_model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Train Vision Transformer model
vit_history = vit_model.fit(train_generator, epochs=10, validation_data=validation_generator)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='CNN Training Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.title('CNN Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(vit_history.history['accuracy'], label='ViT Training Accuracy')
plt.plot(vit_history.history['val_accuracy'], label='ViT Validation Accuracy')
plt.title('ViT Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

