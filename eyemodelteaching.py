import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'eye_training'
test_data_dir = 'eye_testing'

batch_size = 32
epochs = 10
image_height = 128
image_width = 128
num_classes = 2  # Два класса: открытые и закрытые глаза

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary')

# Модель нейронной сети для обнаружения глаз
eye_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Один выход для бинарной классификации
])

eye_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nLoss at epoch {epoch}: {logs["loss"]:.4f}')

eye_model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[LossCallback()])

test_loss, test_acc = eye_model.evaluate(test_generator)
print('Test accuracy:', test_acc)
eye_model.save('eye_model.h5')