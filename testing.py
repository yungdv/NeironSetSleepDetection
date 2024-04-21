import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_dir = 'training'
test_data_dir = 'testing'

batch_size = 32
epochs = 10
image_height = 128
image_width = 128
num_classes = 3

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical')

model = tf.keras.models.load_model('my_model.h5')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class LossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f'\nLoss at epoch {epoch}: {logs["loss"]:.4f}')

history = model.fit(train_generator, epochs=epochs, validation_data=test_generator, callbacks=[LossCallback()])

test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)