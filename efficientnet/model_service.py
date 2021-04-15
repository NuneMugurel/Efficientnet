from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import datetime
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

class EfficientNetService:
    @staticmethod
    def compile_model(model):
        nadam = optimizers.Nadam(
            learning_rate=0.01
        )
        model.compile(optimizer=nadam,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    @staticmethod
    def fit_model(model, path, batch_size, epochs, callbacks, img_size, initial_epoch=0):
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            validation_split=0.2)
        train_generator = datagen.flow_from_directory(
                            path,
                            target_size=(img_size, img_size),
                            batch_size=batch_size,
                            class_mode='categorical',
                            subset='training')
        validation_generator = datagen.flow_from_directory(
                    path,
                    target_size=(img_size, img_size),
                    batch_size=batch_size,
                    class_mode='categorical',
                    subset='validation')
        model.fit(
            train_generator,
            steps_per_epoch = train_generator.samples // batch_size,
            validation_data = validation_generator,
            validation_steps = validation_generator.samples // batch_size,
            epochs = epochs,
            initial_epoch = initial_epoch,
            callbacks = callbacks)

    @staticmethod
    def get_callbacks(log_dir = None, filepath=None):
        callbacks = []
        if log_dir is None:
            log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        callbacks.append(tensorboard_callback)

        if filepath is None:
            filepath = './MnasEfficientNetExperimental.h5'
        checkpointCallback = ModelCheckpoint(filepath=filepath,
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             )
        callbacks.append(checkpointCallback)

        def lr_scheduler(epoch, lr):
            print(f'LR:{lr * tf.math.exp(-0.2)}, Epoch:{epoch + 1}')
            return lr * tf.math.exp(-0.2)

        lrDecayCallback = LearningRateScheduler(lr_scheduler)
        callbacks.append(lrDecayCallback)

        return callbacks   