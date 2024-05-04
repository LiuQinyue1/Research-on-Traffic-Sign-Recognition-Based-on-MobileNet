import tensorflow as tf
from keras.applications.mobilenet import MobileNet
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam

from dataset import data_generate


# Build model structure
def build_model(num_classes, l2_reg=0.01):
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(l2_reg))(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model


# Train model
def train_model_with_generator(model, x_train, y_train, x_val, y_val, epochs=100, batch_size=32):
    train_generator, val_generator = data_generate(x_train, y_train, x_val, y_val, batch_size=batch_size)
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=len(x_train) // batch_size,
        validation_steps=len(x_val) // batch_size,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=0.0005,  patience=10,
                                             restore_best_weights=True)
        ]
    )

    return history
