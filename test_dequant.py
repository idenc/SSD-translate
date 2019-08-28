import tensorflow as tf
import numpy as np

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Input(shape=(256, 256, 3), dtype='float32'))
# model.add(tf.keras.layers.Conv2D(16, 3, strides=(2, 2), padding='same', bias_initializer='random_normal'))
# for i in range(8):
#     x1 = tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2])
#     model.add(x1)
#     model.add(tf.keras.layers.Conv2D(8, 1, bias_initializer='random_normal'))
#     model.add(tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2]))
#     model.add(tf.keras.layers.DepthwiseConv2D(3, padding='same', bias_initializer='random_normal'))
#     x2 = tf.keras.layers.Conv2D(16, 1, bias_initializer='random_normal')
#     model.add(x2)
#     model.add(tf.keras.layers.add([x1, x2]))

inputs = tf.keras.layers.Input(shape=(256, 256, 3), dtype='float32')
x = tf.keras.layers.Conv2D(16, 3, strides=(2, 2), padding='same', bias_initializer='random_normal')(inputs)
for i in range(8):
    x1 = tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2])(x)
    x = tf.keras.layers.Conv2D(8, 1, bias_initializer='random_normal')(x1)
    x = tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2])(x)
    x = tf.keras.layers.DepthwiseConv2D(3, padding='same', bias_initializer='random_normal')(x)
    x2 = tf.keras.layers.Conv2D(16, 1, bias_initializer='random_normal')(x)
    x = tf.keras.layers.add([x1, x2])

x1 = tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2])(x)
x = tf.keras.layers.Conv2D(16, 2, strides=(2, 2), bias_initializer='random_normal')(x1)
x = tf.keras.layers.PReLU(alpha_initializer='random_normal', shared_axes=[1, 2])(x)
x = tf.keras.layers.DepthwiseConv2D(3, padding='same', bias_initializer='random_normal')(x)
y = tf.keras.layers.Conv2D(32, 1, bias_initializer='random_normal')(x)

x = tf.keras.layers.MaxPool2D(strides=(2, 2))(x1)
paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 16]])
x = tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings))(x)
# x = tf.keras.layers.add([y, x])

model = tf.keras.models.Model(inputs=inputs, outputs=x)

model.compile(optimizer='adam', loss='categorical_crossentropy')
tf.keras.experimental.export_saved_model(model, 'prelu_conv', serving_only=True)


def convert_to_tflite(graph_file, input_arrays, output_arrays, output_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        graph_file)
    tflite_model = converter.convert()
    open(output_name, "wb").write(tflite_model)


# model = tf.keras.models.load_model('prelu.h5')
convert_to_tflite('prelu_conv', None, None, 'prelu_conv.tflite')
