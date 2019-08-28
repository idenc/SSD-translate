import tensorflow as tf
import numpy as np

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Input(shape=(128, 128, 16), dtype='float32'))
# model.add(tf.keras.layers.PReLU(alpha_initializer='random_normal', name='output', input_shape=(128, 128, 3)))


# model.compile(optimizer='adam', loss='categorical_crossentropy')
# tf.keras.experimental.export_saved_model(model, 'prelu', serving_only=True)

def convert_to_tflite(graph_file, input_arrays, output_arrays, output_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(
        graph_file)
    tflite_model = converter.convert()
    open(output_name, "wb").write(tflite_model)

# model = tf.keras.models.load_model('prelu.h5')
convert_to_tflite('prelu', None, None, 'prelu.tflite')
