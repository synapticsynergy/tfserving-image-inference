import os
import tensorflow as tf
from build_model import build_COVIDNet


def convert_keras_model(model_path=None, saved_model_path=None, input_tensor="input", input_width=224, input_height=224):
    tf.keras.backend.clear_session()
    keras_model = build_COVIDNet(checkpoint=model_path)

    def serving_input_receiver_fn():
        def decode_and_resize(image_str_tensor):
            """ Decodes a single image string, preprocesses/resizes it,
            and returns a reshaped float32 tensor.
            """
            image = tf.image.decode_image(image_str_tensor, channels=3, dtype=tf.uint8)
            image = tf.cast(image, dtype=tf.float32)
            image = tf.image.resize_image_with_pad(image, input_width, input_height)
            image = tf.reshape(image, [input_width, input_height, 3])
            return image
        input_ph = tf.placeholder(tf.string, shape=[None], name="image_binary")
        images_tensor = tf.map_fn(decode_and_resize, input_ph, back_prop=False, dtype=tf.float32)
        images_tensor = tf.math.divide(images_tensor, 255.)
        return tf.estimator.export.ServingInputReceiver({input_tensor: images_tensor}, {"image_bytes": input_ph})
    
    def dummy_loss(y_true, y_pred):
        if y_pred.dtype == tf.string:
            eq = tf.not_equal(y_true, y_pred)
            return tf.reduce_sum(tf.cast(eq, tf.float32))
        else:
            return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
    
    tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference

    keras_model.compile(optimizer='sgd', loss=dummy_loss)
    estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_model)

    # patch to fix: https://github.com/tensorflow/tensorflow/issues/25772
    # Only applies to tensorflow version less than 1.14.0
    # tf_version = tf.__version__.split('.')
    # if tf_version[0] == '1' and int(tf_version[1]) < 14:
    #     estimator._model_dir = os.path.join(estimator._model_dir, 'keras')

    estimator.export_saved_model(saved_model_path, serving_input_receiver_fn=serving_input_receiver_fn)
    print('estimator converted to saved model successfully')

if __name__ == "__main__":
    convert_keras_model(
        model_path="./model/cp-08.hdf5",
        saved_model_path="./saved_model/",
        input_tensor="input_1"
    )