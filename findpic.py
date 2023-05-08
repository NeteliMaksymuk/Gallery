import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('dataset/test3.h5')

def find_p(img):
    image = tf.keras.preprocessing.image.load_img(
        img,
        target_size=(224, 224)
    )


    x = tf.keras.preprocessing.image.img_to_array(image)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)

    max_class = np.argmax(preds, axis=1)


    print("Клас з максимальною ймовірністю: ", max_class)
    return max_class
