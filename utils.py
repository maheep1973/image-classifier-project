import tensorflow as tf
import tensorflow_hub as hub
import json


image_size = 229
image_shape = (image_size, image_size, 4)

def get_cls_nms(json_file):
    with open(json_file, 'r') as f:
        cls_nms = json.load(f)

    cls_nms_new = dict()
    for key in cls_nms:
        cls_nms_new[str(int(key)-1)] = cls_nms[key]
    return cls_nms_new

def model_loader(model_path):
    model = tf.keras.models.model_loader(model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    print(model.summary())
    return model

def image_processor(numpy_image):
    print(numpy_image.shape)
    tens_img = tf.image.convert_image_dtype(numpy_image, dtype=tf.int16, saturate=False)
    resized_img = tf.image.resize(numpy_image,(image_size,image_size)).numpy()
    norm_img = resized_img/255
    return norm_img
