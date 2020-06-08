import tensorflow_hub as hub
import tensorflow as tf
from PIL import Image
import argparse
import numpy as np
from utils import process_image, get_class_names, load_model
import json

def predict(pic_pth, mod_pth, top_k, all_class_names):
    top_k = int(top_k)
    print(top_k, type(top_k))
    model = load_model(mod_pth)

    image = Image.open(pic_pth)
    tst_img = np.asarray(image)
    processed_tst_img = process_image(tst_img)
    probab_predict = model.predict(np.expand_dims(processed_tst_img,axis=0))
    probab_predict = probab_predict[0].tolist()
    top_predict_class_id = model.predict_classes(np.expand_dims(processed_tst_img,axis=0))
    top_pred_class_prob = probab_predict[top_predict_class_id[0]]
    pred_class = all_class_names[str(top_predict_class_id[0])]
    print("\n\nMost likely class image and it's probability :\n","class_id :",top_predict_class_id, "class_name :", pred_class, "; class_probability :",top_pred_class_prob)
    values, indices= tf.math.top_k(probab_predict, k=top_k)
    topk_probs = values.numpy().tolist()#[0]
    classes_topk = indices.numpy().tolist()#[0]
    print("top k probs:",topk_probs)
    print("top k classes:",classes_topk)
    class_labels = [all_class_names[str(i)] for i in classes_topk]
    print('top k class labels:',class_labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Description of my parser")
    parser.add_argument("pic_pth",help="Path of image", default="")
    parser.add_argument("saved_model",help="path of model", default="")
    parser.add_argument("--top_k", help="Fetching top k predictions", required = False, default = 3)
    parser.add_argument("--category_names", help="Class map json file", required = False, default = "label_map.json")
    args = parser.parse_args()

    all_class_names = get_class_names(args.category_names)

    predict(args.pic_pth, args.saved_model, args.top_k, all_class_names)
