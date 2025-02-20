
import numpy as np
import os
import cv2
from imageio import imread
from skimage.transform import resize
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential,load_model
import pandas as pd
import h5py
import glob
import sys
import scipy.io
import time
from scipy import stats

def load_test_motion(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(carpeta)
    print('Read test images')

    for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
        imagenes = os.path.join(carpeta, imagen)
        img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
        img = img.transpose((-1,0,1))
        X_test.append(img)
        images_names.append(imagenes)
    return X_test, images_names


def load_test_attention(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print('Read test images')

    for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
        imagenes = os.path.join(carpeta, imagen)
        img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
        img = img.transpose((-1,0,1))
        X_test.append(img)
        images_names.append(imagenes)
    return X_test, images_names

np.set_printoptions(threshold=np.inf)
data = []
batch_size = 128
model = load_model('heartrate_model/DeepFakesON-Phys_CelebDF_V2.h5')
# model.add(lyrs.MaxPooling2D(pool_size=(pool_size, pool_size), data_format="channels_last"))
print(model.summary())

def predict(deep_path, raw_path):
  print(deep_path, raw_path)
  image_path = r"eng_heart"#asli vids image path
  carpeta_deep= os.path.join(image_path, deep_path)

  carpeta_raw= os.path.join(image_path, raw_path)

  test_data, images_names = load_test_motion(carpeta_deep)
  test_data2, images_names = load_test_attention(carpeta_raw)
    
  test_data = np.array(test_data, copy=False, dtype=np.float32)
  test_data2 = np.array(test_data2, copy=False, dtype=np.float32)

  predictions = model.predict([test_data, test_data2], batch_size=batch_size, verbose=1)
  bufsize = 1
  nombre_fichero_scores = 'deepfake_scores.txt'
  fichero_scores = open(nombre_fichero_scores,'w',buffering=bufsize)
  fichero_scores.write("img;score\n")
  for i in range(predictions.shape[0]):
      fichero_scores.write("%s" % images_names[i]) #fichero
      if float(predictions[i])<0:
        predictions[i]='0'
      elif float(predictions[i])>1:
        predictions[i]='1'
      fichero_scores.write(";%s\n" % predictions[i]) #scores predichas
  return predictions


result_path = 'eng_heart'
if all(os.path.isdir(os.path.join(result_path, f)) for f in os.listdir(result_path)):
    print("The directory contains no files to process.")
else:
    for f in os.listdir(result_path):
        # if not "euler.avi" in f or "38.mp4euler.avi" in f:
        #     continue
        if os.path.isdir(os.path.join(result_path, f)):
            continue
        raw_path = 'RawFrames/' + str(f)
        deep_path = 'DeepFrames/' + str(f)
        predictions = predict(deep_path, raw_path)

        final_score = np.mean(predictions)  # You can change this aggregation method as needed

        # Write final prediction to a file
        nombre_fichero_scores = 'results/heartrate_scores.txt'
        with open(nombre_fichero_scores, 'w') as fichero_scores:
            final_score2=1-final_score
            fichero_scores.write(f"{final_score2}\n")
