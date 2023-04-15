import os
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from sys import platform
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# Caricare il modello FaceNet
graph: tf.Graph = None
sess: tf.compat.v1.Session = None


def load_facenet_model(model_path):
    global graph
    graph = tf.Graph()
    with graph.as_default():
        with tf.io.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')


def preprocess_image(image_path: Path, image_size=(160, 160)) -> np.ndarray:
    image = Image.open(image_path)
    image = image.resize(image_size, Image.LANCZOS)
    image = np.array(image)
    image = image.astype(np.float32)
    image = (image - 127.5) / 128.0
    image = np.expand_dims(image, axis=0)
    return image


def get_embeddings(image: np.ndarray) -> np.ndarray:
    input_tensor = graph.get_tensor_by_name("input:0")
    output_tensor = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    embeddings = sess.run(output_tensor, feed_dict={input_tensor: image, phase_train_placeholder: False})
    return embeddings


def process_image(image_path: Path):
    image = preprocess_image(image_path)
    embedding = get_embeddings(image)
    return embedding, image


def main():
    #d_drive = Path('/mnt/d' if platform == 'linux' else 'D:\\')
    images_folder = Path('/Users/gabrieletassinari/Downloads/lfw-deepfunneled/')
    facenet_model_path = Path('/Users/gabrieletassinari/Downloads/20180402-114759/20180402-114759.pb')

    load_facenet_model(facenet_model_path)
    image_paths = tuple(images_folder.iterdir())

    # Processa le immagini
    global sess
    with tf.compat.v1.Session(graph=graph) as _sess:
        sess = _sess
        results = list(tqdm(map(process_image, image_paths), total=len(image_paths)))

    x_train_list, y_train_list = zip(*results)

    x_train = np.array(x_train_list).squeeze(1)
    y_train = np.array(y_train_list).squeeze(1)

    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)

    np.save('x_dataset1.npy', x_train)
    np.save('y_dataset1.npy', y_train)


if __name__ == '__main__':
    main()