import os
import time
import pickle
from six.moves import urllib
import tensorflow as tf
import sys
import numpy as np
import tarfile


FLAGS = tf.app.flags.FLAGS
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")


cifar_python_directory = os.path.abspath("cifar-10-batches-py")


def download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,reporthook=_progress)        
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""    
    with tf.gfile.FastGFile(os.path.join(FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def unpickle(file):    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_data(filename):    
    d = unpickle(
        os.path.join(cifar_python_directory, filename)
    )
    data = d[b'data']
    labels = d[b'labels']

    #sample, x, y, color
    data = np.transpose(np.reshape(data,(-1,32,32,3), order='F'),axes=(0,2,1,3)) 
    return data, labels


def extract_features(filename):

    data,labels = load_data(filename)
    length = len(labels)

    FLAGS.model_dir = 'model/'
    download_and_extract()
    create_graph()
    with tf.Session() as sess:
        predictions_tensor = sess.graph.get_tensor_by_name('softmax:0')
        feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        predictions = np.zeros((length, 1008), dtype='float32')
        representations = np.zeros((length, 2048), dtype='float32')
        for i in range(length):
            [reps, preds] = sess.run([feature_tensor, predictions_tensor], {'DecodeJpeg:0': data[i]})
            predictions[i] = np.squeeze(preds)
            representations[i] = np.squeeze(reps)
        np.savez_compressed(filename + "_tensorflow.npz", predictions=predictions, representations=representations, labels=labels)

if __name__ == '__main__':
    extract_features('test_batch')
    extract_features('data_batch_1')
    extract_features('data_batch_2')
    extract_features('data_batch_3')
    extract_features('data_batch_4')
    extract_features('data_batch_5')
