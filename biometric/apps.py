from django.apps import AppConfig
import os
import facenet
import tensorflow as tf
from biometric import detect_face


class BiometricConfig(AppConfig):
    name = 'biometric'
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            facenet_model_path = os.path.dirname(__file__)+"\models\\facenet\\20190624-094358"
            # Load the model
            facenet.load_model(facenet_model_path)
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    gpu_memory_fraction=0.9
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess_detc = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess_detc.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess_detc, None)