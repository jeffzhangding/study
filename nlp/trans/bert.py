__author__ = 'jeff'

import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()



# from official.modeling import tf_utils
# from official import nlp
# from official.nlp import bert
#
# # Load the required submodules
# import official.nlp.optimization
# import official.nlp.bert.bert_models
# import official.nlp.bert.configs
# import official.nlp.bert.run_classifier
# import official.nlp.bert.tokenization
# import official.nlp.data.classifier_data_lib
# import official.nlp.modeling.losses
# import official.nlp.modeling.models
# import official.nlp.modeling.networks

#
# gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
# # gs_folder_bert = "https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/2"
#
# tf.io.gfile.listdir(gs_folder_bert)
#
# hub_url_bert = "https://hub.tensorflow.google.cn/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
#
#
# max_seq_length = 128  # Your choice here.
# input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                        name="input_word_ids")
# input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                    name="input_mask")
# segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                     name="segment_ids")
os.environ['TFHUB_CACHE_DIR'] = 'E:\\模型文件\\bert'
b = os.path.abspath('E:\\模型文件\\bert\\2.tar')
bert_layer = hub.KerasLayer("https://hub.tensorflow.google.cn/tensorflow/bert_zh_L-12_H-768_A-12/2",
                            trainable=True)
# tf.saved_model.load("E:\\模型文件\\bert")
# hub.load()
# bert_layer = hub.KerasLayer(b,
#                             trainable=True)
print('========')
# pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
#
# vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
# do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
# tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

