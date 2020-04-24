__author__ = 'jeff'

import os
import tensorflow as tf
import time
import numpy as np
from data_input import get_data_bow
from config import Config
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Dssm(object):

    def __init__(self, name='jeff_dssm', **kwargs):
        """"""
        super(Dssm, self).__init__()
        self.name = name
        self.graph = tf.Graph()

        self.embeding_lenth = 500  # embedding 的长度
        self.dense_lenth = 150  # 全连接的长度

        self.learning_rate = 0.8
        self.train_step = None
        self.batch = 128
        self.num_epoch = 50
        self.train_epoch_steps = 100000
        self.keep_prob = 0.85
        self.sentence_lenth = 15

        self.file_train = './data/oppo_round1_train_20180929.txt'
        self.file_vali = './data/oppo_round1_vali_20180929.txt'
        self.losses = None
        self.vocab = self.load_vocab('./data/vocab.txt')

        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8

    def model(self):
        """"""
        with self.graph.as_default():
            self.query = tf.placeholder(dtype=tf.int32, shape=(None, self.sentence_lenth))
            self.doc = tf.placeholder(dtype=tf.int32, shape=(None, self.sentence_lenth))
            self.doc_lable = tf.placeholder(dtype=tf.float32, shape=(None, ))
            keep_prob = tf.constant(self.keep_prob)
            self.embedings = tf.Variable(tf.random_normal(shape=(len(self.vocab), self.embeding_lenth)))
            self.input_1 = tf.nn.embedding_lookup(self.embedings, self.query)
            self.input_2 = tf.nn.embedding_lookup(self.embedings, self.doc)
            for i in range(3):
                self.input_1, self.input_2 = self.add_layer(self.input_1, self.input_2)
            tf.nn.dropout(self.input_1, keep_prob=keep_prob)
            tf.nn.dropout(self.input_2, keep_prob=keep_prob)
            self.cos = self.cosine(self.input_1, self.input_2)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.doc_lable, logits=self.cos)
            self.losses = tf.reduce_sum(cross_entropy)
            self.train_step = tf.train.AdamOptimizer().minimize(self.losses)
            self.predict_y = tf.nn.sigmoid(self.cos)

    def add_layer(self, input1, input2):
        """"""
        out1 = tf.layers.dense(input1, self.dense_lenth, activation=tf.tanh)
        out2 = tf.layers.dense(input2, self.dense_lenth, activation=tf.tanh)
        return out1, out2

    def cosine(self, v1, v2):
        """"""
        v1 = tf.reduce_mean(v1, 1)
        v2 = tf.reduce_mean(v2, 1)
        self.v1, self.v2 = v1, v2
        pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(v1), 1))
        pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(v2), 1))
        pooled_mul_12 = tf.reduce_sum(tf.multiply(v1, v2), 1)
        cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
        self.cos_scores = cos_scores
        cos_sim_prob = tf.clip_by_value(cos_scores, -1.0, 1.0)
        # res = tf.transpose(cos_sim_prob)
        self.cos_sim_prob = cos_sim_prob
        return cos_sim_prob

    @staticmethod
    def load_vocab(file_path):
        word_dict = {}
        with open(file_path, encoding='utf8') as f:
            for idx, word in enumerate(f.readlines()):
                word = word.strip()
                word_dict[word] = idx
        return word_dict

    def convert_seq2bow(self, query):
        res = []
        l = len(query)
        for i in range(self.sentence_lenth):
            if i < l:
                idx = self.vocab.get(query[i]) or self.vocab['[UNK]']
                res.append(idx)
            else:
                res.append(self.vocab['[UNK]'])
        return np.array(res)

    def get_data(self, file_path, batch):
        """"""
        prefix_batch, title_batch, label_batch = [], [], []
        with open(file_path, encoding='utf8') as f:
            while True:
                line = f.readline()
                if len(prefix_batch) >= batch:
                    yield np.array(prefix_batch), np.array(title_batch), np.array(label_batch)
                    prefix_batch, title_batch, label_batch = [], [], []
                if line == '':
                    raise StopIteration()
                spline = line.strip().split('\t')
                if len(spline) < 4:
                    continue
                prefix, _, title, tag, label = spline

                prefix_batch.append(self.convert_seq2bow(prefix))
                title_batch.append(self.convert_seq2bow(title))
                label_batch.append(int(label))

    def get_data_train(self):
        """
        gen datasets, convert word into word ids.
        :param file_path:
        :return: [[query, prefix, label]], shape = [n, 3]
        """
        return self.get_data(self.file_train, self.batch)

    def get_data_vaild(self):
        """"""
        return self.get_data(self.file_vali, 30)

    def vaild(self, sess):
        """"""
        v_data = self.get_data_vaild()
        loss = 0
        i = 0
        acc = 0
        while True:
            i += 1
            try:
                query, title, label = next(v_data)
            except StopIteration:
                break
            feed_dt = {
                self.query: query, self.doc: title, self.doc_lable: label,
            }
            _loss, predict_y, cos, v1 = sess.run((self.losses, self.predict_y, self.cos_scores, self.v1), feed_dict=feed_dt)
            loss += _loss
            # print('==== %s' % predict_y)
            predict_y = np.array(predict_y)
            y = np.zeros_like(predict_y)
            x = np.ones_like(predict_y)
            pre_y = np.where(predict_y >= 0.5, x, y)
            acc += np.sum(pre_y)

        print('average vaild loss: %s , accracy: %s' % (str(loss / i), acc / i))

    def prdict(self):
        """"""

    def train(self):
        """"""
        self.model()
        with tf.Session(graph=self.graph, config=self.config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for epoch in range(self.num_epoch):
                data_train = self.get_data_train()
                train_loss = None
                for batch_id in range(self.train_epoch_steps):
                    try:
                        query, title, label = next(data_train)
                    except StopIteration:
                        break
                    feed_dt = {
                        self.query: query, self.doc: title, self.doc_lable: label,
                    }
                    _, train_loss, cos,  v1, cos_scores = sess.run((self.train_step, self.losses,
                                                                    self.cos, self.v1, self.cos_scores),
                                                       feed_dict=feed_dt)
                    # print('===%s' % v1)
                end = time.time()
                print("\nEpoch #%d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                      (epoch, train_loss, end - start))

                self.vaild(sess)
            saver = tf.train.Saver()
            save_path = saver.save(sess, "model/model_1.ckpt")
            print("Model saved in file: ", save_path)


if __name__ == '__main__':
    Dssm().train()
