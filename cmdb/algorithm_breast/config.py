# -*- coding: utf-8 -*-
import tensorflow as tf

path = "./media/resource_breastTxt"

tf.app.flags.DEFINE_string("src_file", path+'/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", path+'/target.txt', "labels.")

# 希望做命名识别的数据
tf.app.flags.DEFINE_string("pred_file", path+'/predict_seg.txt', "test data.")
tf.app.flags.DEFINE_string("src_vocab_file", path+'/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", path+'/target_vocab.txt', "targets.")
tf.app.flags.DEFINE_string("word_embedding_file", path+'/source_vec.vector', "extra word embeddings.")
tf.app.flags.DEFINE_string("model_path", path+'/model/', "model save path")

# 这里默认词向量的维度是100。
tf.app.flags.DEFINE_integer("embeddings_size", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 130, "max sequence length.")

tf.app.flags.DEFINE_integer("batch_size", 32, "batch size.")
tf.app.flags.DEFINE_integer("epoch", 1000, "epoch.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

tf.app.flags.DEFINE_string("action", 'predict', "train | predict")
FLAGS = tf.app.flags.FLAGS
