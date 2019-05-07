# -*- coding: utf-8 -*-
import tensorflow as tf

modelPath = "./media/models/breastModels"
resourcePath = "./media/resources/breastResources"
outputPath = "./media/output/breastOutput"

tf.app.flags.DEFINE_string("src_file", resourcePath + '/source.txt', "Training data.")
tf.app.flags.DEFINE_string("tgt_file", resourcePath + '/target.txt', "labels.")

# 希望做命名识别的数据
tf.app.flags.DEFINE_string("pred_file", outputPath + '/predict_seg.txt', "test data.")
tf.app.flags.DEFINE_string("src_vocab_file", resourcePath + '/source_vocab.txt', "source vocabulary.")
tf.app.flags.DEFINE_string("tgt_vocab_file", resourcePath + '/target_vocab.txt', "targets.")
tf.app.flags.DEFINE_string("word_embedding_file", resourcePath + '/source_vec.vector', "extra word embeddings.")
tf.app.flags.DEFINE_string("model_path", modelPath, "model save path")

# 这里默认词向量的维度是100。
tf.app.flags.DEFINE_integer("embeddings_size", 100, "Size of word embedding.")
tf.app.flags.DEFINE_integer("max_sequence", 130, "max sequence length.")
tf.app.flags.DEFINE_float("dropout", 0.6, "drop out")

FLAGS = tf.app.flags.FLAGS
