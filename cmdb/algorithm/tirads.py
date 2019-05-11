import tensorflow as tf
from cmdb.algorithm.alexnet import AlexNet
from datetime import datetime


def ret_label(img_path):
    img_string = tf.read_file(img_path)
    img_decoded = tf.image.decode_jpeg(img_string, channels=3)
    # img_decoded = tf.image.decode_and_crop_jpeg(img_string,[0,100,360,360],channels=3)
    img_resized = tf.image.resize_images(img_decoded, [227, 227])
    img_resized_float = img_resized / tf.constant(255.0)
    img_resized_float_reshape = tf.reshape(img_resized_float, [1, 227, 227, 3])

    batch_size = 1
    num_classes = 4

    checkpoint_path = './media/models/thyroidModels/model_epoch7.ckpt'

    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)

    model = AlexNet(x, keep_prob, num_classes, False)

    score = model.fc8

    with tf.name_scope("accuracy"):
        pred_label = tf.argmax(score, 1)

    # 准备加载训练好的参数
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 加载参数
        saver.restore(sess, checkpoint_path)
        print("{} Start testing...".format(datetime.now()))

        img_data = sess.run(img_resized_float_reshape)

        pre_label = sess.run(pred_label, feed_dict={x: img_data, keep_prob: 1.0})
    # total_accuracy += tmp_accuracy

    return int(pre_label + 2)
