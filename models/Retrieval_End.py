# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from .import model
import cv2

# import matplotlib.pyplot as plt
allData = [0 for i in range(15)]
rightData = [0 for i in range(15)]
img_label = ['O', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
def get_one_image(train):
    size = (227, 227)
    try:
        image = cv2.imread(train)
        image = image[0:600, 74:600]
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # image = Image.open(train)
        # image = image.resize([227, 227])
        image = np.array(image)
    except EOFError:
        print (train)
        print('error')
        pass

    # image = tf.cast(image, tf.float32)
    # image = tf.image.per_image_standardization(image)
    # image = tf.reshape(image, [1, 256, 256, 3])
    return image


def evaluate_one_image(train_dir, log_path):
    '''Test one image against the saved models and parameters
    '''
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 15
        image = tf.placeholder(shape=[227,227,3], dtype=tf.float32)
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image_array = tf.reshape(image, [1, 227, 227, 3])
        logit, fc7 = model.inference(image_array, BATCH_SIZE, N_CLASSES)

        # fc8 = model.fc8
        logit = tf.nn.softmax(logit)

        # x = tf.placeholder(tf.float32, shape=[208, 208, 3])

        # you need to change the directories to yours.


        saver = tf.train.Saver()

        with tf.Session() as sess:

            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_path)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            i = 0
            image_array = get_one_image(train_dir)
            prediction= sess.run((logit, fc7), feed_dict={image: image_array})
            #获取fc7层的参数
            # print(prediction)
            pre = prediction[0]
            fc = prediction[1]
            label = np.argmax(pre)

            featuress = fc[0]

            return featuress, label


