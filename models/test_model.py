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

    # you need to change the directories to yours.
    #    train_dir = 'F:/tangka_minyuansaomiao_change/'
    # train, train_label = input_data.get_files(train_dir)
    # image_array, img_dir = get_one_image(file_path)

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
            tangka_label = []
            tangka_feature = []
            tangka_file = []
            for file in os.listdir(train_dir):
                print(file)
                file_path = train_dir + file
                image_array = get_one_image(file_path)
                prediction= sess.run((logit, fc7), feed_dict={image: image_array})
                #获取fc7层的参数
                # print(prediction)
                pre = prediction[0]
                fc = prediction[1]



                max_index = np.argmax(pre)

                featuress = fc[0]

                tangka_feature.append(featuress)
                tangka_file.append(file)
                tangka_label.append(max_index)
            return tangka_label,tangka_file,tangka_feature



                # print(max_index)
                # try:
                #     i += 1
                #     file_list = file.split('_')
                #     allData[max_index] += 1
                #     if img_label[max_index] == file_list[0]:
                #         right_tangka.append(file)
                #         rightData[max_index] += 1
                #     else:
                #         error_tangka.append(file)
                #     if i % 100 == 0:
                #         print(i)
                #
                # except Exception as err:
                #     print(err)
                #     print('file Error!')
                #     print(file_path)
                #     pass
            # with open('test.csv','w') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerows(allData)
            #     writer.writerows(rightData)
            # try:
            #     fp = open('right_data.txt', 'w+')
            #     fw = open('error_data.txt', 'w+')
            #     for i in range(len(right_tangka)):
            #         str_data = right_tangka[i]+'\t\n'
            #         fp.write(str_data)
            #     fp.close()
            #     for j in range(len(error_tangka)):
            #         str_err = error_tangka[j] + '\t\n'
            #         fw.write(str_err)
            #     fw.close()
            # except EOFError:
            #     print('error')
            #     pass
            # print('allData')
            # print(allData)
            # print(sum(allData))
            # print('rightData')
            # print(rightData)
            # print(sum(rightData))
            # print('正确率为：')
            # sum_data = float(sum(rightData))/float(sum(allData))
            # print(sum_data)






#
#
#
# if __name__ == '__main__':
#     train_dir = 'D:/tangkaImages/tangka_jpg/'
#     log_path = './checkpoints1/'
#     evaluate_one_image(train_dir,log_path)