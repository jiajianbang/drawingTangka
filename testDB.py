#!/usr/bin/python
# -*- coding: UTF-8 -*-
import models.Retrieval_End as model1
import numpy as np
import math
log_path = './checkpoints2w/'
class GetImage:
    def __init__(self,imgUrl,imgLabel = None,imgList = None):
        self.imgUrl = imgUrl
        self.imgLabel = imgLabel
        self.imgList = imgList
        self.Relusult = self.getResult()
    def getResult(self):
        featuress, label = model1.evaluate_one_image(train_dir=self.imgUrl,log_path=log_path)
        imageResult = []
        with open('feature.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                featureList = line.split(',')
                if not self.imgLabel is None:
                    feature_label = featureList[1].split('_')
                    if feature_label[0] != self.imgLabel:
                        continue
                    if featureList[1] in self.imgList:
                        continue
                features = featureList[0].split(' ')
                feature_oth = np.array(features, dtype=float)
                feature_ret = np.array(featuress, dtype=float)
                sim = feature_ret.dot(feature_oth) / (
                        math.sqrt((feature_ret ** 2).sum()) * math.sqrt((feature_oth ** 2).sum()))
                tuple1 = (sim,featureList[1])
                imageResult.append(tuple1)
        max_value = 0.0;
        reture_image = ""
        for imageRe in imageResult:
            if (imageRe[0] > max_value):
                max_value = imageRe[0]
                reture_image = imageRe[1]
                label_list = imageRe[1].split('_')
                self.imgLabel = label_list[0]
        if max_value == 0.0:
            return self.imgList[0]
        return reture_image




# if __name__ == '__main__':
#     image_dir = "./tangka_jpg/"
#     log_path = './checkpoints2w/'
#     tangka_label, tangka_file, tangka_feature = model1.evaluate_one_image(train_dir=image_dir,log_path=log_path)
#     fp = open('feature.txt', 'w+')
#     for i in range(len(tangka_file)):
#         list_str = []
#         fe = np.array(tangka_feature[i],dtype=str)
#
#         list_str = fe.tolist();
#
#         # print (list_str)
#         str_fe = " ".join(list_str)
#
#         fp.write(str_fe)
#
#
#         fp.write(','+tangka_file[i]+'\n')
#
#         print (i)
#     fp.close()
#     print(tangka_file)

# import MySQLdb
#
# class ImageMatching:
#     def __init__(self, imageUrl):
#         self.imageUrl = imageUrl


# # 打开数据库连接
# db = MySQLdb.connect("localhost", "root", "123456", "feiyi_db", charset='utf8' )
#
# # 使用cursor()方法获取操作游标
# cursor = db.cursor()
#
# # 如果数据表已经存在使用 execute() 方法删除表。
# # cursor.execute("DROP TABLE IF EXISTS EMPLOYEE")
#
# # 创建数据表SQL语句
# sql = """CREATE TABLE FEATURES (
#          MD_VALUE  CHAR(100) NOT NULL,
#          RETURAL_Image  CHAR(100),
#          Rate FLOAT ,
#          RETURALED_Image CHAR(100),
#          ORIGION_IMAGE CHAR (100) )"""
#
# cursor.execute(sql)
#
# # 关闭数据库连接
# db.close()