# -*- coding: utf-8 -*-

#helper functions to load data from celeb folder

import PIL.ImageOps
from PIL import Image
from sklearn import preprocessing
import numpy as np
import keras
import random
import dataloader
from tqdm import tqdm
import os 
import tensorflow as tf
import csv
import pandas as pd

bs = 64

train_generator, train_male_generator, train_female_generator, \
val_generator, val_male_generator, val_female_generator, \
male_generator, female_generator = dataloader.loader(bs)

data_size = len(train_generator)
model_name_list = ['vgg16_128_2048_no_top', 'vgg19_128_2048_no_top']
model_path = 'image_model/save_model_ICML2020/'


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path = 'ICML_data/'

for model_name in model_name_list:
    vgg = keras.models.load_model(model_path + model_name + '.hdf5')
    
    with open(path + model_name + '_train_data.txt', 'w') as f:
        for i in tqdm(range(data_size)):
            image_data, (label, g_label) = train_generator.next()
            feature = vgg.predict(image_data)

            for k in range(feature.shape[0]):
                for j in range(feature.shape[-1]):
                    f.write("{0} ".format(feature[k, j]))
                f.write("{0} {1} \n".format(label[k], g_label[k]))


    train_label = []
    train_data = []
    with open(path + model_name + '_train_data.txt', 'r') as File:
        infoFile = File.readlines() #Reading all the lines from File
        for line in infoFile: #Reading line-by-line
            words = line.split('\n')[0].split(' ') #Splitting lines in words using space character as separator
            words1 = [float(i) for i in words[0:2048]]
            words1.append(float(words[2049])) #g_label
            words1.append(float(words[2048])) #label
            train_data.append(words1)
            #train_label.append(int(words[100]))

    csvfile = open(path + model_name + '_train_data.csv', 'w', newline = "")

    csvwriter = csv.writer(csvfile)
    for row in train_data:
        csvwriter.writerow(row)

    csvfile.close()


#     #############VALID##############               


#     with open(path + model_name + '_valid_data.txt', 'w') as f:
#         for i in tqdm(range(data_size)):
#             image_data, (label, g_label) = val_generator.next()
#             feature = vgg.predict(image_data)

#             for k in range(feature.shape[0]):
#                 for j in range(feature.shape[-1]):
#                     f.write("{0} ".format(feature[k, j]))
#                 f.write("{0} {1} \n".format(label[k], g_label[k]))


#     valid_label = []
#     valid_data = []
#     with open(path + model_name + '_valid_data.txt', 'r') as File:
#         infoFile = File.readlines() #Reading all the lines from File
#         for line in infoFile: #Reading line-by-line
#             words = line.split('\n')[0].split(' ') #Splitting lines in words using space character as separator
#             words1 = [float(i) for i in words[0:1000]]
#             words1.append(float(words[1001])) #g_label
#             words1.append(float(words[1000])) #label
#             valid_data.append(words1)
#             #train_label.append(int(words[100]))

#     csvfile = open(path + model_name + '_valid_data.csv', 'w', newline = "")

#     csvwriter = csv.writer(csvfile)
#     for row in valid_data:
#         csvwriter.writerow(row)

#     csvfile.close()



#     #############TEST##############                     



#     with open(path + model_name + '_test_male_data.txt', 'w') as f:
#         for i in tqdm(range(data_size)):
#             image_data, (label, g_label) = male_generator.next()
#             feature = vgg.predict(image_data)

#             for k in range(feature.shape[0]):
#                 for j in range(feature.shape[-1]):
#                     f.write("{0} ".format(feature[k, j]))
#                 f.write("{0} {1} \n".format(label[k], g_label[k]))

#     test_male_label = []
#     test_male_data = []
#     with open(path + model_name + '_test_male_data.txt', 'r') as File:
#         infoFile = File.readlines() #Reading all the lines from File
#         for line in infoFile: #Reading line-by-line
#             words = line.split('\n')[0].split(' ') #Splitting lines in words using space character as separator
#             words1 = [float(i) for i in words[0:1000]]
#             words1.append(float(words[1001])) #g_label
#             words1.append(float(words[1000])) #label
#             test_male_data.append(words1)
#             #train_label.append(int(words[100]))

#     csvfile = open(path + model_name + '_test_male_data.csv', 'w', newline = "")

#     csvwriter = csv.writer(csvfile)
#     for row in test_male_data:
#         csvwriter.writerow(row)

#     csvfile.close()


#     with open(path + model_name + '_test_female_data.txt', 'w') as f:
#         for i in tqdm(range(data_size)):
#             image_data, (label, g_label) = female_generator.next()
#             feature = vgg.predict(image_data)

#             for k in range(feature.shape[0]):
#                 for j in range(feature.shape[-1]):
#                     f.write("{0} ".format(feature[k, j]))
#                 f.write("{0} {1} \n".format(label[k], g_label[k]))


#     test_female_label = []
#     test_female_data = []
#     with open(path + model_name + '_test_female_data.txt', 'r') as File:
#         infoFile = File.readlines() #Reading all the lines from File
#         for line in infoFile: #Reading line-by-line
#             words = line.split('\n')[0].split(' ') #Splitting lines in words using space character as separator
#             words1 = [float(i) for i in words[0:1000]]
#             words1.append(float(words[1001])) #g_label
#             words1.append(float(words[1000])) #label
#             test_female_data.append(words1)
#             #train_label.append(int(words[100]))

#     csvfile = open(path + model_name + '_test_female_data.csv', 'w', newline = "")

#     csvwriter = csv.writer(csvfile)
#     for row in test_female_data:
#         csvwriter.writerow(row)

#     csvfile.close()

               
#
            
#vgg = keras.models.load_model('image_model/vgg19_128_1000_no_top.hdf5')

#
#
#with open('vgg19_128_to_1000_train_data.txt', 'w') as f:
#    for i in tqdm(range(data_size)):
#    #for i in range(5):
#        image_data, (label, g_label) = train_generator.next()
#        feature = vgg.predict(image_data)
#        #print(i)
#        #feature = np.hstack((feature, label.reshape(-1,1), g_label.reshape(-1,1)))
#        
#    
#        #for item in val_data:
#        for k in range(feature.shape[0]):
#            for j in range(feature.shape[-1]):
#                f.write("{0} ".format(feature[k, j]))
#            f.write("{0} {1} \n".format(label[k], g_label[k]))
#            
#
#vgg = keras.models.load_model('image_model/res50_128_1000_no_top.hdf5')
#
#
#with open('res50_128_to_1000_train_data.txt', 'w') as f:
#    for i in tqdm(range(data_size)):
#    #for i in range(5):
#        image_data, (label, g_label) = train_generator.next()
#        feature = vgg.predict(image_data)
#        #print(i)
#        #feature = np.hstack((feature, label.reshape(-1,1), g_label.reshape(-1,1)))
#        
#    
#        #for item in val_data:
#        for k in range(feature.shape[0]):
#            for j in range(feature.shape[-1]):
#                f.write("{0} ".format(feature[k, j]))
#            f.write("{0} {1} \n".format(label[k], g_label[k]))

'''    

            
with open('eval_feature.txt', 'w') as f:
    for i in range(len(train_generator)):
    #for i in range(5):
        image_data, (label, g_label) = val_generator.next()
        feature = vgg.predict(image_data)
        #print(i)
        #feature = np.hstack((feature, label.reshape(-1,1), g_label.reshape(-1,1)))
        
    
        #for item in val_data:
        for k in range(feature.shape[0]):
            for j in range(feature.shape[-1]):
                f.write("{0} ".format(feature[k, j]))
            f.write("{0} {1} \n".format(label[k], g_label[k]))



    with open('t.txt', 'w') as f:
        #for item in train_data:       
        f.write(" \n".join(map(str, np.hstack((feature, label.reshape(-1,1))))))
        
    

with open('val_data.txt', 'w') as f:
    for item in val_data:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))

with open('train_data_male.txt', 'w') as f:
    for item in train_data_male:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
        
with open('train_data_female.txt', 'w') as f:
    for item in train_data_female:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
        
with open('test_data.txt', 'w') as f:
    for item in test_data:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
        
with open('val_data_male.txt', 'w') as f:
    for item in val_data_male:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
                
with open('val_data_female.txt', 'w') as f:
    for item in val_data_female:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
        
        
with open('test_data_male.txt', 'w') as f:
    for item in test_data_male:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))
                
with open('test_data_female.txt', 'w') as f:
    for item in test_data_female:
        f.write("{0} {1} {2}\n".format(imagepath+item[0], item[1], item[2]))

'''