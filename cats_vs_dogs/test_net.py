#encoding=utf-8
__auther__ = "tcd1112"
import time
import datetime
import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import input_data
import model
import random

from PIL import Image
import matplotlib.pyplot as plt

#datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        stop_time = time.time()
        print ("函数运行时间为%s" % (stop_time - start_time))

    return wrapper

def get_one_image(train_dir):
    files = os.listdir((train_dir))
    n=len(files)

    ind = np.random.randint(0,n)
    img_dir = os.path.join(train_dir,files[ind])

    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([208,208])
    image = np.array(image)
    return image

def evaluate_one_image():
    train_dir = "/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/data/test"
    image_array=get_one_image(train_dir)
    with tf.Graph().as_default():
        BATCH_SIZE=1
        N_CLASSES=2

        image = tf.cast(image_array,tf.float32)
        image=tf.reshape(image,[1,208,208,3])
        logit = model.inference(image,BATCH_SIZE,N_CLASSES)

        logit = tf.nn.softmax(logit)
        x=tf.placeholder(tf.float32,shape=[208,208,3])

        logs_train_dir = "/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/logs"

        saver = tf.train.Saver()
        with tf.Session() as sess:
            print ("Reading checkpoints...")

            ckpt=tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                saver.restore(sess,ckpt.model_checkpoint_path)
                print("loading sueccess ,globel_step is %s "%global_step)

            else:
                print("NO checkpoint file found")

            prediction = sess.run(logit,feed_dict={x:image_array})

            max_index = np.argmax(prediction)

            if max_index ==0:
                print ("this is s cat with possibility %.6f"%prediction[:,0])
            else:
                print ("this is s cat with possibility %.6f"%prediction[:,1])

def get_more_image(train_dir):
    files = os.listdir((train_dir))
    image_list=[]
    for i in files:
        img_dir = os.path.join(train_dir,i)
        image=Image.open(img_dir)
        image = image.resize([208, 208])
        image = np.array(image)

        image_list.append(image)
    return image_list
def evaluate_more_image():
    train_dir = "/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/data/test"
    image_list =get_more_image(train_dir)

    # image_array=get_one_image(train_dir)

    with tf.Graph().as_default():
        BATCH_SIZE=1
        N_CLASSES=2
        for i in image_list:
            image = tf.cast(i,tf.float32)
            image=tf.reshape(image,[1,208,208,3])
            logit = model.inference(image,BATCH_SIZE,N_CLASSES)
            logit = tf.nn.softmax(logit)

            x=tf.placeholder(tf.float32,shape=[208,208,3])
            logs_train_dir = "/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/logs"
            saver = tf.train.Saver()
            with tf.Session() as sess:
                print ("Reading checkpoints...")
                ckpt=tf.train.get_checkpoint_state(logs_train_dir)
                print("ckpt",ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    global_step=ckpt.model_checkpoint_path.split("/")[-1].split("-")[-1]
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    print("loading sueccess ,globel_step is %s "%global_step)
                else:
                    print("NO checkpoint file found")
                prediction = sess.run(logit,feed_dict={x:image_list})
                max_index = np.argmax(prediction)
                if max_index ==0:
                    print ("this is s cat with possibility %.6f"%prediction[:,0])
                else:
                    print ("this is s cat with possibility %.6f"%prediction[:,1])
# test
if __name__ == '__main__':
    evaluate_one_image()
    # evaluate_more_image()

