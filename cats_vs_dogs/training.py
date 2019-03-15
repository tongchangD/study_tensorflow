#encoding=utf-8
__auther__ = "tcd1112"
import time
import datetime


#datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        stop_time = time.time()
        print ("函数运行时间为%s" % (stop_time - start_time))

    return wrapper

import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import input_data
import model
import random

from PIL import Image
import  matplotlib as plt

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEP = 15000
learning_rate=0.0001

def run_training():
    train_dir="/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/data/train"
    logs_train_dir="/home/tcd/PycharmProject/Study_tensorflow/cats_vs_dogs/logs"
    train,train_label=input_data.get_files(train_dir)
    train_batch,train_label_batch=input_data.get_batch(train,
                                                       train_label,
                                                       IMG_W,
                                                       IMG_H,
                                                       BATCH_SIZE,
                                                       CAPACITY)
    train_logits=model.inference(train_batch,BATCH_SIZE,N_CLASSES)
    train_loss= model.losses(train_logits,train_label_batch)
    train_op=model.training(train_loss,learning_rate)
    train_acc= model.evaluation(train_logits,train_label_batch)
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(logs_train_dir,sess.graph)
        saver= tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break
                _, tra_loss, tra_acc = sess.run([train_op,train_loss,train_acc])
                if step%100==0:
                    print ('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    summary_str=sess.run(summary_op)
                    train_writer.add_summary(summary_str,step)
                if step%500==0 or (step+1)==MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir,"model.ckpt")
                    saver.save(sess,checkpoint_path,global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
    sess.close()
#train

run_training()



