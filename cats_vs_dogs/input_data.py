#encoding=utf-8
__auther__ = "tcd1112"
import os
import time
import datetime
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt


#datetime.datetime.now().strftime('_%Y_%m_%d_%H_%M')
def timmer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        stop_time = time.time()
        print ("函数运行时间为%s" % (stop_time - start_time))

    return wrapper

def get_files(train_dir):
    cats=[]
    label_cats=[]
    dogs=[]
    label_dogs=[]
    for file in os.listdir(train_dir):
        name=file.split(".")
        if name[0]=="cat":
            cats.append(os.path.join(train_dir,file))
            label_cats.append(0)
        else:
            dogs.append(os.path.join(train_dir,file))
            label_dogs.append(1)
    print ("there are %d cats\nthere are %d dogs"%(len(cats),len(dogs)))

    image_list=np.hstack((cats,dogs))
    label_list=np.hstack((label_cats,label_dogs))
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list=list(temp[:,0])
    label_list=list(temp[:,1])
    label_list=[int(i) for i in label_list]
    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label=input_queue[1]
    image_contents=tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    """
    #可以加 数据的特征工程加强
    """
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image=tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,
                                            num_threads=8,
                                            capacity=capacity)
    #之前打乱过数据
    #image_batch,label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,
    #                                         num_threads=64,
    #                                         capacity=capacity)

    label_batch =tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch #4D，1D



if __name__ == '__main__':
    # test_show_image()

    batch_size = 2
    capacity = 256
    img_w = 208
    img_h = 208
    img_width = 208
    img_height = 208

    train_dir = os.getcwd() + "/data/train"
    image_list, label_list=get_files(train_dir)
    image_batch, label_batch = get_batch(image_list, label_list, img_w, img_h, batch_size, capacity)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threeds = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop() and i < 1:
                img, label = sess.run([image_batch, label_batch])
                for j in np.arange(batch_size):
                    print("label:%d" % label[j])
                    plt.imshow(img[j, :, :, :])
                    plt.show()
                i += 1
        except tf.errors.OutOfRangeError:
            print("Done")
        finally:
            coord.request_stop()
        coord.join(threeds)