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

