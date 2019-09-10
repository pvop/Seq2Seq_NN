import tensorflow as tf
from data_load import data
import os
os.environ['CUDA_VISIBLE_DEVICES']=str(1)
from model import Model
from hyperparameters import Hyperparameters
hp = Hyperparameters()

data = data()
model = Model(mode="train")
session_config = tf.ConfigProto(
            log_device_placement=False,
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            allow_soft_placement=True)
session_config.gpu_options.allow_growth = True
session_config.gpu_options.allocator_type = 'BFC'
with tf.Session(config=session_config) as sess:
    saver = tf.train.Saver()
    saver.restore(sess, "../model/result.ckpt")
    for i in range(50):
        model.train(sess,data.source_train_datas,data.target_train_datas,data.target_train_length,hp.batch_size)
        saver.save(sess,"../model/result.ckpt")
