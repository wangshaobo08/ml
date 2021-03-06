#!/usr/bin/env python3
# coding=utf-8

import os
import tensorflow as tf

# # 必须做一步显示的初始化op
# init_op= tf.global_variables_initializer()
#
#  # 把程序的图结构写入事件文件，graph：吧置顶的图写进事件文件当中
#  filewriter = tf.summary.FileWriter("./tmp",graph=sess.graph)

# 1、 训练参数问题：trainable
# 学习率和步数的设置：

# 2、添加权重参数，损失值等在tensorboard观察的情况 1、收集变量 2、合并变量

def myregression():
    """
    自实现一个线性回归预测
    return: None
    """
    with tf.variable_scope('data'):
        # 1、准备数据，x 特征值 [100,1]  y 目标值[100]
        x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='X_data')

        # 矩阵相乘必须是二维的
        y_true = tf.matmul(x,[[0.7]]) + 0.8

    # 2、建立线性回归模型 1个特征 1个权重，1个偏秩 y = wx + b
    # 随机给一个权重和偏秩的值，让他去计算损失，然后再当前状态下优化
    # 用变量定义才能优化
    with tf.variable_scope('model'):
        weight = tf.Variable(tf.random_normal([1,1], mean=0, stddev=1.0), name='w')
        bias = tf.Variable(0.0, name='b')

        y_predict = tf.matmul(x,weight) + bias

    # 3、建立损失函数，均方误差
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4、梯度下降优化损失 learning_rate:0 ~ 1，2，4，5，6
    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 收集变量
    tf.summary.scalar("losses",loss)
    tf.summary.histogram("weights",weight)

    # 定义合并tesor的op
    merged = tf.summary.merge_all()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)


        #打印随机最先初始化的权重和偏秩
        print("随机初始化的参数权重为：%f, 偏秩为：%f" % (weight.eval(),bias.eval()))

        # 建立事件文件
        filewriter = tf.compat.v1.summary.FileWriter('./tmp/',graph=sess.graph)

        # 加载模型，覆盖模型当中随机定义的参数，从上次训练的参数结果开始
        if os.path.exists('./tmp/ckpt/checkpoint'):
            saver.restore(sess,'./tmp/ckpt/model')

        #运行优化
        for i in range(500):
            sess.run(train_op)
            # 运行合并的op
            summary = sess.run(merged)

            filewriter.add_summary(summary,i)
            if i% 50 == 0:
                print("第%d次训练参数权重为：%f, 偏秩为：%f" % (i,weight.eval(),bias.eval()))
        saver.save(sess,'./tmp/ckpt/model')


if __name__ == '__main__':
    myregression()







