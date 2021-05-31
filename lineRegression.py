#!/usr/bin/env python3
# coding=utf-8

import tensorflow as tf

def myregression():
    """
    自实现一个线性回归预测
    return: None
    """
    # 1、准备数据，x 特征值 [100,1]  y 目标值[100]
    x = tf.random_normal([100,1],mean=1.75,stddev=0.5,name='X_data')

    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x,[[0.7]]) + 0.8

    # 2、建立线性回归模型 1个特征 1个权重，1个偏秩 y = wx + b
    # 随机给一个权重和偏秩的值，让他去计算损失，然后再当前状态下优化
    # 用变量定义才能优化
    weight = tf.Variable(tf.random_normal([1,1], mean=0, stddev=1.0), name='w')
    bias = tf.Variable(0.0, name='b')

    y_predict = tf.matmul(x,weight) + bias

    # 3、建立损失函数，均方误差
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 4、梯度下降优化损失 learning_rate:0 ~ 1，2，4，5，6
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 通过会话运行程序
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        #打印随机最先初始化的权重和偏秩
        print("随机初始化的参数权重为：%f, 偏秩为：%f" % (weight.eval(),bias.eval()))

        #运行优化
        for i in range(2000):
            sess.run(train_op)
            if i%50 == 0:
                print("第%d次训练参数权重为：%f, 偏秩为：%f" % (i,weight.eval(),bias.eval()))


if __name__ == '__main__':
    myregression()







