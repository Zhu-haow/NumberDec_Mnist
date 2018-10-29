import  tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
import  os
import mnist_inference

BATCH_SIZE = 100    #一个训练的批处理的训练数据个数。越小时越接近随机梯度下降；越大越接近梯度下降

Learning_Rate = 0.8 #基础学习率
Learning_Decay = 0.99 #学习的衰减率 利用指数衰减更新学习率，从而快速迭代得到参数最优解

Regularization_Rate = 0.0001 #描述模型参数复杂度的正则化项在损失函数中的系数λ

Traning_Step = 30000  #训练轮数

Moving_Average_Decay = 0.99 #滑动平均衰减率

def train(mnist):
    #设置x,y占位符,并定义初始化方法
    x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-output")

    # 计算L2正则损失函数
    regularizer = tf.contrib.layers.l2_regularizer(Regularization_Rate)

    #初始调用得到前向传播计算结果
    y = mnist_inference.inference(x,regularizer)

    #定义存储训练轮数的变量
    global_step = tf.Variable(0,trainable= False)

    #初始化滑动平均类
    mavg = tf.train.ExponentialMovingAverage(Moving_Average_Decay,global_step)

    #对所有可训练变量进行滑动平均
    vars_avg = mavg.apply(tf.trainable_variables())

    #计算损失函数 利用交叉熵+正则损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y ,labels=tf.arg_max(y_,1))
    #计算所有batch交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算模型的总损失= 交叉熵+总正则化损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #设置学习率
    learning_rate = tf.train.exponential_decay(Learning_Rate,global_step,mnist.train.num_examples / BATCH_SIZE
                                               ,Learning_Decay)
    #优化后的损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    #反向传播更新参数
    train_op = tf.group(train_step,vars_avg)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #初始化持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #validate_feed = {x: mnist.validation.images,
        #                y_: mnist.validation.labels}
        for i in range(Traning_Step):

            #if(i%1000 == 0):
                #validate_acc = sess.run(accuracy,feed_dict=)
                #print("ater %d steps ,accuracy on model is %g"%(i,validate_acc))

            # 产生下一轮训练的Batch数据
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
        saver.save(sess,os.path.join("model/","model.ckpt"),global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()