import  tensorflow as tf

#定义数据集相关常数
INPUT_NODE = 784  #输入层的节点数，一个手写字符对应的像素矩阵大小
OUTPUT_NODE = 10  #输出层的节点数。因为是分类问题，所以是十个数字输出节点

#配置神经网络的参数
LAYER1_NODE = 500     #隐藏层的节点数，只用一个有500个节点的隐藏层。


def get_weights_variable(shape,regularizer):
    weights = tf.get_variable("weights",shape,
                              initializer=tf.truncated_normal_initializer(stddev=0.1))


    if regularizer != None :
        tf.add_to_collection('losses',regularizer(weights))

    return  weights

def inference(input_sensor,regularizer):
    with tf.variable_scope('layer1'):
        weights = get_weights_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_sensor,weights)+biases)
    with tf.variable_scope('layer2'):
        weights = get_weights_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2

        # #训练结束后，检测模型正确率
        # test_acc = sess.run(accuracy,feed_dict=test_feed)
        # print("after training ,test acc is  %g"%test_acc)
