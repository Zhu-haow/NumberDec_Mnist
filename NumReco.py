import tensorflow as tf
import mnist_inference
import mnist_train
import time
import cv2

def imageprepare(file_name):

    im = cv2.imread(file_name,0)
    print(file_name)
    pixels = []
    h, w = im.shape
    #normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    for i in range(h):
        for j in range(w):
            pixels.append((255-im[i, j])*1.0/255.0)
    print(pixels)
    return pixels


def recognize(file_name):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
        y_ = tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y-output")

        y = mnist_inference.inference(x,None)

        #将预测值与正确值相比较得到Bool型 然后将Bool型转为实数型并求平均值得准确度
        #correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化滑动平均类
        mavg = tf.train.ExponentialMovingAverage(mnist_train.Moving_Average_Decay)

        variable_to_restore = mavg.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        # validate_feed = {x: mnist.validation.images,
        #                  y_: mnist.validation.labels}
        with tf.Session() as sess:
            result = imageprepare(file_name)
            #checkpoint函数会自动找到最新模型的文件名
            ckpt = tf.train.get_checkpoint_state("model/")
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                prediction = tf.argmax(y, 1)
                predint = prediction.eval(feed_dict={x: [result]}, session=sess)
                print("result :",predint[0])
                return (predint[0])
            else:
                print("no model found")
                return

#
# def main(argv=None):
#     starttime = time.time()
#     recognize()
#
#     endtime = time.time()
#     print("pass time:",(endtime - starttime))
#
# if __name__ == '__main__':
#     tf.app.run()

