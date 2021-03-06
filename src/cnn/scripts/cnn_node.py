#!/usr/bin/env python
import cv2
import rospy
import rospkg
import ros_numpy
import numpy as np
from scipy.misc import imresize
from sensor_msgs.msg import Image

class CNN():
    def __init__(self):

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        set_session(tf.Session(config=config))

        from keras.models import load_model

        model_path = rospy.get_param('/cnn/model_name', 'model_j_nd.h5')

        rospack = rospkg.RosPack()

        model_path = rospack.get_path('cnn') + '/' + model_path

        self.model = load_model(model_path)
        self.model._make_predict_function()

        self.sub = rospy.Subscriber(
            rospy.get_param('/cnn/sub_topic', '/cnn/image'),
            Image, self._callback)

        self.pub = rospy.Publisher(
            rospy.get_param('/cnn/pub_topic', '/cnn/output'),
            Image, queue_size=1)

        self.imgpub = rospy.Publisher(
            rospy.get_param('/cnn/imgpub_topic', '/cnn/image_output'),
            Image, queue_size=1)

        rospy.init_node('cnn', anonymous=True)

        rospy.spin()

    def _callback(self, image):
        data = ros_numpy.numpify(image)

        shape = data.shape

        original = data

        data = imresize(data, (80, 160, 3))

        data = data[None,:,:,:]

        prediction = self.model.predict(data, batch_size=1)[0] * 255

        prediction = prediction.astype(np.uint8)

        prediction = prediction[:,:,0]

        prediction = imresize(prediction, (shape[0], shape[1]))

        blanks = np.zeros_like(prediction).astype(np.uint8)
        lane_drawn = np.dstack((blanks, prediction, blanks))

        msg = ros_numpy.msgify(Image, prediction, encoding='mono8')

        self.pub.publish(msg)

        outimg = cv2.addWeighted(original, 1, lane_drawn, 1, 0)

        imgmsg = ros_numpy.msgify(Image, outimg, encoding='rgb8')

        self.imgpub.publish(imgmsg)


if __name__ == '__main__':
    try:
        cnn = CNN()
    except rospy.ROSInterruptException:
        pass
