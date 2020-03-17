from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import tensorflow as tf
class Test(tf.test.TestCase):
    def setUp(self):
        super(Test, self).setUp()
        self.classifier=tf.keras.Sequential()
         

    def testOutput(self):
        try:
        	self.classifier.add(Convolution2D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        except:
        	print("Cant do")     


tf.test.main()
