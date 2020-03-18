#from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D,Convolution1D,Convolution3D,Conv2DTranspose,Conv3DTranspose,SeparableConv1D,SeparableConv2D,DepthwiseConv2D
#from tensorflow.keras.layers import MaxPooling2D
#from tensorflow.keras.layers import Flatten
#from tensorflow.keras.layers import Dense
import tensorflow as tf
class Test(tf.test.TestCase):
    def setUp(self):
        super(Test, self).setUp()
        self.classifier=tf.keras.Sequential()
         

    def testOutput(self):
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(Convolution1D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(Convolution2D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(Convolution3D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(Conv2DTranspose(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(Conv3DTranspose(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(SeparableConv1D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(SeparableConv2D(64,0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
        with self.assertRaises(ValueError) as cm:
        	self.classifier.add(DepthwiseConv2D(0,padding="same",input_shape=(32,32,1),activation='relu'))
        print(cm.expected)
            


tf.test.main()
