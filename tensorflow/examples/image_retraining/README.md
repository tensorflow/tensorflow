**NOTE: This code has moved to**
https://github.com/tensorflow/hub/tree/master/examples/image_retraining

retrain.py is an example script that shows how one can adapt a pretrained
network for other classification problems (including use with TFLite and
quantization).

As of TensorFlow 1.7, it is recommended to use a pretrained network from
TensorFlow Hub, using the new version of this example found in the location
above, as explained in TensorFlow's revised
[image retraining tutorial](https://www.tensorflow.org/hub/tutorials/tf2_image_retraining).

Older versions of this example (using frozen GraphDefs instead of
TensorFlow Hub modules) are available in the release branches of
TensorFlow versions up to and including 1.7.
