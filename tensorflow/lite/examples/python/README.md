# TensorFlow Lite Python image classification demo

This `label_image.py` script shows how you can load a pre-trained and converted
TensorFlow Lite model and use it to recognize objects in images. The Python
script accepts arguments specifying the model to use, the corresponding labels
file, and the image to process.

**Tip:**
If you're using a Raspberry Pi, instead try the [classify_picamera.py example](
https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/raspberry_pi).

Before you begin,
make sure you [have TensorFlow installed](https://www.tensorflow.org/install).


## Download sample model and image

You can use any compatible model, but the following MobileNet v1 model offers
a good demonstration of a model trained to recognize 1,000 different objects.

```
# Get photo
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > /tmp/grace_hopper.bmp
# Get model
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp
# Get labels
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt

mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/
```

## Run the sample

Note: Instead use `python` if you're using Python 2.x.

```
python3 label_image.py \
  --model_file /tmp/mobilenet_v1_1.0_224.tflite \
  --label_file /tmp/labels.txt \
  --image /tmp/grace_hopper.bmp
```

You should see results like this:

```
0.728693: military uniform
0.116163: Windsor tie
0.035517: bow tie
0.014874: mortarboard
0.011758: bolo tie
```
