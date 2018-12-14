
With model, input image (grace_hopper.bmp), and labels file (labels.txt)
in /tmp.

The example input image and labels file are from TensorFlow repo and
MobileNet V1 model files.

```
curl https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp > /tmp/grace_hopper.bmp

curl  https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz  | tar xzv -C /tmp  mobilenet_v1_1.0_224/labels.txt
mv /tmp/mobilenet_v1_1.0_224/labels.txt /tmp/

```

Run

```
curl http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz | tar xzv -C /tmp
bazel run --config opt //tensorflow/lite/examples/python:label_image
```

We can get results like

```
0.470588: military uniform
0.337255: Windsor tie
0.047059: bow tie
0.031373: mortarboard
0.019608: suit
```

Run

```
curl http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz | tar xzv -C /tmp
bazel run --config opt //tensorflow/lite/examples/python:label_image \
-- --model_file /tmp/mobilenet_v1_1.0_224.tflite
```

We can get results like
```
0.728693: military uniform
0.116163: Windsor tie
0.035517: bow tie
0.014874: mortarboard
0.011758: bolo tie
```

Check [models](../../g3doc/models.md) for models hosted by Google.
