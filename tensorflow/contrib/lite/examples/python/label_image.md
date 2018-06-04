With model (mobilenet_v1_1.0_224_quant.tflite), input image
(grace_hooper.bmp), and labels file (labels.txt) in /tmp.
Run

```
bazel run --config opt //tensorflow/contrib/lite/examples/python:label_image
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
bazel run --config opt //tensorflow/contrib/lite/examples/python:label_image \
-- --graph /tmp/mobilenet_v1_1.0_224.tflite
```

We can get results like
```
0.728693: military uniform
0.116163: Windsor tie
0.035517: bow tie
0.014874: mortarboard
0.011758: bolo tie
```

Check [models](../../g3doc/models.md) hosted by Google.
