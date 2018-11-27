First, get pre-trained MobileNet SSD model(s), e.g.,
```
curl https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip -o /tmp/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip

(cd /tmp; unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
```
To prepare your own tflite models, read the object dection for TF Lite tutorial [1] and the descriptions in [2].

Then prepare input file, e.g.,
```
cp ${TF_MODELS}/research/object_detection/test_images/image2.jpg /tmp/
```


Run it

```
python3 tensorflow/lite/examples/python/object_detection.py --graph /tmp/detect.tflite --image /tmp/image2.jpg --labels /tmp/labelmap.txt --show_image True
```


[1] https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193 

[2] https://github.com/tensorflow/models/blob/master/research/object_detection/export_tflite_ssd_graph.py
