# Object Detection evaluation using the 2014 COCO minival dataset.

**NOTE: This is still a Work-In-Progress**

This binary evaluates the following parameters of TFLite models trained for the
**bounding box-based**
[COCO Object Detection](http://cocodataset.org/#detection-eval) task:

*   Native pre-processing latency
*   Inference latency
*   mean Average Precision (mAP) averaged across IoU thresholds from 0.5 to 0.95
    (in increments of 0.05) and all object categories.

## Preprocessing the minival dataset

To compute mAP in a consistent and interpretable way, we utilize the same 2014
COCO 'minival' dataset that is mentioned in the
[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

The links to download the components of the validation set are:

*   [2014 COCO Validation Images](http://images.cocodataset.org/zips/val2014.zip)
*   [2014 COCO Train/Val annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip):
    Out of the files from this zip, we only require `instances_val2014.json`.
*   [minival Image IDs](https://github.com/tensorflow/models/blob/master/research/object_detection/data/mscoco_minival_ids.txt) :
    Only applies to the 2014 validation set. You would need to copy the contents
    into a text file.

Since evaluation has to be performed on-device, we first filter the above data
and extract a subset that only contains the images & ground-truth bounding boxes
we need.

To do so, we utilize the `preprocess_coco_minival` Python binary as follows:

```
bazel run //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:preprocess_coco_minival -- \
  --images_folder=/path/to/val2014 \
  --instances_file=/path/to/instances_val2014.json \
  --whitelist_file=/path/to/minival_whitelist.txt \
  --output_folder=/path/to/output/folder

```

Optionally, you can specify a `--num_images=N` argument, to preprocess the first
`N` image files (based on sorted list of filenames).

The script generates the following within the output folder:

*   `images/`: the resulting subset of the 2014 COCO Validation images.

*   `ground_truth.pbtxt`: a `.pbtxt` (text proto) file holding
    `tflite::evaluation::ObjectDetectionGroundTruth` corresponding to image
    subset.
