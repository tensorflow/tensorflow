# Object Detection evaluation using the 2014 COCO minival dataset.

This binary evaluates the following parameters of TFLite models trained for the
**bounding box-based**
[COCO Object Detection](http://cocodataset.org/#detection-eval) task:

*   Native pre-processing latency
*   Inference latency
*   mean Average Precision (mAP) averaged across IoU thresholds from 0.5 to 0.95
    (in increments of 0.05) and all object categories.

The binary takes the path to validation images and a ground truth proto file as
inputs, along with the model and inference-specific parameters such as delegate
and number of threads. It outputs the metrics as a text proto to a file, similar
to the following:

```
num_runs: 8059
process_metrics {
  object_detection_metrics {
    pre_processing_latency {
      last_us: 27197
      max_us: 61372
      min_us: 6166
      sum_us: 189403170
      avg_us: 23502.068494850479
    }
    inference_latency {
      last_us: 386378
      max_us: 412804
      min_us: 378841
      sum_us: 3122849071
      avg_us: 387498.33366422635 # Average Inference Latency.
    }
    inference_metrics {
      num_inferences: 8059 # Number of images evaluated.
    }
    average_precision_metrics {
      individual_average_precisions {
        iou_threshold: 0.5
        average_precision: 0.26113987
      }
      individual_average_precisions {
        iou_threshold: 0.55
        average_precision: 0.2456704
      }
      individual_average_precisions {
        iou_threshold: 0.6
        average_precision: 0.22885525
      }
      individual_average_precisions {
        iou_threshold: 0.65
        average_precision: 0.20678344
      }
      individual_average_precisions {
        iou_threshold: 0.7
        average_precision: 0.18185228
      }
      individual_average_precisions {
        iou_threshold: 0.75
        average_precision: 0.14681709 # AP at IoU threshold of 0.75.
      }
      individual_average_precisions {
        iou_threshold: 0.8
        average_precision: 0.107850626
      }
      individual_average_precisions {
        iou_threshold: 0.85
        average_precision: 0.061735578
      }
      individual_average_precisions {
        iou_threshold: 0.9
        average_precision: 0.017980274
      }
      individual_average_precisions {
        iou_threshold: 0.95
        average_precision: 0.0010084915
      }
      overall_mean_average_precision: 0.14596924 # Overall mAP average.
    }
  }
}
```

To run the binary, please follow the
[Preprocessing section](#preprocessing-the-minival-dataset) to prepare the data,
and then execute the commands in the
[Running the binary section](#running-the-binary).

## Parameters

The binary takes the following parameters:

*   `model_file` : `string` \
    Path to the TFlite model file. It should accept images preprocessed in the
    Inception format, and the output signature should be similar to the
    [SSD MobileNet model](https://www.tensorflow.org/lite/models/object_detection/overview#output.):

*   `model_output_labels`: `string` \
    Path to labels that correspond to output of model. E.g. in case of
    COCO-trained SSD model, this is the path to a file where each line contains
    a class detected by the model in correct order, starting from 'background'.

A sample model & label-list combination for COCO can be downloaded from the
TFLite
[Hosted models page](https://www.tensorflow.org/lite/guide/hosted_models#object_detection).

*   `ground_truth_images_path`: `string` \
    The path to the directory containing ground truth images.

*   `ground_truth_proto`: `string` \
    Path to file containing tflite::evaluation::ObjectDetectionGroundTruth proto
    in text format. If left empty, mAP numbers are not provided.

The above two parameters can be prepared using the `preprocess_coco_minival`
script included in this folder.

*   `output_file_path`: `string` \
    The final metrics are dumped into `output_file_path` as a string-serialized
    instance of `tflite::evaluation::EvaluationStageMetrics`.

The following optional parameters can be used to modify the inference runtime:

*   `num_interpreter_threads`: `int` (default=1) \
    This modifies the number of threads used by the TFLite Interpreter for
    inference.

*   `delegate`: `string` \
    If provided, tries to use the specified delegate for accuracy evaluation.
    Valid values: "nnapi", "gpu", "hexagon".

    NOTE: Please refer to the
    [Hexagon delegate documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/hexagon_delegate.md)
    for instructions on how to set it up for the Hexagon delegate. The tool
    assumes that `libhexagon_interface.so` and Qualcomm libraries lie in
    `/data/local/tmp`.

### Debug Mode

The script also supports a debug mode with the following parameter:

*   `debug_mode`: `boolean` \
    Whether to enable debug mode. Per-image predictions are written to the
    output file along with metrics. NOTE: Its not possible to parse the output
    file as a proto in this mode, since it contains demarcations between
    per-file outputs for readability.

This mode lets you debug the output of an object detection model that isn't
necessarily trained on the COCO dataset (by leaving `ground_truth_proto` empty).
The model output signature would still need to follow the convention mentioned
above, and you we still need an output labels file.

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

## Running the binary

### On Android

(0) Refer to
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android
for configuring NDK and SDK.

(1) Build using the following command:

```
bazel build -c opt \
  --config=android_arm64 \
  --cxxopt='--std=c++17' \
  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval
```

(2) Connect your phone. Push the binary to your phone with adb push (make the
directory if required):

```
adb push bazel-bin/third_party/tensorflow/lite/tools/evaluation/tasks/coco_object_detection/run_eval /data/local/tmp
```

(3) Make the binary executable.

```
adb shell chmod +x /data/local/tmp/run_eval
```

(4) Push the TFLite model that you need to test:

```
adb push ssd_mobilenet_v1_float.tflite /data/local/tmp
```

(5) Push the model labels text file to device.

```
adb push /path/to/labelmap.txt /data/local/tmp/labelmap.txt
```

(6) Preprocess the dataset using the instructions given in the
[Preprocessing section](#preprocessing-the-minival-dataset) and push the data
(folder containing images & ground truth proto) to the device:

```
adb shell mkdir /data/local/tmp/coco_validation && \
adb push /path/to/output/folder /data/local/tmp/coco_validation
```

(7) Run the binary.

```
adb shell /data/local/tmp/run_eval \
  --model_file=/data/local/tmp/ssd_mobilenet_v1_float.tflite \
  --ground_truth_images_path=/data/local/tmp/coco_validation/images \
  --ground_truth_proto=/data/local/tmp/coco_validation/ground_truth.pbtxt \
  --model_output_labels=/data/local/tmp/labelmap.txt \
  --output_file_path=/data/local/tmp/coco_output.txt
```

Optionally, you could also pass in the `--num_interpreter_threads` &
`--delegate` arguments to run with different configurations.

### On Desktop

(1) Build and run using the following command:

```
bazel run -c opt \
  -- \
  //tensorflow/lite/tools/evaluation/tasks/coco_object_detection:run_eval \
  --model_file=/path/to/ssd_mobilenet_v1_float.tflite \
  --ground_truth_images_path=/path/to/images \
  --ground_truth_proto=/path/to/ground_truth.pbtxt \
  --model_output_labels=/path/to/labelmap.txt \
  --output_file_path=/path/to/coco_output.txt
```
