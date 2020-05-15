# TensorFlow 1.x Compatibility <a name="differences"></a>

The `tf.lite.TFLiteConverter` Python API was updated between TensorFlow 1.x and
2.x. This document explains the differences between the two versions, and
provides information about how to use the 1.x version if required.

If any of the changes raise concerns, please file a
[GitHub Issue](https://github.com/tensorflow/tensorflow/issues).

Note: We highly recommend that you
[migrate your TensorFlow 1.x code to TensorFlow 2.x code](https://www.tensorflow.org/guide/migrate)
.

## Model formats

#### SavedModel and Keras

The `tf.lite.TFLiteConverter` API supports SavedModel and Keras HDF5 files
generated in both TensorFlow 1.x and 2.x.

#### Frozen Graph

Note: TensorFlow 2.x no longer supports the generation of frozen graph models.

The `tf.compat.v1.lite.TFLiteConverter` API supports frozen graph models
generated in TensorFlow 1.x, as shown below:

```python
import tensorflow as tf
# Path to the frozen graph file
graph_def_file = 'frozen_graph.pb'
# A list of the names of the model's input tensors
input_arrays = ['input_name']
# A list of the names of the model's output tensors
output_arrays = ['output_name']
# Load and convert the frozen graph
converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
# Write the converted model to disk
open("converted_model.tflite", "wb").write(tflite_model)
```

## Converter attributes

#### Renamed attributes

The following 1.x attribute has been renamed in 2.x.

*   `target_ops` has been renamed to `target_spec.supported_ops` - In 2.x, in
    line with future additions to the optimization framework, it has become an
    attribute of `TargetSpec` and has been renamed to `supported_ops`.

#### Unsupported attributes

The following 1.x attributes have been removed in 2.x.

*   _Quantization_ - In 2.x,
    [quantize aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)
    is supported through the Keras API and
    [post training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
    uses fewer streamlined converter flags. Thus, the following attributes and
    methods related to quantization have been removed:
    *   `inference_type`
    *   `quantized_input_stats`
    *   `post_training_quantize`
    *   `default_ranges_stats`
    *   `reorder_across_fake_quant`
    *   `change_concat_input_ranges`
    *   `get_input_arrays()`
*   _Visualization_ - In 2.x, the recommended approach for visualizing a
    TensorFlow Lite graph is to use
    [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)
    . Unlike GraphViz, it enables users to visualize the graph after post
    training quantization has occurred. Thus, the following attributes related
    to graph visualization have been removed:
    *   `output_format`
    *   `dump_graphviz_dir`
    *   `dump_graphviz_video`
*   _Frozen graph_ - In 2.x, the frozen graph model format has been removed.
    Thus, the following attribute related to frozen graphs has been removed:
    *   `drop_control_dependency`

## Unsupported APIs

The following section explains several significant features in 1.x that have
been removed in 2.x.

#### Conversion APIs

The following methods were deprecated in 1.x and have been removed in 2.x:

*   `lite.toco_convert`
*   `lite.TocoConverter`

#### `lite.constants` API

The `lite.constants` API was removed in 2.x in order to decrease duplication
between TensorFlow and TensorFlow Lite. The following list maps the
`lite.constant` type to the TensorFlow type:

*   `lite.constants.FLOAT`: `tf.float32`
*   `lite.constants.INT8`: `tf.int8`
*   `lite.constants.INT32`: `tf.int32`
*   `lite.constants.INT64`: `tf.int64`
*   `lite.constants.STRING`: `tf.string`
*   `lite.constants.QUANTIZED_UINT8`: `tf.uint8`

Additionally, the deprecation of the `output_format` flag in `TFLiteConverter`
led to the removal of the following constants:

*   `lite.constants.TFLITE`
*   `lite.constants.GRAPHVIZ_DOT`

#### `lite.OpHint` API

The `OpHint` API is currently unsupported due to an incompatibility with the 2.x
APIs. This API enables conversion of LSTM based models. Support for LSTMs in 2.x
is being investigated. All related `lite.experimental` APIs have been removed
due to this issue.
