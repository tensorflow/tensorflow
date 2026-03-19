# API Updates <a name="api_updates"></a>

This page provides information about updates made to the
`tf.lite.TFLiteConverter` [Python API](index.md) in TensorFlow 2.x.

Note: If any of the changes raise concerns, please file a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues/new?template=60-tflite-converter-issue.md).

*   TensorFlow 2.3

    *   Support integer (previously, only float) input/output type for integer
        quantized models using the new `inference_input_type` and
        `inference_output_type` attributes. Refer to this
        [example usage](../../performance/post_training_quantization.md#integer_only).
    *   Support conversion and resizing of models with dynamic dimensions.
    *   Added a new experimental quantization mode with 16-bit activations and
        8-bit weights.

*   TensorFlow 2.2

    *   By default, leverage [MLIR-based conversion](https://mlir.llvm.org/),
        Google's cutting edge compiler technology for machine learning. This
        enables conversion of new classes of models, including Mask R-CNN,
        Mobile BERT, etc and supports models with functional control flow.

*   TensorFlow 2.0 vs TensorFlow 1.x

    *   Renamed the `target_ops` attribute to `target_spec.supported_ops`
    *   Removed the following attributes:
        *   _quantization_: `inference_type`, `quantized_input_stats`,
            `post_training_quantize`, `default_ranges_stats`,
            `reorder_across_fake_quant`, `change_concat_input_ranges`,
            `get_input_arrays()`. Instead,
            [quantize aware training](https://www.tensorflow.org/model_optimization/guide/quantization/training)
            is supported through the `tf.keras` API and
            [post training quantization](../../performance/post_training_quantization.md)
            uses fewer attributes.
        *   _visualization_: `output_format`, `dump_graphviz_dir`,
            `dump_graphviz_video`. Instead, the recommended approach for
            visualizing a TensorFlow Lite model is to use
            [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py).
        *   _frozen graphs_: `drop_control_dependency`, as frozen graphs are
            unsupported in TensorFlow 2.x.
    *   Removed other converter APIs such as `tf.lite.toco_convert` and
        `tf.lite.TocoConverter`
    *   Removed other related APIs such as `tf.lite.OpHint` and
        `tf.lite.constants` (the `tf.lite.constants.*` types have been mapped to
        `tf.*` TensorFlow data types, to reduce duplication)
