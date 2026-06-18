# Converter command line reference

This page is complete reference of command-line flags used by the TensorFlow
Lite Converter's command line tool.

## High-level flags

The following high level flags specify the details of the input and output
files. The flag `--output_file` is always required. Additionally, either
`--saved_model_dir`, `--keras_model_file` or `--graph_def_file` is required.

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   `--saved_model_dir`. Type: string. Specifies the full path to the directory
    containing the SavedModel.
*   `--keras_model_file`. Type: string. Specifies the full path of the HDF5 file
    containing the tf.keras model.
*   `--graph_def_file`. Type: string. Specifies the full path of the input
    GraphDef file frozen using
    [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).
*   `--output_format`. Type: string. Default: `TFLITE`. Specifies the format of
    the output file. Allowed values:
    *   `TFLITE`: TensorFlow Lite model format.
    *   `GRAPHVIZ_DOT`: GraphViz `.dot` format containing a visualization of the
        graph after graph transformations. *Note: This only works when you set
        flag `experimental_new_converter=False`. Also, as this format leads to
        loss of TFLite specific transformations, we recommend that you use
        `--dump_graphviz_dir` instead to get a final visualization with all
        graph transformations.*
*   `--experimental_new_converter`. Type: bool. Default: True (from TF 2.2). To
    leverage MLIR-based conversion, Google's cutting edge compiler technology
    for machine learning. This enables conversion of new classes of models,
    including Mask R-CNN, Mobile BERT, etc and supports models with functional
    control flow.

The following flags specify optional parameters when using SavedModels.

*   `--saved_model_tag_set`. Type: string. Default: "serve" (for more options,
    refer to
    [tag_constants.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)).
    Specifies a comma-separated set of tags identifying the MetaGraphDef within
    the SavedModel to analyze. All tags in the tag set must be specified.
*   `--saved_model_signature_key`. Type: string. Default: "serving_default" (for
    more options, refer to
    [tf.compat.v1.saved_model.signature_constants](https://www.tensorflow.org/api_docs/python/tf/compat/v1/saved_model/signature_constants)).
    Specifies the key identifying the SignatureDef containing inputs and
    outputs.

## Model flags

*Model flags* provide additional information about the model stored in the input
file.

*   `--input_arrays`. Type: comma-separated list of strings. Specifies the list
    of names of input tensors.
*   `--output_arrays`. Type: comma-separated list of strings. Specifies the list
    of names of output tensors.

The following flags define properties of the input tensors. Each item in the
`--input_arrays` flag should correspond to each item in the following flags
based on index.

*   `--input_shapes`. Type: colon-separated list of comma-separated lists of
    integers. Each comma-separated list of integers gives the shape of one of
    the input arrays.
    *   Example: `--input_shapes=1,60,80,3` for a typical vision model means a
        batch size of 1, an input image height of 60, an input image width of
        80, and an input image depth of 3 (representing RGB channels).
    *   Example: `--input_arrays=foo,bar --input_shapes=2,3:4,5,6` means "foo"
        has a shape of [2, 3] and "bar" has a shape of [4, 5, 6].
*   `--std_dev_values`, `--mean_values`. Type: comma-separated list of floats.
    These specify the (de-)quantization parameters of the input array, when it
    is quantized. Only needed if `inference_input_type` is `INT8` or `UINT8`.
    *   The meaning of `mean_values` and `std_dev_values` is as follows: each
        quantized value in the quantized input array will be interpreted as a
        mathematical real number (i.e. as an input activation value) according
        to the following formula:
        *   `real_value = (quantized_value - mean_value) / std_dev_value`.
    *   When performing float inference (`--inference_type=FLOAT`) on a
        quantized input, the quantized input would be immediately dequantized by
        the inference code according to the above formula, before proceeding
        with float inference.
    *   When performing quantized inference (`inference_type` is `INT8` or
        `UINT8`), no dequantization is performed by the inference code. However,
        the quantization parameters of all arrays, including those of the input
        arrays as specified by `mean_value` and `std_dev_value`, determine the
        fixed-point multipliers used in the quantized inference code.The
        `mean_value` must be an integer when performing quantized inference.

## Transformation flags

*Transformation flags* specify options of the transformations to be applied to
the graph, i.e. they specify requested properties that the output file should
have.

*   `--inference_type`. Type: string. Default: `FLOAT`. Data type of all
    real-number arrays in the output file except for input arrays (defined by
    `--inference_input_type`). Must be `{FLOAT, INT8, UINT8}`.

    This flag only impacts real-number arrays including float and quantized
    arrays. This excludes all other data types including plain integer arrays
    and string arrays. Specifically:

    *   If `FLOAT`, then real-numbers arrays will be of type float in the output
        file. If they were quantized in the input file, then they get
        dequantized.
    *   If `INT8`, then real-numbers arrays will be quantized as int8 in the
        output file. If they were float in the input file, then they get
        quantized.
    *   If `UINT8`, then real-numbers arrays will be quantized as uint8 in the
        output file. If they were float in the input file, then they get
        quantized.

*   `--inference_input_type`. Type: string. Data type of a real-number input
    array in the output file. By default the `--inference_type` is used as type
    of all of the input arrays. Flag is primarily intended for generating a
    float-point graph with a quantized input array. A Dequantized operator is
    added immediately after the input array. Must be `{FLOAT, INT8, UINT8}`.

    The flag is typically used for vision models taking a bitmap as input but
    requiring floating-point inference. For such image models, the uint8 input
    is quantized and the quantization parameters used for such input arrays are
    their `mean_value` and `std_dev_value` parameters.

*   `--default_ranges_min`, `--default_ranges_max`. Type: floating-point.
    Default value for the (min, max) range values used for all arrays without a
    specified range. Allows user to proceed with quantization of non-quantized
    or incorrectly-quantized input files. These flags produce models with low
    accuracy. They are intended for easy experimentation with quantization via
    "dummy quantization".

*   `--post_training_quantize`. Type: boolean. Default: False. Boolean
    indicating whether to quantize the weights of the converted float model.
    Model size will be reduced and there will be latency improvements (at the
    cost of accuracy).

*   `--quantize_to_float16`. Type: boolean. Default: False. Boolean indicating
    whether to quantize weights to fp16 instead of the default int8 when
    `--post_training_quantize=True`.

*   `--reorder_across_fake_quant`. Type: boolean. Default: False. Indicates
    whether to reorder FakeQuant nodes in unexpected locations. Used when the
    location of the FakeQuant nodes is preventing graph transformations
    necessary to convert the graph. Results in a graph that differs from the
    quantized training graph, potentially causing differing arithmetic behavior.

*   `--change_concat_input_ranges`. Type: boolean. Default: False. Boolean to
    change behavior of min/max ranges for inputs and outputs of the concat
    operator for quantized models. Changes the ranges of concat operator overlap
    when true.

*   `--drop_control_dependency`. Type: boolean. Default: True. Indicates whether
    to drop control dependencies silently. This is due to TensorFlow Lite not
    supporting control dependencies.

*   `--target_ops`. Type: string. Default: TFLITE_BUILTINS. Experimental flag,
    subject to change. Set of OpsSet options indicating which converter to use.
    Options: TF LITE_BUILTINS,SELECT_TF_OPS,TFLITE_BUILTINS_INT8,EXPER
    IMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 . One or more option
    may be specified.

*   `--allow_custom_ops`. Type: bool. Default: False. Indicates whether to allow
    custom operations. When False, any unknown operation is an error. When True,
    custom ops are created for any op that is unknown. The developer will need
    to provide these to the TensorFlow Lite runtime with a custom resolver.

*   `--custom_opdefs`. Type: string. String representing a list of custom ops
    OpDefs delineated with commas that are included in the GraphDef. Required
    when using custom operations with `--experimental_new_converter`.

## Logging flags

The following flags generate graph visualizations of the graph as
[GraphViz](https://www.graphviz.org/) `.dot` files at various points during
graph transformations:

*   `--dump_graphviz_dir`. Type: string. Specifies the full path of the
    directory to output GraphViz `.dot` files. Outputs the graph immediately
    after reading in the graph and after all of the transformations have been
    completed.
*   `--dump_graphviz_video`. Type: boolean. Outputs GraphViz after every graph
    transformation. Requires `--dump_graphviz_dir` to be specified.

The following flag controls generating the conversion logs. The conversion log
includes a protocol buffer of analytics collected during conversion, and an HTML
file where user can preview the conversion summary.

*   `--conversion_summary_dir`. Type: string. Specifies the full path of the
    directory to output conversion logs.
