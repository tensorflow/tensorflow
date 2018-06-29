# TensorFlow Lite Optimizing Converter command-line glossary

This page is complete reference of command-line flags used by TOCO's command
line starting from TensorFlow 1.9 up until the most recent build of TensorFlow.
It is complemented by the following other documents:

*   [README](../README.md)
*   [Command-line examples](cmdline_examples.md)
*   [Python API examples](python_api.md)

Table of contents:

*   [High-level flags](#high-level-flags)
*   [Model flags](#model-flags)
*   [Transformation flags](#transformation-flags)
*   [Logging flags](#logging-flags)

## High-level flags

The following high level flags specify the details of the input and output
files. The flag `--output_file` is always required. Additionally, either
`--graph_def_file` or `--saved_model_dir` is required.

*   `--output_file`. Type: string. Specifies the full path of the output file.
*   `--graph_def_file`. Type: string. Specifies the full path of the input
    GraphDef file frozen using
    [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py).
*   `--saved_model_dir`. Type: string. Specifies the full path to the directory
    containing the SavedModel.
*   `--output_format`. Type: string. Default: `TFLITE`. Specifies the format of
    the output file. Allowed values:
    *   `TFLITE`: TensorFlow Lite FlatBuffer format.
    *   `GRAPHVIZ_DOT`: GraphViz `.dot` format containg a visualization of the
        graph after graph transformations.
        *   Note that passing `GRAPHVIZ_DOT` to `--output_format` leads to loss
            of TFLite specific transformations. Therefore, the resulting
            visualization may not reflect the final set of graph
            transformations. To get a final visualization with all graph
            transformations use `--dump_graphviz` instead.

The following flags specify optional parameters when using SavedModels.

*   `--saved_model_tag_set`. Type: string. Default:
    [kSavedModelTagServe](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h).
    Specifies a comma-separated set of tags identifying the MetaGraphDef within
    the SavedModel to analyze. All tags in the tag set must be specified.
*   `--saved_model_signature_key`. Type: string. Default:
    [DEFAULT_SERVING_SIGNATURE_DEF_KEY](https://www.tensorflow.org/api_docs/python/tf/saved_model/signature_constants).
    Specifies the key identifying the SignatureDef containing inputs and
    outputs.

## Model flags

*Model flags* provide additional information about the model stored in the input
file.

*   `--input_arrays`. Type: comma-separated list of strings. Specifies the list
    of names of input activation tensors.
*   `--output_arrays`. Type: comma-separated list of strings. Specifies the list
    of names of output activation tensors.

The following flags define properties of the input tensors. Each item in the
`--input_arrays` flag should correspond to each item in the following flags
based on index.

*   `--input_shapes`. Type: colon-separated list of comma-separated lists of
    integers. Each comma-separated list of integers gives the shape of one of
    the input arrays specified in [TensorFlow
    convention](https://www.tensorflow.org/versions/r1.2/programmers_guide/dims_types#shape).
    *   Example: `--input_shapes=1,60,80,3` for a typical vision model means a
        batch size of 1, an input image height of 60, an input image width of
        80, and an input image depth of 3 (representing RGB channels).
    *   Example: `--input_arrays=foo,bar --input_shapes=2,3:4,5,6` means "foo"
        has a shape of [2, 3] and "bar" has a shape of [4, 5, 6].
*   `--std_dev_values`, `--mean_values`. Type: comma-separated list of integers.
    These specify the (de-)quantization parameters of the input array, when it
    is quantized.
    *   The meaning of `mean_values` and `std_dev_values` is as follows: each
        quantized value in the quantized input array will be interpreted as a
        mathematical real number (i.e. as an input activation value) according
        to the following formula:
        *   `real_value = (quantized_input_value - mean_value) / std_value`.
    *   When performing float inference (`--inference_type=FLOAT`) on a
        quantized input, the quantized input would be immediately dequantized by
        the inference code according to the above formula, before proceeding
        with float inference.
    *   When performing quantized inference
        (`--inference_type=QUANTIZED_UINT8`), no dequantization is performed by
        the inference code. However, the quantization parameters of all arrays,
        including those of the input arrays as specified by `mean_value` and
        `std_dev_value`, determine the fixed-point multipliers used in the
        quantized inference code.

## Transformation flags

*Transformation flags* specify options of the transformations to be applied to
the graph, i.e. they specify requested properties that the output file should
have.

*   `--inference_type`. Type: string. Default: `FLOAT`. Data type of all
    real-number arrays in the output file except for input arrays (defined by
    `--inference_input_type`). Must be `{FLOAT, QUANTIZED_UINT8}`.

    This flag only impacts real-number arrays including float and quantized
    arrays. This excludes all other data types including plain integer arrays
    and string arrays. Specifically:

    *   If `FLOAT`, then real-numbers arrays will be of type float in the output
        file. If they were quantized in the input file, then they get
        dequantized.
    *   If `QUANTIZED_UINT8`, then real-numbers arrays will be quantized as
        uint8 in the output file. If they were float in the input file, then
        they get quantized.

*   `--inference_input_type`. Type: string. Data type of a real-number input
    array in the output file. By default the `--inference_type` is used as type
    of all of the input arrays. Flag is primarily intended for generating a
    float-point graph with a quantized input array. A Dequantized operator is
    added immediately after the input array. Must be `{FLOAT, QUANTIZED_UINT8}`.

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

*   `--drop_control_dependency`. Type: boolean. Default: True. Indicates whether
    to drop control dependencies silently. This is due to TensorFlow Lite not
    supporting control dependencies.

*   `--reorder_across_fake_quant`. Type: boolean. Default: False. Indicates
    whether to reorder FakeQuant nodes in unexpected locations. Used when the
    location of the FakeQuant nodes is preventing graph transformations
    necessary to convert the graph. Results in a graph that differs from the
    quantized training graph, potentially causing differing arithmetic behavior.

*   `--allow_custom_ops`. Type: string. Default: False. Indicates whether to
    allow custom operations. When false, any unknown operation is an error. When
    true, custom ops are created for any op that is unknown. The developer will
    need to provide these to the TensorFlow Lite runtime with a custom resolver.

*   `--quantize_weights`. Type: boolean. Default: False. Indicates whether to
    store weights as quantized weights followed by dequantize operations.
    Computation is still done in float, but reduces model size (at the cost of
    accuracy and latency).

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
