# TensorFlow Lite Optimizing Converter command-line reference

This page is complete reference of command-line flags. It is complemented by the
following other documents:

*   [README](../README.md)
*   [Command-line examples](cmdline_examples.md)

Table of contents:

[TOC]

## High-level overview

A full list and detailed specification of all flags is given in the next
section. For now we focus on a higher-level description of command lines:

```
toco \
  --input_format=... \
  --output_format=... \
  --input_file=... \
  --output_file=... \
  [model flags...] \
  [transformation flags...] \
  [logging flags...]
```

In other words, the converter requires at least the following mandatory flags:
`--input_format`, `--output_format`, `--input_file`, `--output_file`. Depending
on the input and output formats, additional flags may be allowed or mandatory:

*   *Model flags* provide additional information about the model stored in the
    input file.
    *   `--output_array` or `--output_arrays` specify which arrays in the input
        file are to be considered the output activations.
    *   `--input_array` or `--input_arrays` specify which arrays in the input
        file are to be considered the input activations.
    *   `--input_shape` or `--input_shapes` specify the shapes of the input
        arrays.
    *   `--mean_value` or `--mean_values`, and `--std_value` or `--std_values`,
        give the dequantization parameters of the input arrays, for the case
        when the output file will accept quantized input arrays.
*   *Transformation flags* specify options of the transformations to be applied
    to the graph, i.e. they specify requested properties that the output file
    should have.
    *   `--input_type` specifies the type that the input arrays should have
        after transformations, in the output file. This is where you choose
        whether you want runtime inference code to accept float or quantized
        inputs. This flag only applies to float or quantized inputs, and allows
        to convert between the two. This flag has no effect on all other types
        of inputs, such as ordinary integer arrays.
    *   `--inference_type` or `--inference_types` specify the type that generic
        intermediate and output activation arrays should have after
        transformations, in the output file. This is where you choose whether
        you want runtime inference code to perform float or quantized inference
        arithmetic.
    *   Some transformation flags allow to carry on with quantization when the
        input graph is not properly quantized: `--default_ranges_min`,
        `--default_ranges_max`, `--drop_fake_quant`,
        `--reorder_across_fake_quant`.
*   *Logging flags* described below.

## Command-line flags complete reference

### Mandatory flags

*   `--input_format`. Type: string. Specifies the format of the input file.
    Allowed values:
    *   `TENSORFLOW_GRAPHDEF` &mdash; The TensorFlow GraphDef format. Both
        binary and text proto formats are allowed.
    *   `TFLITE` &mdash; The TensorFlow Lite flatbuffers format.
*   `--output_format`. Type: string. Specifies the format of the output file.
    Allowed values:
    *   `TENSORFLOW_GRAPHDEF` &mdash; The TensorFlow GraphDef format. Always
        produces a file in binary (not text) proto format.
    *   `TFLITE` &mdash; The TensorFlow Lite flatbuffers format.
        *   Whether a float or quantized TensorFlow Lite file will be produced
            depends on the `--inference_type` flag.
        *   Whether the produced TensorFlow Lite file will accept a float or
            quantized input depends on the `--input_type` flag.
    *   `GRAPHVIZ_DOT` &mdash; The GraphViz `.dot` format. This asks the
        converter to generate a reasonable graphical representation of the graph
        after simplification by a generic set of transformation.
        *   A typical `dot` command line to view the resulting graph might look
            like: `dot -Tpdf -O file.dot`.
        *   Note that since passing this `--output_format` means losing the
            information of which output format you actually care about, and
            since the converter's transformations depend on the specific output
            format, the resulting visualization may not fully reflect what you
            would get on the actual output format that you are using. To avoid
            that concern, and generally to get a visualization of exactly what
            you get in your actual output format as opposed to just a merely
            plausible visualization of a model, consider using `--dump_graphviz`
            instead and keeping your true `--output_format`.
*   `--input_file`. Type: string. Specifies the path of the input file. This may
    be either an absolute or a relative path.
*   `--output_file`. Type: string. Specifies the path of the output file.

### Model flags

*   `--output_array`. Type: string. Specifies a single array as the output
    activations. Incompatible with `--output_arrays`.
*   `--output_arrays`. Type: comma-separated list of strings. Specifies a list
    of arrays as the output activations, for models with multiple outputs.
    Incompatible with `--output_array`.
*   `--input_array`. Type: string. Specifies a single array as the input
    activations. Incompatible with `--input_arrays`.
*   `--input_arrays`. Type: comma-separated list of strings. Specifies a list of
    arrays as the input activations, for models with multiple inputs.
    Incompatible with `--input_array`.

When `--input_array` is used, the following flags are available to provide
additional information about the single input array:

*   `--input_shape`. Type: comma-separated list of integers. Specifies the shape
    of the input array, in TensorFlow convention: starting with the outer-most
    dimension (the dimension corresponding to the largest offset stride in the
    array layout), ending with the inner-most dimension (the dimension along
    which array entries are typically laid out contiguously in memory).
    *   For example, a typical vision model might pass
        `--input_shape=1,60,80,3`, meaning a batch size of 1 (no batching), an
        input image height of 60, an input image width of 80, and an input image
        depth of 3, for the typical case where the input image is a RGB bitmap
        (3 channels, depth=3) stored by horizontal scanlines (so 'width' is the
        next innermost dimension after 'depth').
*   `--mean_value` and `--std_value`. Type: floating-point. The decimal point
    character is always the dot (`.`) regardless of the locale. These specify
    the (de-)quantization parameters of the input array, to use when the output
    file will take a quantized input array (that is, when passing
    `--input_type=QUANTIZED_UINT8`).
    *   The meaning of mean_value and std_value is as follows: each quantized
        value in the quantized input array will be interpreted as a mathematical
        real number (i.e. as an input activation value) according to the
        following formula:
        *   `real_value = (quantized_input_value - mean_value) / std_value`.
    *   When performing float inference (`--inference_type=FLOAT`) on a
        quantized input, the quantized input would be immediately dequantized by
        the inference code according to the above formula, before proceeding
        with float inference.
    *   When performing quantized inference
        (`--inference_type=QUANTIZED_UINT8`), no dequantization is ever to be
        performed by the inference code; however, the quantization parameters of
        all arrays, including those of the input arrays as specified by
        mean_value and std_value, all participate in the determination of the
        fixed-point multipliers used in the quantized inference code.

When `--input_arrays` is used, the following flags are available to provide
additional information about the multiple input arrays:

*   `--input_shapes`. Type: colon-separated list of comma-separated lists of
    integers. Each comma-separated list of integer gives the shape of one of the
    input arrays specified in `--input_arrays`, in the same order. See
    `--input_shape` for details.
    *   Example: `--input_arrays=foo,bar --input_shapes=2,3:4,5,6` means that
        there are two input arrays. The first one, "foo", has shape [2,3]. The
        second one, "bar", has shape [4,5,6].
*   `--mean_values`, `--std_values`. Type: comma-separated lists of
    floating-point numbers. Each number gives the corresponding value for one of
    the input arrays specified in `--input_arrays`, in the same order. See
    `--mean_value`, `--std_value` for details.

### Transformation flags

*   `--input_type`. Type: string. Specifies what should be the type of the
    entries in the input array(s) in the output file, after transformations, for
    those input arrays that are originally either floating-point or quantized
    real numbers in the input file. If there are multiple such input arrays,
    then they all use this type. Input arrays of other types, such as arrays of
    plain integers or strings, are not concerned with this flag. Allowed values:
    *   `FLOAT` &mdash; Keep floating-point input arrays as such. Dequantize any
        quantized input array. entries ("float32").
    *   `QUANTIZED_UINT8` &mdash; Quantize floating-point input arrays, to have
        8-bit unsigned integer entries. The quantization params are specified by
        `--mean_value`, `--std_value` flags as explained in the documentation of
        these flags.
*   `--inference_type`. Type: string. Specifies what to do with floating-point
    arrays found in the input file, besides input arrays. In other words, this
    controls the possible quantization of floating-point weights, intermediate
    activations, and output activations. Has no effect on arrays that aren't
    floating-point in the input file. Allowed values:
    *   `FLOAT` &mdash; Keep floating-point arrays as floating-point in the
        output file. This corresponds to what is commonly called "floating-point
        inference".
    *   `QUANTIZED_UINT8` &mdash; Quantize floating-point arrays, changing their
        storage data type from float to some integer type:
        *   All float activations are quantized as `uint8`.
        *   Almost all float weights are quantized as `uint8`.
            *   A few exceptions exist. In particular, the bias-vectors in
                "Conv" and "FullyConnected" layers are quantized as `int32`
                instead for technical reasons.
*   `--default_ranges_min`, `--default_ranges_max`. Type: floating-point. The
    decimal point character is always the dot (`.`) regardless of the locale.
    These flags enable what is called "dummy quantization". If defined, their
    effect is to define fallback (min, max) range values for all arrays that do
    not have a properly specified (min, max) range in the input file, thus
    allowing to proceed with quantization of non-quantized or
    incorrectly-quantized input files. This enables easy performance prototyping
    ("how fast would my model run if I quantized it?") but should never be used
    in production as the resulting quantized arithmetic is inaccurate.
*   `--drop_fake_quant`. Type: boolean. Default: false. Causes fake-quantization
    nodes to be dropped from the graph. This may be used to recover a plain
    float graph from a fake-quantized graph.
*   `--reorder_across_fake_quant`. Type: boolean. Default: false. Normally,
    fake-quantization nodes must be strict boundaries for graph transformations,
    in order to ensure that quantized inference has the exact same arithmetic
    behavior as quantized training --- which is the whole point of quantized
    training and of FakeQuant nodes in the first place. However, that entails
    subtle requirements on where exactly FakeQuant nodes must be placed in the
    graph. Some quantized graphs have FakeQuant nodes at unexpected locations,
    that prevent graph transformations that are necessary in order to generate a
    well-formed quantized representation of these graphs. Such graphs should be
    fixed, but as a temporary work-around, setting this
    reorder_across_fake_quant flag allows the converter to perform necessary
    graph transformaitons on them, at the cost of no longer faithfully matching
    inference and training arithmetic.

### Logging flags

The following are standard Google logging flags:

*   `--logtostderr` redirects Google logging to standard error, typically making
    it visible in a terminal.
*   `--v` sets verbose logging levels (for debugging purposes). Defined levels:
    *   `--v=1`: log all graph transformations that did make a change on the
        graph.
    *   `--v=2`: log all graph transformations that did *not* make a change on
        the graph.

The following flags allow to generate graph visualizations of the actual graph
at various points during transformations:

*   `--dump_graphviz=/path` enables dumping of the graphs at various stages of
    processing as GraphViz `.dot` files. Generally preferred over
    `--output_format=GRAPHVIZ_DOT` as this allows you to keep your actually
    relevant `--output_format`.
*   `--dump_graphviz_video` enables dumping of the graph after every single
    graph transformation (for debugging purposes).
