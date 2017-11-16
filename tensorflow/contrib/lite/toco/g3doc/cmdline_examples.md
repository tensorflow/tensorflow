# TensorFlow Lite Optimizing Converter command-line examples

This page is a guide to using the TensorFlow Lite Optimizing Converter by
looking at some example command lines. It is complemented by the following other
documents:

*   [README](../README.md)
*   [Command-line reference](cmdline_reference.md)

Table of contents:

[TOC]

## Convert a TensorFlow GraphDef to TensorFlow Lite for float inference

In this example, we look at the most common task: we have an ordinary TensorFlow
GraphDef and want to convert it to a TensorFlow Lite flatbuffer to perform
floating-point inference.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=FLOAT \
  --inference_type=FLOAT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

To explain each of these flags:

*   `--input_format` and `--output_format` determine the formats of the input
    and output files: here we are converting from `TENSORFLOW_GRAPHDEF` to
    `TFLITE`.
*   `--input_file` specifies the path of the input file, to be converted. When
    `--input_format=TENSORFLOW_GRAPHDEF`, this file should be a
    *[frozen](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)*
    *inference* graph. Being frozen means in particular that the input file is
    self-contained, and does not reference any external "checkpoint" file. An
    *inference* graph is a version of a graph meant to be used for inference,
    typically not the same graph file as was used for training a given model.
*   `--output_file` specifies the destination to write the converted file to.
*   `--input_array` specifies the input activations, that is, the input "tensor"
    in the input TensorFlow GraphDef file. The array designated by
    `--input_array` is the one that the user will have to provide the contents
    of as input to the runtime inference code.
*   `--output_array` specifies the output activations, that is, the output
    "tensor" in the input TensorFlow GraphDef file. The runtime inference code
    will store its results in the array designated by `--output_array`.
*   `--input_shape` specifies the shape of the input array. It is currently
    required, but the plan is for a future version to no longer require it,
    allowing to defer the specification of the input shape until runtime. The
    format of `input_shape` is always a comma-separated list of dimensions,
    always in TensorFlow convention.
*   `--input_type` specifies what should be the type of the input arrays in the
    **output** file. `--input_type` does not describe a property of the input
    file: the type of input arrays is already encoded in the input graph.
    Rather, `--input_type` is how you specify what should be the type of the
    inputs to be provided to the output converted graph. This only affects
    arrays of real numbers: this flag allows to quantized/dequantize
    real-numbers inputs, switching between floating-point and quantized forms.
    This flag has no incidence on all other types of input arrays, such as plain
    integers or strings.
*   `--inference_type` specifies what type of arithmetic the output file should
    be relying on. It implies in particular the choice of type of the output
    arrays in the output file. Like `--input_type`, `--inference_type` does not
    describe a property of the input file.

## Just optimize a TensorFlow GraphDef

The converter accepts both TENSORFLOW_GRAPHDEF and TFLITE file formats as both
`--input_format` and `--output_format`. This means that conversion from and to
any supported format is possible, and in particular, same-format "conversions"
are possible, and effectively ask the converter to optimize and simplify a
graph. Example:

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TENSORFLOW_GRAPHDEF \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

Here we did not pass `--input_type` and `--inference_type` because they are
considered not applicable to the TensorFlow GraphDef format (as far as we are
concerned, TensorFlow GraphDefs are technically always float, and the only
flavor of "quantized" GraphDef that the converter deals with is "FakeQuantized"
graphs that are still technically float graphs).

Below in the section about passing arbitrary input/output arrays we give another
example, using the converter to extract just a sub-graph from a TensorFlow
GraphDef.

## Convert a TensorFlow Lite flatbuffer back into TensorFlow GraphDef format

As we mentioned that the converter supports file format conversions in any
direction, let us just give an example of that:

```
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/foo.lite \
  --output_file=/tmp/foo.pb \
  --input_format=TFLITE \
  --output_format=TENSORFLOW_GRAPHDEF \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

## Convert a TensorFlow GraphDef to TensorFlow Lite for quantized inference

Let us now look at a quantized model. As mentioned above, the only flavor of
quantized TensorFlow GraphDefs that the converter is concerned with, is
"FakeQuantized" models. These are technically float models, but with special
`FakeQuant*` ops inserted at the boundaries of fused layers to record min-max
range information allowing to generate a quantized inference workload that is
able to reproduce exactly the specific quantization behavior that was used
during training. Indeed, the whole point of quantized training is to allow for
both training and inference to perform exactly the same arithmetic, so that the
way that the training process about around quantization inaccuracy is
effectively helping the quantized inference process to be more accurate.

Given a quantized TensorFlow GraphDef, generating a quantized TensorFlow Lite
flatbuffer is done like this:

```
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/some_quantized_graph.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=QUANTIZED_UINT8 \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --mean_value=128 \
  --std_value=127
```

Here, besides changing `--input_file` to point to a (fake-)quantized GraphDef,
the only other changes are:

*   To change `--input_type` and `--inference_type` to `QUANTIZED_UINT8`. This
    effectively tells the converter to generate an output file that can take a
    quantized uint8 array as input (`--input_type=QUANTIZED_UINT8`), and have
    quantized uint8 internal and output arrays as well
    (`--inference_type=QUANTIZED_UINT8`).
*   To pass `--mean_value` and `--std_value` flags to describe how the quantized
    uint8 input array values are to be interpreted as the mathematical real
    numbers that the graph is concerned with (keep in mind that even a
    "fake-quantized" TensorFlow GraphDef is still technically a float graph).
    The meaning of `--mean_value` and `--std_value` is explained in the
    command-line reference; it suffices for now to say that they are a property
    of each model.

## Use dummy-quantization to try out quantized inference on a float graph

Sometimes, one only has a plain float graph, and one is curious as to how much
faster inference might run if one could perform quantized inference instead of
float inference. Rather than requiring users to first invest in quantizing their
graphs before they can evaluate a possible benefit, the converter allows to
simply experiment with what we call "dummy quantization": provide some vaguely
plausible values for the min-max ranges of values in all arrays that do not have
min-max information, so that quantization can carry on, certainly producing
inaccurate results (do not use that in production!) but with performance
characteristics that should be identical to those of an actually quantized
flavor of the model.

In the present example, we have a model using Relu6 activation functions almost
everywhere, so a reasonable guess is that most activation ranges should be
contained in [0, 6] and roughly comparable to it.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.cc \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=QUANTIZED_UINT8 \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --default_ranges_min=0 \
  --default_ranges_max=6 \
  --mean_value=127.5 \
  --std_value=127.5
```

## Multiple output arrays

Some models have multiple outputs. Even in a model with only one output, you may
want for the inference code to return the contents of other arrays as well, or
to perform inference on a subgraph with multiple outputs (see the section below
on specifying arbitrary arrays as input/output arrays).

Either way, using `--output_arrays` instead of `--output_array` allows to
specify a comma-separated list of output arrays.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=FLOAT \
  --inference_type=FLOAT \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu
```

## Multiple input arrays

Some models have multiple inputs; even in a model with a single input, you may
want for the inference code to implement only a subgraph with multiple inputs
(see the section below on specifying arbitrary arrays as input/output arrays).

Either way, multiple input arrays are specified by using `--input_arrays`
instead of `--input_array` to specify a comma-separated list of input arrays. In
that case, one also needs to use `--input_shapes` instead of `--input_shape`.
The syntax for `--input_shapes` is a bit trickier, since already the singular
`--input_shape` was a comma-separated list of integers! Multiple input shapes
are delimited by a colon (`:`) in `--input_shapes`.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=FLOAT \
  --inference_type=FLOAT \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_array=InceptionV1/Logits/Predictions/Reshape_1
```

## Specifying arbitrary arrays in a graph as input or output arrays

Any array in the input file can be specified as an input or output array. This
allows to use the converter to extract a sub-graph out of the input graph file.
The converter then automatically discards any part of the graph that is not
needed for the subgraph identified by the specified input and output arrays.
Another use case for specifying multiple output arrays is to get inference code
to return the contents of some specified intermediate activations array, not
just the output activations.

In order to know which array you want to pass as `--input_arrays` /
`--output_arrays`, it helps to have a visualization of the graph. See the
section below on graph visualization. When using graph visualization for that
purpose, make sure to use `--dump_graphviz=` to visualize exactly the graph as
it is in the actual final form being exported to the output file.

Note that the final representation of an on-device inference workload (say, in
TensorFlow Lite flatbuffers format) tends to have coarser granularity than the
very fine granularity of the TensorFlow GraphDef representation. For example,
while a fully-connected layer is typically represented as at least four separate
ops in TensorFlow GraphDef (Reshape, MatMul, BiasAdd, Relu...), it is typically
represented as a single "fused" op (FullyConnected) in the converter's optimized
representation and in the final on-device representation (e.g. in TensorFlow
Lite flatbuffer format). As the level of granularity gets coarser, some
intermediate arrays (say, the array between the MatMul and the BiasAdd in the
TensorFlow GraphDef) are dropped. When specifying intermediate arrays as
`--input_arrays` / `--output_arrays`, it is generally at least desirable (and
often required) to specify arrays that are meant to survive in the final form of
the graph, after fusing. These are typically the outputs of activation functions
(since everything in each layer until the activation function tends to get
fused).

Here is an example of extracting just a sub-graph, namely just a single fused
layer, out of a TensorFlow GraphDef, and exporting a TensorFlow GraphDef
containing just that subgraph:

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TENSORFLOW_GRAPHDEF \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_array=InceptionV1/InceptionV1/Mixed_3b/concat_v2
```

## Logging

### Standard logging

The converter generates some informative log messages during processing. The
easiest way to view them is to add `--logtostderr` to command lines. For the
previous example, that gives:

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=FLOAT \
  --inference_type=FLOAT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --logtostderr
```

After some initialization messages, we get the following informative messages:

```
I1101 21:51:33.297475    5339 graph_transformations.cc:39] Before general graph transformations: 416 operators, 583 arrays (0 quantized)
I1101 21:51:33.308972    5339 graph_transformations.cc:39] After general graph transformations pass 1: 31 operators, 89 arrays (0 quantized)
I1101 21:51:33.309204    5339 graph_transformations.cc:39] Before dequantization graph transformations: 31 operators, 89 arrays (0 quantized)
I1101 21:51:33.309368    5339 allocate_transient_arrays.cc:312] Total transient array allocated size: 1048576 bytes, theoretical optimal value: 786432 bytes.
I1101 21:51:33.309484    5339 toco_tooling.cc:249] Estimated count of arithmetic ops: 0.099218 billion (note that a multiply-add is counted as 2 ops).
```

### Verbose logging

For debugging purposes, the converter supports two levels of verbose logging,
which can be set by passing a `--v=` flag:

*   At `--v=1`, the converter generates text dumps of the graph at various
    points during processing, as well as log messages about every graph
    transformation that did take place, typically answering questions of the
    form "why was my graph transformed in this way"?
*   At `--v=2`, the converter additionally generates log messages about graph
    transformations that were considered but not actually performed, typically
    answering questions of the form "why was my graph NOT transformed when I
    expected it would be?".

### Graph "video" logging

When `--dump_graphviz=` is used (see the section on Graph visualizations), one
may additionally pass `--dump_graphviz_video`, which causes a graph
visualization to be dumped after each individual graph transformations, often
resulting in thousands of files. Typically, one would then bisect into these
files to understand when a given change was introduced in the graph.

## Graph visualizations

The converter is able to export a graph to the GraphViz Dot format, for easy
visualization. Combined with the converter's ability to transform the graph into
a simpler, coarser-granularity representation, that makes it a very powerful
visualization tool.

There are two ways to get the converter to export a GraphViz Dot file,
corresponding to two separate use cases. Understanding the difference between
them is key to getting useful graph visualizations.

### Using `--output_format=GRAPHVIZ_DOT`

The first way to get a graphviz rendering is to pass
`--output_format=GRAPHVIZ_DOT`, instead of the `--output_format` that you would
otherwise use. This says: "I just want to get a plausible visualization of that
graph". The upside is that it makes for very simple command lines, and makes the
converter very lax about aspects of the graph or the command line that it would
otherwise complain about. Example:

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.dot \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=GRAPHVIZ_DOT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

The resulting `.dot` file can be rendered into a PDF as follows:

```
dot -Tpdf -O /tmp/foo.dot
```

And the resulting `.dot.pdf` can be viewed in any PDF viewer, but we suggest one
with a good ability to pan and zoom across a very large page; Google Chrome does
well in that respect.

```
google-chrome /tmp/foo.dot.pdf
```

Example PDF files are viewable online in the next section.

### Using `--dump_graphviz=`

The second way to get a graphviz rendering is to pass a `--dump_graphviz=` flag
specifying a destination directory to dump GraphViz rendering to. Unlike the
previous approach, this one allows you to keep your real command-line (with your
real `--output_format` and other flags) unchanged, just appending a
`--dump_graphviz=` flag to it. This says: "I want visualizations of the actual
graph during this specific conversion process". Example:

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.lite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_type=FLOAT \
  --inference_type=FLOAT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --dump_graphviz=/tmp
```

This generates a few files in the destination directory, here `/tmp`. Most
important are these two files:

```
/tmp/toco_AT_IMPORT.dot
/tmp/toco_AFTER_TRANSFORMATIONS.dot
```

`toco_AT_IMPORT.dot` represents the graph as it was imported from
`--input_file`, before any transformation was applied to it (besides some
transformations that are applied immediately while importing). This tends to be
a complex visualization with limited information, but is useful especially in
situations where a conversion command fails (this file is generated even if the
conversion subsequently fails).

`toco_AFTER_TRANSFORMATIONS.dot` represents the graph after all transformations
were applied to it, just before it was exported to the `--output_file`.
Typically, this is a much smaller graph, and it conveys much more information
about each node.

Again, these can be rendered to PDFs:

```
dot -Tpdf -O /tmp/toco_*.dot
```

The resulting files can be seen here:

*   [toco_AT_IMPORT.dot.pdf](https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AT_IMPORT.dot.pdf)
*   [toco_AFTER_TRANSFORMATIONS.dot.pdf](https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AFTER_TRANSFORMATIONS.dot.pdf).

### Legend for the graph visualizations

*   Operators are red square boxes with the following hues of red:
    *   Most operators are
        <span style="background-color:#db4437;color:white;border:1px;border-style:solid;border-color:black;padding:1px">bright
        red</span>.
    *   Some typically heavy operators (e.g. Conv) are rendered in a
        <span style="background-color:#c53929;color:white;border:1px;border-style:solid;border-color:black;padding:1px">darker
        red</span>.
*   Arrays are octogons with the following colors:
    *   Constant arrays are
        <span style="background-color:#4285f4;color:white;border:1px;border-style:solid;border-color:black;padding:1px">blue</span>.
    *   Activation arrays are gray:
        *   Internal (intermediate) activation arrays are
            <span style="background-color:#f5f5f5;border:1px;border-style:solid;border-color:black;border:1px;border-style:solid;border-color:black;padding:1px">light
            gray</span>.
        *   Those activation arrays that are designated as `--input_arrays` or
            `--output_arrays` are
            <span style="background-color:#9e9e9e;border:1px;border-style:solid;border-color:black;padding:1px">dark
            gray</span>.
    *   RNN state arrays are green. Because of the way that the converter
        represents RNN back-edges explicitly, each RNN state is represented by a
        pair of green arrays:
        *   The activation array that is the source of the RNN back-edge (i.e.
            whose contents are copied into the RNN state array after having been
            computed) is
            <span style="background-color:#b7e1cd;border:1px;border-style:solid;border-color:black;padding:1px">light
            green</span>.
        *   The actual RNN state array is
            <span style="background-color:#0f9d58;color:white;border:1px;border-style:solid;border-color:black;padding:1px">dark
            green</span>. It is the destination of the RNN back-edge updating
            it.
