# TensorFlow Lite Optimizing Converter command-line examples

This page provides examples on how to use TOCO via command line. It is
complemented by the following documents:

*   [README](../README.md)
*   [Command-line glossary](cmdline_reference.md)
*   [Python API examples](python_api.md)

Table of contents:

*   [Convert a TensorFlow SavedModel to TensorFlow Lite](#savedmodel)
*   [Convert a TensorFlow GraphDef to TensorFlow Lite for float
    inference](#graphdef-float)
*   [Quantization](#quantization)
    *   [Convert a TensorFlow GraphDef to TensorFlow Lite for quantized
        inference](#graphdef-quant)
    *   [Use "dummy-quantization" to try out quantized inference on a float
        graph](#dummy-quant)
*   [Specifying input and output arrays](#specifying-input-and-output-arrays)
    *   [Multiple output arrays](#multiple-output-arrays)
    *   [Multiple input arrays](#multiple-input-arrays)
    *   [Specifying subgraphs](#specifying-subgraphs)
*   [Other conversions supported by TOCO](#other-conversions)
    *   [Optimize a TensorFlow GraphDef](#optimize-graphdef)
    *   [Convert a TensorFlow Lite FlatBuffer back into TensorFlow GraphDef
        format](#to-graphdef)
*   [Logging](#logging)
    *   [Standard logging](#standard-logging)
    *   [Verbose logging](#verbose-logging)
    *   [Graph "video" logging](#graph-video-logging)
*   [Graph visualizations](#graph-visualizations)
    *   [Using --output_format=GRAPHVIZ_DOT](#using-output-formatgraphviz-dot)
    *   [Using --dump_graphviz](#using-dump-graphviz)
    *   [Legend for the graph visualizations](#graphviz-legend)

## Convert a TensorFlow SavedModel to TensorFlow Lite <a name="savedmodel"></a>

The follow example converts a basic TensorFlow SavedModel into a Tensorflow Lite
FlatBuffer to perform floating-point inference.

```
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --savedmodel_directory=/tmp/saved_model \
  --output_file=/tmp/foo.tflite
```

[SavedModel](https://www.tensorflow.org/programmers_guide/saved_model#using_savedmodel_with_estimators)
has fewer required flags than frozen graphs (described [below](#graphdef-float))
due to access to additional data contained within the SavedModel. The values for
`--input_arrays` and `--output_arrays` are an aggregated, alphabetized list of
the inputs and outputs in the
[SignatureDefs](https://www.tensorflow.org/serving/signature_defs) within the
[MetaGraphDef](https://www.tensorflow.org/programmers_guide/saved_model#apis_to_build_and_load_a_savedmodel)
specified by `--savedmodel_tagset`. The value for `input_shapes` is
automatically determined from the MetaGraphDef whenever possible. The default
value for `--inference_type` for SavedModels is `FLOAT`.

There is currently no support for MetaGraphDefs without a SignatureDef or for
MetaGraphDefs that use the [`assets/`
directory](https://www.tensorflow.org/programmers_guide/saved_model#structure_of_a_savedmodel_directory).

## Convert a TensorFlow GraphDef to TensorFlow Lite for float inference <a name="graphdef-float"></a>

The follow example converts a basic TensorFlow GraphDef (frozen by
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py))
into a TensorFlow Lite FlatBuffer to perform floating-point inference. Frozen
graphs contain the variables stored in Checkpoint files as Const ops.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.tflite \
  --inference_type=FLOAT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

## Quantization

### Convert a TensorFlow GraphDef to TensorFlow Lite for quantized inference <a name="graphdef-quant"></a>

TOCO is compatible with fixed point quantization models described
[here](https://www.tensorflow.org/performance/quantization). These are float
models with
[`FakeQuant*`](https://www.tensorflow.org/api_guides/python/array_ops#Fake_quantization)
ops inserted at the boundaries of fused layers to record min-max range
information. This generates a quantized inference workload that reproduces the
quantization behavior that was used during training.

The following command generates a quantized TensorFlow Lite FlatBuffer from a
"quantized" TensorFlow GraphDef.

```
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/some_quantized_graph.pb \
  --output_file=/tmp/foo.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --mean_value=128 \
  --std_value=127
```

### Use \"dummy-quantization\" to try out quantized inference on a float graph <a name="dummy-quant"></a>

In order to evaluate the possible benefit of generating a quantized graph, TOCO
allows "dummy-quantization" on float graphs. The flags `--default_ranges_min`
and `--default_ranges_max` accept plausable values for the min-max ranges of the
values in all arrays that do not have min-max information. "Dummy-quantization"
will produce lower accuracy but will emulate the performance of a correctly
quantized model.

The example below contains a model using Relu6 activation functions. Therefore,
a reasonable guess is that most activation ranges should be contained in [0, 6].

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.cc \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --default_ranges_min=0 \
  --default_ranges_max=6 \
  --mean_value=127.5 \
  --std_value=127.5
```

## Specifying input and output arrays

### Multiple output arrays

The flag `output_arrays` takes in a comma-separated list of output arrays as
seen in the example below. This is useful for models or subgraphs with multiple
outputs.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu
```

### Multiple input arrays

The flag `input_arrays` takes in a comma-separated list of input arrays as seen
in the example below. This is useful for models or subgraphs with multiple
inputs.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_array=InceptionV1/Logits/Predictions/Reshape_1
```

Note that `input_shapes` is provided as a colon-separated list. Each input shape
corresponds to the input array at the same position in the respective list.

### Specifying subgraphs

Any array in the input file can be specified as an input or output array in
order to extract subgraphs out of an input graph file. TOCO discards the parts
of the graph outside of the specific subgraph. Use [graph
visualizations](#graph-visualizations) to identify the input and output arrays
that make up the desired subgraph.

The follow command shows how to extract a single fused layer out of a TensorFlow
GraphDef.

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

Note that the final representation of an on-device inference workload (say, in
TensorFlow Lite FlatBuffers format) tends to have coarser granularity than the
very fine granularity of the TensorFlow GraphDef representation. For example,
while a fully-connected layer is typically represented as at least four separate
ops in TensorFlow GraphDef (Reshape, MatMul, BiasAdd, Relu...), it is typically
represented as a single "fused" op (FullyConnected) in the converter's optimized
representation and in the final on-device representation (e.g. in TensorFlow
Lite FlatBuffer format). As the level of granularity gets coarser, some
intermediate arrays (say, the array between the MatMul and the BiasAdd in the
TensorFlow GraphDef) are dropped. When specifying intermediate arrays as
`--input_arrays` / `--output_arrays`, it is desirable (and often required) to
specify arrays that are meant to survive in the final form of the graph, after
fusing. These are typically the outputs of activation functions (since
everything in each layer until the activation function tends to get fused).

## Other conversions supported by TOCO <a name="other-conversions"></a>

The converter accepts both TENSORFLOW_GRAPHDEF and TFLITE file formats as both
`--input_format` and `--output_format`. This means that conversion to and from
any supported format is possible.

### Optimize a TensorFlow GraphDef <a name="optimize-graphdef"></a>

Same-format "conversions" can be used to optimize and simplify a graph or be
used to [get a subgraph](#specifying-subgraphs) of a graph. The flag
`--inference_type` is not required because TensorFlow graphs, including those
containing the
[`FakeQuant*`](https://www.tensorflow.org/api_guides/python/array_ops#Fake_quantization)
ops are always float graphs.

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

### Convert a TensorFlow Lite FlatBuffer back into TensorFlow GraphDef format <a name="to-graphdef"></a>

The converter supports file format conversions from TensorFlow Lite, back into
TensorFlow GraphDef format.

```
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/foo.tflite \
  --output_file=/tmp/foo.pb \
  --input_format=TFLITE \
  --output_format=TENSORFLOW_GRAPHDEF \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1
```

## Logging

### Standard logging

The converter generates some informative log messages during processing. The
easiest way to view them is to add `--logtostderr` to command lines as seen in
the following example.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
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

*   For `--v=1`, the converter generates text dumps of the graph at various
    points during processing as well as log messages about every graph
    transformation that took place.
*   For `--v=2`, the converter additionally generates log messages about graph
    transformations that were considered but not performed.

### Graph "video" logging

When `--dump_graphviz=` is used (see the section on [graph
visualizations](#graph-visualizations)), one may additionally pass
`--dump_graphviz_video`, which causes a graph visualization to be dumped after
each individual graph transformation. This results in thousands of files.
Typically, one would then bisect into these files to understand when a given
change was introduced in the graph.

## Graph visualizations

TOCO can export a graph to the GraphViz Dot format for easy visualization via
either the `--output_format` flag or the `--dump_graphviz` flag. The subsections
below outline the use cases for each.

### Using `--output_format=GRAPHVIZ_DOT`

The first way to get a graphviz rendering is to pass `GRAPHVIZ_DOT` into
`--output_format`. This results in a plausable visualization of the graph. This
reduces the requirements that normally exist during conversion between other
input and output formats. For example, this may be useful if conversion from
TENSORFLOW_GRAPHDEF to TFLITE is failing.

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
with a good ability to pan and zoom across a very large page. Google Chrome does
well in that respect.

```
google-chrome /tmp/foo.dot.pdf
```

Example PDF files are viewable online in the next section.

### Using `--dump_graphviz`

The second way to get a graphviz rendering is to pass the `--dump_graphviz=`
flag, specifying a destination directory to dump GraphViz rendering to. Unlike
the previous approach, this one allows you to keep your real command-line (with
your real `--output_format` and other flags) unchanged, just appending a
`--dump_graphviz=` flag to it. This provides a visualization of the actual graph
during a specific conversion process.

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
bazel run --config=opt \
  //tensorflow/contrib/lite/toco:toco -- \
  --input_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shape=1,128,128,3 \
  --input_array=input \
  --output_array=MobilenetV1/Predictions/Reshape_1 \
  --dump_graphviz=/tmp
```

This generates a few files in the destination directory, here `/tmp`. The two
most important files are:

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
Typically, this is a much smaller graph with more information about each node.

Again, these can be rendered to PDFs:

```
dot -Tpdf -O /tmp/toco_*.dot
```

Sample output files can be seen here:

*   [toco_AT_IMPORT.dot.pdf](https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AT_IMPORT.dot.pdf)
*   [toco_AFTER_TRANSFORMATIONS.dot.pdf](https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AFTER_TRANSFORMATIONS.dot.pdf).

### Legend for the graph visualizations <a name="graphviz-legend"></a>

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
