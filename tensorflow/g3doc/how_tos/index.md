# How-Tos


## Variables: Creation, Initializing, Saving, and Restoring

TensorFlow Variables are in-memory buffers containing tensors.  Learn how to
use them to hold and update model parameters during training.

[View Tutorial](variables/index.md)


## TensorFlow Mechanics 101

A step-by-step walk through of the details of using TensorFlow infrastructure
to train models at scale, using MNIST handwritten digit recognition as a toy
example.

[View Tutorial](../tutorials/mnist/tf/index.md)


## TensorBoard: Visualizing Learning

TensorBoard is a useful tool for visualizing the training and evaluation of
your model(s).  This tutorial describes how to build and run TensorBoard as well
as how to add Summary ops to automatically output data to the Events files that
TensorBoard uses for display.

[View Tutorial](summaries_and_tensorboard/index.md)


## TensorBoard: Graph Visualization

This tutorial describes how to use the graph visualizer in TensorBoard to help
you understand the dataflow graph and debug it.

[View Tutorial](graph_viz/index.md)


## Reading Data

This tutorial describes the three main methods of getting data into your
TensorFlow program: Feeding, Reading and Preloading.

[View Tutorial](reading_data/index.md)

## Distributed TensorFlow

This tutorial describes how to execute TensorFlow programs using a cluster of
TensorFlow servers.

[View Tutorial](distributed/index.md)


## Threading and Queues

This tutorial describes the various constructs implemented by TensorFlow
to facilitate asynchronous and concurrent training.

[View Tutorial](threading_and_queues/index.md)


## Adding a New Op

TensorFlow already has a large suite of node operations from which you can
compose in your graph, but here are the details of how to add you own custom Op.

[View Tutorial](adding_an_op/index.md)


## How to write TensorFlow code

TensorFlow Style Guide is set of style decisions that both developers
and users of TensorFlow should follow to increase the readability of their code,
reduce the number of errors, and promote consistency.

[View Style Guide](style_guide.md)


## Writing Documentation

TensorFlow's documentation is largely generated from its source code. Here is an
introduction to the formats we use, a style guide, and instructions on how to
build updated documentation from the source.

[View Tutorial](documentation/index.md)


## Custom Data Readers

If you have a sizable custom data set, you may want to consider extending
TensorFlow to read your data directly in it's native format.  Here's how.

[View Tutorial](new_data_formats/index.md)


## Using GPUs

This tutorial describes how to construct and execute models on GPU(s).

[View Tutorial](using_gpu/index.md)


## Sharing Variables

When deploying large models on multiple GPUs, or when unrolling complex LSTMs
or RNNs, it is often necessary to access the same Variable objects from
different locations in the model construction code.

The "Variable Scope" mechanism is designed to facilitate that.

[View Tutorial](variable_scope/index.md)

## A Tool Developer's Guide to TensorFlow Model Files

If you're developing a tool to load, analyze, or manipulate TensorFlow model
files, it's useful to understand a bit about the format in which they're stored.
This guide covers the details of the saved model format.

[View Tutorial](../how_tos/tool_developers/index.md)

## How to Retrain Inception using Transfer Learning

Training a full object recognition model like Inception takes a long time and a
lot of images. This example shows how to use the technique of transfer learning
to retrain just the final layer of a fully-trained model to recognize new
categories of objects, which is a lot faster and easier than completely
retraining a new model.

[View Tutorial](../how_tos/image_retraining/index.md)

## How to Export and Import a Model

This tutorial describes how to export everything pertaining to a running
model and import it later for various purposes.

[View Tutorial](../how_tos/meta_graph/index.md)

## How to Quantize Neural Networks with TensorFlow

This guide shows how you can convert a float model into one using eight-bit
quantized parameters and calculations. It also describes how the quantization
process works under the hood.

[View Tutorial](../how_tos/quantization/index.md)

## How to run TensorFlow on Hadoop

This tutorial shows how to read and write HDFS files, and will later describe
running on cluster managers.

[View Tutorial](../how_tos/hadoop/index.md)

## TensorFlow in other languages

This guide describes how TensorFlow features can be provided in other
programming languages.

[View Tutorial](language_bindings/index.md)
