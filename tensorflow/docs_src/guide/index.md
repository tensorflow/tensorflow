# TensorFlow Guide

The documents in this unit dive into the details of how TensorFlow
works. The units are as follows:

## High Level APIs

  * [Keras](../guide/keras.md), TensorFlow's high-level API for building and
    training deep learning models.
  * [Eager Execution](../guide/eager.md), an API for writing TensorFlow code
    imperatively, like you would use Numpy.
  * [Importing Data](../guide/datasets.md), easy input pipelines to bring your data into
    your TensorFlow program.
  * [Estimators](../guide/estimators.md), a high-level API that provides
    fully-packaged models ready for large-scale training and production.

## Estimators

* [Premade Estimators](../guide/premade_estimators.md), the basics of premade Estimators.
* [Checkpoints](../guide/checkpoints.md), save training progress and resume where you left off.
* [Feature Columns](../guide/feature_columns.md), handle a variety of input data types without changes to the model.
* [Datasets for Estimators](../guide/datasets_for_estimators.md), use `tf.data` to input data.
* [Creating Custom Estimators](../guide/custom_estimators.md), write your own Estimator.

## Accelerators

  * [Using GPUs](../guide/using_gpu.md) explains how TensorFlow assigns operations to
    devices and how you can change the arrangement manually.
  * [Using TPUs](../guide/using_tpu.md) explains how to modify `Estimator` programs to run on a TPU.

## Low Level APIs

  * [Introduction](../guide/low_level_intro.md), which introduces the
    basics of how you can use TensorFlow outside of the high Level APIs.
  * [Tensors](../guide/tensors.md), which explains how to create,
    manipulate, and access Tensors--the fundamental object in TensorFlow.
  * [Variables](../guide/variables.md), which details how
    to represent shared, persistent state in your program.
  * [Graphs and Sessions](../guide/graphs.md), which explains:
      * dataflow graphs, which are TensorFlow's representation of computations
        as dependencies between operations.
      * sessions, which are TensorFlow's mechanism for running dataflow graphs
        across one or more local or remote devices.
    If you are programming with the low-level TensorFlow API, this unit
    is essential. If you are programming with a high-level TensorFlow API
    such as Estimators or Keras, the high-level API creates and manages
    graphs and sessions for you, but understanding graphs and sessions
    can still be helpful.
  * [Save and Restore](../guide/saved_model.md), which
    explains how to save and restore variables and models.

## ML Concepts

  * [Embeddings](../guide/embedding.md), which introduces the concept
    of embeddings, provides a simple example of training an embedding in
    TensorFlow, and explains how to view embeddings with the TensorBoard
    Embedding Projector.

## Debugging

  * [TensorFlow Debugger](../guide/debugger.md), which
    explains how to use the TensorFlow debugger (tfdbg).

## TensorBoard

TensorBoard is a utility to visualize different aspects of machine learning.
The following guides explain how to use TensorBoard:

  * [TensorBoard: Visualizing Learning](../guide/summaries_and_tensorboard.md),
    which introduces TensorBoard.
  * [TensorBoard: Graph Visualization](../guide/graph_viz.md), which
    explains how to visualize the computational graph.
  * [TensorBoard Histogram Dashboard](../guide/tensorboard_histograms.md) which demonstrates the how to
    use TensorBoard's histogram dashboard.


## Misc

  * [TensorFlow Version Compatibility](../guide/version_compat.md),
    which explains backward compatibility guarantees and non-guarantees.
  * [Frequently Asked Questions](../guide/faq.md), which contains frequently asked
    questions about TensorFlow.
