# TensorFlow Guide

The documents in this unit dive into the details of how TensorFlow
works. The units are as follows:

## High Level APIs

  * @{$guide/keras}, TensorFlow's high-level API for building and
    training deep learning models.
  * @{$guide/eager}, an API for writing TensorFlow code
    imperatively, like you would use Numpy.
  * @{$guide/datasets}, easy input pipelines to bring your data into
    your TensorFlow program.
  * @{$guide/estimators}, a high-level API that provides
    fully-packaged models ready for large-scale training and production.

## Estimators

* @{$premade_estimators}, the basics of premade Estimators.
* @{$checkpoints}, save training progress and resume where you left off.
* @{$feature_columns}, handle a variety of input data types without changes to the model.
* @{$datasets_for_estimators}, use `tf.data` to input data.
* @{$custom_estimators}, write your own Estimator.

## Accelerators

  * @{$using_gpu} explains how TensorFlow assigns operations to
    devices and how you can change the arrangement manually.
  * @{$using_tpu} explains how to modify `Estimator` programs to run on a TPU.

## Low Level APIs

  * @{$guide/low_level_intro}, which introduces the
    basics of how you can use TensorFlow outside of the high Level APIs.
  * @{$guide/tensors}, which explains how to create,
    manipulate, and access Tensors--the fundamental object in TensorFlow.
  * @{$guide/variables}, which details how
    to represent shared, persistent state in your program.
  * @{$guide/graphs}, which explains:
      * dataflow graphs, which are TensorFlow's representation of computations
        as dependencies between operations.
      * sessions, which are TensorFlow's mechanism for running dataflow graphs
        across one or more local or remote devices.
    If you are programming with the low-level TensorFlow API, this unit
    is essential. If you are programming with a high-level TensorFlow API
    such as Estimators or Keras, the high-level API creates and manages
    graphs and sessions for you, but understanding graphs and sessions
    can still be helpful.
  * @{$guide/saved_model}, which
    explains how to save and restore variables and models.

## ML Concepts

  * @{$guide/embedding}, which introduces the concept
    of embeddings, provides a simple example of training an embedding in
    TensorFlow, and explains how to view embeddings with the TensorBoard
    Embedding Projector.

## Debugging

  * @{$guide/debugger}, which
    explains how to use the TensorFlow debugger (tfdbg).

## TensorBoard

TensorBoard is a utility to visualize different aspects of machine learning.
The following guides explain how to use TensorBoard:

  * @{$guide/summaries_and_tensorboard},
    which introduces TensorBoard.
  * @{$guide/graph_viz}, which
    explains how to visualize the computational graph.
  * @{$guide/tensorboard_histograms} which demonstrates the how to
    use TensorBoard's histogram dashboard.


## Misc

  * @{$guide/version_compat},
    which explains backward compatibility guarantees and non-guarantees.
  * @{$guide/faq}, which contains frequently asked
    questions about TensorFlow.
