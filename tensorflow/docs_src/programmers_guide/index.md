# Programmer's Guide

The documents in this unit dive into the details of how TensorFlow
works. The units are as follows:

## High Level APIs

  * @{$programmers_guide/eager}, which is the easiest way to use TensorFlow.
  * @{$programmers_guide/estimators}, which introduces a high-level
    TensorFlow API that greatly simplifies ML programming.
  * @{$programmers_guide/datasets}, which explains how to
    set up data pipelines to read data sets into your TensorFlow program.

## Low Level APIs

  * @{$programmers_guide/low_level_intro}, which introduces the
    basics of how you can use TensorFlow outside of the high Level APIs.
  * @{$programmers_guide/tensors}, which explains how to create,
    manipulate, and access Tensors--the fundamental object in TensorFlow.
  * @{$programmers_guide/variables}, which details how
    to represent shared, persistent state in your program.
  * @{$programmers_guide/graphs}, which explains:
      * dataflow graphs, which are TensorFlow's representation of computations
        as dependencies between operations.
      * sessions, which are TensorFlow's mechanism for running dataflow graphs
        across one or more local or remote devices.
    If you are programming with the low-level TensorFlow API, this unit
    is essential. If you are programming with a high-level TensorFlow API
    such as Estimators or Keras, the high-level API creates and manages
    graphs and sessions for you, but understanding graphs and sessions
    can still be helpful.
  * @{$programmers_guide/saved_model}, which
    explains how to save and restore variables and models.

## Accelerators

  * @{$using_gpu} explains how TensorFlow assigns operations to
    devices and how you can change the arrangement manually.
  * @{$using_tpu} explains how to modify `Estimator` programs to run on a TPU.


## ML Concepts

  * @{$programmers_guide/embedding}, which introduces the concept
    of embeddings, provides a simple example of training an embedding in
    TensorFlow, and explains how to view embeddings with the TensorBoard
    Embedding Projector.

## Debugging

  * @{$programmers_guide/debugger}, which
    explains how to use the TensorFlow debugger (tfdbg).

## TensorBoard

TensorBoard is a utility to visualize different aspects of machine learning.
The following guides explain how to use TensorBoard:

  * @{$programmers_guide/summaries_and_tensorboard},
    which introduces TensorBoard.
  * @{$programmers_guide/graph_viz}, which
    explains how to visualize the computational graph.
  * @{$programmers_guide/tensorboard_histograms} which demonstrates the how to
    use TensorBoard's histogram dashboard.


## Misc

  * @{$programmers_guide/version_compat},
    which explains backward compatibility guarantees and non-guarantees.
  * @{$programmers_guide/faq}, which contains frequently asked
    questions about TensorFlow.
