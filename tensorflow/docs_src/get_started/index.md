# Getting Started

TensorFlow is a tool for machine learning. While it contains a wide range of
functionality, it is mainly designed for deep neural network models.

The fastest way to build a fully-featured model trained on your data is to use
TensorFlow's high-level API. In the following examples, we will use the
high-level API on the classic [Iris dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).
We will train a model that predicts what species a flower is based on its
characteristics, and along the way get a quick introduction to the basic tasks
in TensorFlow using Estimators.

This tutorial is divided into the following parts:

  * @{$get_started/premade_estimators}, which shows you
    how to quickly setup prebuilt models to train on in-memory data.
  * @{$get_started/checkpoints}, which shows you how to save training progress,
    and resume where you left off.
  * @{$get_started/feature_columns}, which shows how an
    Estimator can handle a variety of input data types without changes to the
    model.
  * @{$get_started/datasets_quickstart}, which is a minimal introduction to
    the TensorFlow's input pipelines.
  * @{$get_started/custom_estimators}, which demonstrates how
    to build and train models you design yourself.

For more advanced users:

  * The @{$low_level_intro$Low Level Introduction} demonstrates how to use
    tensorflow outside of the Estimator framework, for debugging and
    experimentation.
  * The remainder of the @{$programmers_guide$Programmer's Guide} contains
    in-depth guides to various major components of TensorFlow.
  * The @{$tutorials$Tutorials} provide walkthroughs of a variety of
    TensorFlow models.
