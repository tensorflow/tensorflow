# Get Started

If you are new to machine learning, we recommend taking the following online
course prior to diving into TensorFlow documentation:

  * [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/),
    which introduces machine learning concepts and encourages experimentation
    with existing TensorFlow code.

TensorFlow is a tool for machine learning. While it contains a wide range of
functionality, TensorFlow is mainly designed for deep neural network models.

TensorFlow provides many APIs. This section focuses on the high-level APIs.
If you are new to TensorFlow, begin by reading one of the following documents:

  * @{$get_started/eager} is for machine learning beginners and uses
    @{$programmers_guide/eager}.
  * @{$get_started/get_started_for_beginners} is also for machine learning
    beginners and uses @{$programmers_guide/graphs}.
  * @{$get_started/premade_estimators} assumes some machine learning background
    and uses an @{tf.estimator.Estimator$Estimator}.

Then, read the following documents, which demonstrate the key features
in the high-level APIs:

  * @{$get_started/checkpoints}, which explains how to save training progress
    and resume where you left off.
  * @{$get_started/feature_columns}, which shows how an
    Estimator can handle a variety of input data types without changes to the
    model.
  * @{$get_started/datasets_quickstart}, which introduces TensorFlow's
    input pipelines.
  * @{$get_started/custom_estimators}, which demonstrates how
    to build and train models you design yourself.

For more advanced users:

  * The @{$low_level_intro$Low Level Introduction} demonstrates how to use
    TensorFlow outside of the Estimator framework, for debugging and
    experimentation.
  * The @{$programmers_guide$Programmer's Guide} details major
    TensorFlow components.
  * The @{$tutorials$Tutorials} provide walkthroughs of a variety of
    TensorFlow models.
