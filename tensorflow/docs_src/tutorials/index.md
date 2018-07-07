# Tutorials


This section contains tutorials demonstrating how to do specific tasks
in TensorFlow.  If you are new to TensorFlow, we recommend reading
[Get Started with TensorFlow](/get_started/).

## Images

These tutorials cover different aspects of image recognition:

  * @{$layers$MNIST}, which introduces convolutional neural networks (CNNs) and
    demonstrates how to build a CNN in TensorFlow.
  * @{$image_recognition}, which introduces the field of image recognition and
    uses a pre-trained model (Inception) for recognizing images.
  * @{$image_retraining}, which has a wonderfully self-explanatory title.
  * @{$deep_cnn}, which demonstrates how to build a small CNN for recognizing
    images.  This tutorial is aimed at advanced TensorFlow users.


## Sequences

These tutorials focus on machine learning problems dealing with sequence data.

  * @{$recurrent}, which demonstrates how to use a
    recurrent neural network to predict the next word in a sentence.
  * @{$seq2seq}, which demonstrates how to use a
    sequence-to-sequence model to translate text from English to French.
  * @{$recurrent_quickdraw}
    builds a classification model for drawings, directly from the sequence of
    pen strokes.
  * @{$audio_recognition}, which shows how to
    build a basic speech recognition network.

## Data representation

These tutorials demonstrate various data representations that can be used in
TensorFlow.

  * @{$wide}, uses
    @{tf.feature_column$feature columns} to feed a variety of data types
    to linear model, to solve a classification problem.
  * @{$wide_and_deep}, builds on the
    above linear model tutorial, adding a deep feed-forward neural network
    component and a DNN-compatible data representation.
  * @{$word2vec}, which demonstrates how to
    create an embedding for words.
  * @{$kernel_methods},
    which shows how to improve the quality of a linear model by using explicit
    kernel mappings.

## Non Machine Learning

Although TensorFlow specializes in machine learning, the core of TensorFlow is
a powerful numeric computation system which you can also use to solve other
kinds of math problems.  For example:

  * @{$mandelbrot}
  * @{$pdes}
