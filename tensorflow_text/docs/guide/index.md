# TensorFlow text processing guide

The TensorFlow text processing guide documents libraries and workflows for
natural language processing (NLP) and introduces important concepts for working
with text.

## KerasNLP

KerasNLP is a high-level natural language processing (NLP) library that includes
all the latest Transformer-based models as well as lower-level tokenization
utilities. It's the recommended solution for most NLP use cases.

* [Getting Started with KerasNLP](https://keras.io/guides/keras_nlp/getting_started/){: .external}:
  Learn KerasNLP by performing sentiment analysis at progressive levels of
  complexity, from using a pre-trained model to building your own Transformer
  from scratch.

## `tf.strings`

The `tf.strings` module provides operations for working with string Tensors.

* [Unicode strings](https://tensorflow.org/text/guide/unicode):
  Represent Unicode strings in TensorFlow and manipulate them using Unicode
  equivalents of standard string ops.

## TensorFlow Text

If you need access to lower-level text processing tools, you can use TensorFlow
Text. TensorFlow Text provides a collection of ops and libraries to help you
work with input in text form such as raw text strings or documents.

* [Introduction to TensorFlow Text](https://tensorflow.org/text/guide/tf_text_intro):
  Learn how to install TensorFlow Text or build it from source.
* [Converting TensorFlow Text operators to TensorFlow Lite](https://tensorflow.org/text/guide/text_tf_lite):
  Convert a TensorFlow Text model to TensorFlow Lite for deployment to mobile,
  embedded, and IoT devices.

### Pre-processing

* [BERT Preprocessing with TF Text](https://tensorflow.org/text/guide/bert_preprocessing_guide):
  Use TensorFlow Text preprocessing ops to transform text data into inputs for
  BERT.
* [Tokenizing with TF Text](https://tensorflow.org/text/guide/tokenizers):
  Understand the tokenization options provided by TensorFlow Text. Learn when
  you might want to use one option over another, and how these tokenizers are
  called from within your model.
* [Subword tokenizers](https://tensorflow.org/text/guide/subwords_tokenizer):
  Generate a subword vocabulary from a dataset, and use it to build a
  [`text.BertTokenizer`](https://www.tensorflow.org/text/api_docs/python/text/BertTokenizer)
  from the vocabulary.

## TensorFlow models &ndash; NLP

The TensorFlow Models - NLP library provides Keras primitives that can be
assembled into Transformer-based models, and scaffold classes that enable easy
experimentation with novel architectures.

* [Introduction to the TensorFlow Models NLP library](https://tensorflow.org/tfmodels/nlp):
  Build Transformer-based models for common NLP tasks including pre-training,
  span labelling, and classification using building blocks from the
  [NLP modeling library](https://github.com/tensorflow/models/tree/master/official/nlp/modeling).
* [Customizing a Transformer Encoder](https://tensorflow.org/tfmodels/nlp/customize_encoder):
  Customize
  [`tfm.nlp.networks.EncoderScaffold`](https://www.tensorflow.org/api_docs/python/tfm/nlp/networks/EncoderScaffold),
  a bi-directional Transformer-based encoder network scaffold, to employ new
  network architectures.