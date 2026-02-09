# API reference for text and NLP libraries.

## KerasNLP

[API reference](https://keras.io/api/keras_nlp/){: .external}

The easiest way to get started processing text in TensorFlow is to use KerasNLP,
a natural language processing library that provides modular components with
state-of-the-art preset weights and architectures. You can use KerasNLP
components out-of-the-box or customize them as needed. KerasNLP
emphasizes in-graph computation for all workflows, so you can expect easy
productionization using the TensorFlow ecosystem.

To install KerasNLP, see
[Installation](https://keras.io/keras_nlp/#installation){: .external}.

## TensorFlow Text

[API reference](https://www.tensorflow.org/text/api_docs/python/text)

The `tensorflow_text` package provides a collection of text related classes and
ops ready to use with TensorFlow. The library can perform the preprocessing
regularly required by text-based models, and includes other features useful for
sequence modeling not provided by core TensorFlow.

For installation details, refer to the
[the guide](https://www.tensorflow.org/text/guide/tf_text_intro)

## TensorFlow Models - NLP

[API reference](https://tensorflow.org/api_docs/python/tfm/nlp){: .external}

The
[TensorFlow Models repository](https://github.com/tensorflow/models){: .external}
provides implementations of state-of-the-art (SOTA) models. The
`tensorflow-models-official` pip package includes many high-level functions and
classes for building SOTA NLP models including `nlp.layers`, `nlp.losses`,
`nlp.models` and `nlp.tasks`.

You can install the package with `pip`:

```
$ pip install tensorflow-models-official  # For the latest release
$ #or
$ pip install tf-models-nightly # For the nightly build
```

The NLP functionality is available in the `tfm.nlp` submodule.

```
import tensorflow_models as tfm
tfm.nlp
```
