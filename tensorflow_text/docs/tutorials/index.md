# TensorFlow text processing tutorials

The TensorFlow text processing tutorials provide step-by-step instructions for
solving common text and natural language processing (NLP) problems.

TensorFlow provides two solutions for text and natural language processing:
KerasNLP and TensorFlow Text. KerasNLP is a high-level NLP library that includes
all the latest Transformer-based models as well as lower-level tokenization
utilities. It's the recommended solution for most NLP use cases.

If you need access to lower-level text processing tools, you can use
TensorFlow Text. TensorFlow Text provides a collection of ops and libraries to
help you work with input in text form such as raw text strings or documents.

## KerasNLP

* [Getting Started with KerasNLP](https://keras.io/guides/keras_nlp/getting_started/){: .external}:
  Learn KerasNLP by performing sentiment analysis at progressive levels of
  complexity, from using a pre-trained model to building your own Transformer
  from scratch.

## Text generation

* [Text generation with an RNN](https://tensorflow.org/text/tutorials/text_generation):
  Generate text using a character-based RNN and a dataset of Shakespeare's
  writing.
* [Neural machine translation with attention](https://tensorflow.org/text/tutorials/nmt_with_attention):
  Train a sequence-to-sequence (seq2seq) model for Spanish-to-English
  translation.
* [Neural machine translation with a Transformer and Keras](https://tensorflow.org/text/tutorials/transformer):
  Create and train a sequence-to-sequence Transformer model to translate
  Portuguese into English.
* [Image captioning with visual attention](https://tensorflow.org/text/tutorials/image_captioning):
  Generate image captions using a Transformer-decoder model built with attention
  layers.

## Text classification

* [Classify text with BERT](https://tensorflow.org/text/tutorials/classify_text_with_bert):
  Fine-tune BERT to perform sentiment analysis on a dataset of plain-text IMDb
  movie reviews.
* [Text classification with an RNN](https://tensorflow.org/text/tutorials/text_classification_rnn):
  Train an RNN to perform sentiment analysis on IMDb movie reviews.
* [TF.Text Metrics](https://tensorflow.org/text/tutorials/text_similarity):
  Learn about the metrics available through TensorFlow Text. The library
  contains implementations of text-similarity metrics such as ROUGE-L, which can
  be used for automatic evaluation of text generation models.

## NLP with BERT

* [Solve GLUE tasks using BERT on TPU](https://tensorflow.org/text/tutorials/bert_glue):
  Learn how to fine-tune BERT for tasks from the
  [GLUE benchmark](https://gluebenchmark.com/).
* [Fine-tuning a BERT model](https://tensorflow.org/tfmodels/nlp/fine_tune_bert):
  Fine-tune a BERT model using
  [TensorFlow Model Garden](https://github.com/tensorflow/models).
* [Uncertainty-aware Deep Language Learning with BERT-SNGP](https://tensorflow.org/text/tutorials/uncertainty_quantification_with_sngp_bert):
  Apply [SNGP](https://arxiv.org/abs/2006.10108) to a natural language
  understanding (NLU) task. Building on a BERT encoder, you'll improve the NLU
  model's ability to detect out-of-scope queries.

## Embeddings

* [Word embeddings](https://tensorflow.org/text/guide/word_embeddings):
  Train your own word embeddings using a simple Keras model for a sentiment
  classification task, and then visualize them using the
  [Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
* [Warm-start embedding layer matrix](https://tensorflow.org/text/tutorials/warmstart_embedding_matrix):
  Learn how to "warm-start" training for a text sentiment classification model.
* [word2vec](https://tensorflow.org/text/tutorials/word2vec): Train a word2vec
  model on a small dataset and visualize the trained embeddings in the
  [Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).