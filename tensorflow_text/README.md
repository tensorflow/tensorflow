<div align="center">
  <img src="https://raw.githubusercontent.com/tensorflow/text/master/docs/include/tftext.png" width="60%"><br><br>
</div>

-----------------

[![PyPI version](https://img.shields.io/pypi/v/tensorflow-text)](https://badge.fury.io/py/tensorflow-text)
[![PyPI nightly version](https://img.shields.io/pypi/v/tensorflow-text-nightly?color=informational&label=pypi%20%40%20nightly)](https://badge.fury.io/py/tensorflow-text-nightly)
[![PyPI Python version](https://img.shields.io/pypi/pyversions/tensorflow-text)](https://pypi.org/project/tensorflow-text/)
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://github.com/tensorflow/text/blob/master/docs/api_docs/python/index.md)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- TODO(broken):  Uncomment when badges are made public.
### Continuous Integration Test Status

| Build      | Status |
| ---             | ---    |
| **Linux**   | [![Status](https://storage.googleapis.com/tf-text-badges/ubuntu-gpu-py3.svg)] |
| **MacOS**   | [![Status](https://storage.googleapis.com/tf-text-badges/ubuntu-gpu-py3.svg)] |
| **Windows**   | [![Status](https://storage.googleapis.com/tf-text-badges/ubuntu-gpu-py3.svg)] |
-->

# TensorFlow Text - Text processing in Tensorflow

**IMPORTANT**: When installing TF Text with `pip install`, please note the
version of TensorFlow you are running, as you should specify the corresponding
minor version of TF Text (eg. for tensorflow==2.3.x use tensorflow_text==2.3.x).

## INDEX
* [Introduction](#introduction)
* [Documentation](#documentation)
* [Unicode](#unicode)
* [Normalization](#normalization)
* [Tokenization](#tokenization)
  * [Whitespace Tokenizer](#whitespacetokenizer)
  * [UnicodeScript Tokenizer](#unicodescripttokenizer)
  * [Unicode split](#unicode-split)
  * [Offsets](#offsets)
  * [TF.Data Example](#tfdata-example)
  * [Keras API](#keras-api)
* [Other Text Ops](#other-text-ops)
  * [Wordshape](#wordshape)
  * [N-grams & Sliding Window](#n-grams--sliding-window)
* [Installation](#installation)
  * [Install using PIP](#install-using-pip)
  * [Build from source steps:](#build-from-source-steps)

## Introduction

TensorFlow Text provides a collection of text related classes and ops ready to
use with TensorFlow 2.0. The library can perform the preprocessing regularly
required by text-based models, and includes other features useful for sequence
modeling not provided by core TensorFlow.

The benefit of using these ops in your text preprocessing is that they are done
in the TensorFlow graph. You do not need to worry about tokenization in
training being different than the tokenization at inference, or managing
preprocessing scripts.

## Documentation

Please visit [http://tensorflow.org/text](http://tensorflow.org/text) for all
documentation. This site includes API docs, guides for working with TensorFlow
Text, as well as tutorials for building specific models.

## Unicode

Most ops expect that the strings are in UTF-8. If you're using a different
encoding, you can use the core tensorflow transcode op to transcode into UTF-8.
You can also use the same op to coerce your string to structurally valid UTF-8
if your input could be invalid.

```python
docs = tf.constant([u'Everything not saved will be lost.'.encode('UTF-16-BE'),
                    u'Sad☹'.encode('UTF-16-BE')])
utf8_docs = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE',
                                         output_encoding='UTF-8')
```

## Normalization

When dealing with different sources of text, it's important that the same words
are recognized to be identical. A common technique for case-insensitive matching
in Unicode is case folding (similar to lower-casing). (Note that case folding
internally applies NFKC normalization.)

We also provide Unicode normalization ops for transforming strings into a
canonical representation of characters, with Normalization Form KC being the
default ([NFKC](http://unicode.org/reports/tr15/)).

```python
print(text.case_fold_utf8(['Everything not saved will be lost.']))
print(text.normalize_utf8(['Äffin']))
print(text.normalize_utf8(['Äffin'], 'nfkd'))
```

```sh
tf.Tensor(['everything not saved will be lost.'], shape=(1,), dtype=string)
tf.Tensor(['\xc3\x84ffin'], shape=(1,), dtype=string)
tf.Tensor(['A\xcc\x88ffin'], shape=(1,), dtype=string)
```

## Tokenization

Tokenization is the process of breaking up a string into tokens. Commonly, these
tokens are words, numbers, and/or punctuation.

The main interfaces are `Tokenizer` and `TokenizerWithOffsets` which each have a
single method `tokenize` and `tokenizeWithOffsets` respectively. There are
multiple implementing tokenizers available now. Each of these implement
`TokenizerWithOffsets` (which extends `Tokenizer`) which includes an option for
getting byte offsets into the original string. This allows the caller to know
the bytes in the original string the token was created from.

All of the tokenizers return RaggedTensors with the inner-most dimension of
tokens mapping to the original individual strings. As a result, the resulting
shape's rank is increased by one. Please review the ragged tensor guide if you
are unfamiliar with them. https://www.tensorflow.org/guide/ragged_tensor

### WhitespaceTokenizer

This is a basic tokenizer that splits UTF-8 strings on ICU defined whitespace
characters (eg. space, tab, new line).

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost.'], ['Sad\xe2\x98\xb9']]
```

### UnicodeScriptTokenizer

This tokenizer splits UTF-8 strings based on Unicode script boundaries. The
script codes used correspond to International Components for Unicode (ICU)
UScriptCode values. See: http://icu-project.org/apiref/icu4c/uscript_8h.html

In practice, this is similar to the `WhitespaceTokenizer` with the most apparent
difference being that it will split punctuation (USCRIPT_COMMON) from language
texts (eg. USCRIPT_LATIN, USCRIPT_CYRILLIC, etc) while also separating language
texts from each other.

```python
tokenizer = text.UnicodeScriptTokenizer()
tokens = tokenizer.tokenize(['everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost', '.'],
 ['Sad', '\xe2\x98\xb9']]
```

### Unicode split

When tokenizing languages without whitespace to segment words, it is common to
just split by character, which can be accomplished using the
[unicode_split](https://www.tensorflow.org/api_docs/python/tf/strings/unicode_split)
op found in core.

```python
tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
print(tokens.to_list())
```

```sh
[['\xe4\xbb\x85', '\xe4\xbb\x8a', '\xe5\xb9\xb4', '\xe5\x89\x8d']]
```

### Offsets

When tokenizing strings, it is often desired to know where in the original
string the token originated from. For this reason, each tokenizer which
implements `TokenizerWithOffsets` has a *tokenize_with_offsets* method that will
return the byte offsets along with the tokens. The start_offsets lists the bytes
in the original string each token starts at (inclusive), and the end_offsets
lists the bytes where each token ends at (exclusive, i.e., first byte *after*
the token).

```python
tokenizer = text.UnicodeScriptTokenizer()
(tokens, start_offsets, end_offsets) = tokenizer.tokenize_with_offsets(
    ['everything not saved will be lost.', u'Sad☹'.encode('UTF-8')])
print(tokens.to_list())
print(start_offsets.to_list())
print(end_offsets.to_list())
```

```sh
[['everything', 'not', 'saved', 'will', 'be', 'lost', '.'],
 ['Sad', '\xe2\x98\xb9']]
[[0, 11, 15, 21, 26, 29, 33], [0, 3]]
[[10, 14, 20, 25, 28, 33, 34], [3, 6]]
```

### TF.Data Example

Tokenizers work as expected with the tf.data API. A simple example is provided
below.

```python
docs = tf.data.Dataset.from_tensor_slices([['Never tell me the odds.'],
                                           ["It's a trap!"]])
tokenizer = text.WhitespaceTokenizer()
tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
iterator = tokenized_docs.make_one_shot_iterator()
print(iterator.get_next().to_list())
print(iterator.get_next().to_list())
```

```sh
[['Never', 'tell', 'me', 'the', 'odds.']]
[["It's", 'a', 'trap!']]
```

### Keras API

When you use different tokenizers and ops to preprocess your data, the resulting
outputs are Ragged Tensors. The Keras API makes it easy now to train a model
using Ragged Tensors without having to worry about padding or masking the data,
by either using the ToDense layer which handles all of these for you or relying
on Keras built-in layers support for natively working on ragged data.

```python
model = tf.keras.Sequential([
  tf.keras.layers.InputLayer(input_shape=(None,), dtype='int32', ragged=True)
  text.keras.layers.ToDense(pad_value=0, mask=True),
  tf.keras.layers.Embedding(100, 16),
  tf.keras.layers.LSTM(32),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
```

## Other Text Ops

TF.Text packages other useful preprocessing ops. We will review a couple below.

### Wordshape

A common feature used in some natural language understanding models is to see
if the text string has a certain property. For example, a sentence breaking
model might contain features which check for word capitalization or if a
punctuation character is at the end of a string.

Wordshape defines a variety of useful regular expression based helper functions
for matching various relevant patterns in your input text. Here are a few
examples.

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])

# Is capitalized?
f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
# Are all letters uppercased?
f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
# Does the token contain punctuation?
f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
# Is the token a number?
f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

print(f1.to_list())
print(f2.to_list())
print(f3.to_list())
print(f4.to_list())
```

```sh
[[True, False, False, False, False, False], [True]]
[[False, False, False, False, False, False], [False]]
[[False, False, False, False, False, True], [True]]
[[False, False, False, False, False, False], [False]]
```

### N-grams & Sliding Window

N-grams are sequential words given a sliding window size of *n*. When combining
the tokens, there are three reduction mechanisms supported. For text, you would
want to use `Reduction.STRING_JOIN` which appends the strings to each other.
The default separator character is a space, but this can be changed with the
string_separater argument.

The other two reduction methods are most often used with numerical values, and
these are `Reduction.SUM` and `Reduction.MEAN`.

```python
tokenizer = text.WhitespaceTokenizer()
tokens = tokenizer.tokenize(['Everything not saved will be lost.',
                             u'Sad☹'.encode('UTF-8')])

# Ngrams, in this case bi-gram (n = 2)
bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

print(bigrams.to_list())
```

```sh
[['Everything not', 'not saved', 'saved will', 'will be', 'be lost.'], []]
```

## Installation

### Install using PIP

When installing TF Text with `pip install`, please note the version
of TensorFlow you are running, as you should specify the corresponding version
of TF Text. For example, if you're using TF 2.0, install the 2.0 version of TF
Text, and if you're using TF 1.15, install the 1.15 version of TF Text.

```bash
pip install -U tensorflow-text==<version>
```

### A note about different operating system packages

After version 2.10, we will only be providing pip packages for Linux x86_64 and
Intel-based Macs. TensorFlow Text has always leveraged the release
infrastructure of the core TensorFlow package to more easily maintain compatible
releases with minimal maintenance, allowing the team to focus on TF Text itself
and contributions to other parts of the TensorFlow ecosystem.

For other systems like Windows, Aarch64, and Apple Macs, TensorFlow relies on
[build collaborators](https://blog.tensorflow.org/2022/09/announcing-tensorflow-official-build-collaborators.html),
and so we will not be providing packages for them. However, we will continue to
accept PRs to make building for these OSs easy for users, and will try to point
to community efforts related to them.


### Build from source steps:

Note that TF Text needs to be built in the same environment as TensorFlow. Thus,
if you manually build TF Text, it is highly recommended that you also build
TensorFlow.

If building on MacOS, you must have coreutils installed. It is probably easiest
to do with Homebrew.

1. [build and install TensorFlow](https://www.tensorflow.org/install/source).
1. Clone the TF Text repo:
   ```Shell
   git clone https://github.com/tensorflow/text.git
   cd text
   ```
1. Run the build script to create a pip package:
   ```Shell
   ./oss_scripts/run_build.sh
   ```
   After this step, there should be a `*.whl` file in current directory. File name similar to `tensorflow_text-2.5.0rc0-cp38-cp38-linux_x86_64.whl`.
1. Install the package to environment:
   ```Shell
   pip install ./tensorflow_text-*-*-*-os_platform.whl
   ```

### Build or test using TensorFlow's SIG docker image:

1.  Pull image from
    [Tensorflow SIG docker builds](https://hub.docker.com/r/tensorflow/build/tags).

1.  Run a container based with the pulled image and create a bash session.
    This can be done by running `docker run -it {image_name} bash`. <br />
    `{image_name}` can be any name with `{tf_verison}-python{python_version}` format.
    An example for python 3.10 and TF version 2.10 :- `2.10-python3.10`.
1.  Clone the TF-Text Github repository inside container:  `git clone https://github.com/tensorflow/text.git`. <br />
    Once cloned, change to the working directory using `cd text/`.
1.  Run the configuration script(s): `./oss_scripts/configure.sh` and `./oss_scripts/prepare_tf_dep.sh`. <br />
    This will update bazel and TF dependencies to installed tensorflow in the container.
1.  To run the tests, use the bazel command: `bazel test --test_output=errors tensorflow_text:all`. This will run all the tests declared in the `BUILD` file. <br />
    To run a specific test, modify the above command replacing `:all` with the test name (for example `:fast_bert_normalizer`).
    
1.  Build the pip package/wheel: \
    `bazel build --config=release_cpu_linux
    oss_scripts/pip_package:build_pip_package` \
    `./bazel-bin/oss_scripts/pip_package/build_pip_package
    /{wheel_dir}` <br />

    Once the build is complete, you should see the wheel available under
    `{wheel_dir}` directory.
