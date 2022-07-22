# Integrate text embedders.

Text embedders allow embedding text into a high-dimensional feature vector
representing its semantic meaning, which can then be compared with the feature
vector of other texts to evaluate their semantic similarity.

As opposed to
[text search](https://www.tensorflow.org/lite/inference_with_metadata/task_library/text_searcher),
the text embedder allows computing the similarity between texts on-the-fly
instead of searching through a predefined index built from a corpus.

Use the Task Library `TextEmbedder` API to deploy your custom text embedder into
your mobile apps.

## Key features of the TextEmbedder API

*   Input text processing, including in-graph or out-of-graph
    [Wordpiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/bert_tokenizer.h)
    or
    [Sentencepiece](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/text/tokenizers/sentencepiece_tokenizer.h)
    tokenizations on input text.

*   Built-in utility function to compute the
    [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between
    feature vectors.

## Supported text embedder models

The following models are guaranteed to be compatible with the `TextEmbedder`
API.

*   The
    [Universal Sentence Encoder TFLite model from TensorFlow Hub](https://tfhub.dev/google/lite-model/universal-sentence-encoder-qa-ondevice/1)

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in C++

```c++
// Initialization.
TextEmbedderOptions options:
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
std::unique_ptr<TextEmbedder> text_embedder = TextEmbedder::CreateFromOptions(options).value();

// Run inference on two texts.
const EmbeddingResult result_1 = text_embedder->Embed(text_1);
const EmbeddingResult result_2 = text_embedder->Embed(text_2);

// Compute cosine similarity.
double similarity = TextEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector()
    result_2.embeddings[0].feature_vector());
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/text_embedder.h)
for more options to configure `TextEmbedder`.

## Run inference in Python

### Step 1: Install TensorFlow Lite Support Pypi package.

You can install the TensorFlow Lite Support Pypi package using the following
command:

```sh
pip install tflite-support
```

### Step 2: Using the model

```python
from tflite_support.task import text

# Initialization.
text_embedder = text.TextEmbedder.create_from_file(model_file)

# Run inference on two texts.
result_1 = text_embedder.embed(text_1)
result_2 = text_embedder.embed(text_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = text_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/text/text_embedder.py)
for more options to configure `TextEmbedder`.

## Example results

Cosine similarity between normalized feature vectors return a score between -1
and 1. Higher is better, i.e. a cosine similarity of 1 means the two vectors are
identical.

```
Cosine similarity: 0.954312
```

Try out the simple
[CLI demo tool for TextEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/text/desktop#textembedder)
with your own model and test data.

## Model compatibility requirements

The `TextEmbedder` API expects a TFLite model with mandatory
[TFLite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata).

Three main types of models are supported:

*   BERT-based models (see
    [source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/bert_utils.h)
    for more details):

    -   Exactly 3 input tensors (kTfLiteString)

        -   IDs tensor, with metadata name "ids",
        -   Mask tensor, with metadata name "mask".
        -   Segment IDs tensor, with metadata name "segment_ids"

    -   Exactly one output tensor (kTfLiteUInt8/kTfLiteFloat32)

        -   with `N` components corresponding to the `N` dimensions of the
            returned feature vector for this output layer.
        -   Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.

    -   An input_process_units for Wordpiece/Sentencepiece Tokenizer

*   Universal Sentence Encoder-based models (see
    [source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/text/utils/universal_sentence_encoder_utils.h)
    for more details):

    -   Exactly 3 input tensors (kTfLiteString)

        -   Query text tensor, with metadata name "inp_text".
        -   Response context tensor, with metadata name "res_context".
        -   Response text tensor, with metadata name "res_text".

    -   Exactly 2 output tensors (kTfLiteUInt8/kTfLiteFloat32)

        -   Query encoding tensor, with metadata name "query_encoding".
        -   Response encoding tensor, with metadata name "response_encoding".
        -   Both with `N` components corresponding to the `N` dimensions of the
            returned feature vector for this output layer.
        -   Both with either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x
            N]`.

*   Any text embedder model with:

    -   An input text tensor (kTfLiteString)
    -   At least one output embedding tensor (kTfLiteUInt8/kTfLiteFloat32)

        -   with `N` components corresponding to the `N` dimensions of the
            returned feature vector for this output layer.
        -   Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
