# Integrate image embedders.

Image embedders allow embedding images into a high-dimensional feature vector
representing the semantic meaning of an image, which can then be compared with
the feature vector of other images to evaluate their semantic similarity.

As opposed to
[image search](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_searcher),
the image embedder allows computing the similarity between images on-the-fly
instead of searching through a predefined index built from a corpus of images.

Use the Task Library `ImageEmbedder` API to deploy your custom image embedder
into your mobile apps.

## Key features of the ImageEmbedder API

*   Input image processing, including rotation, resizing, and color space
    conversion.

*   Region of interest of the input image.

*   Built-in utility function to compute the
    [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) between
    feature vectors.

## Supported image embedder models

The following models are guaranteed to be compatible with the `ImageEmbedder`
API.

*   Feature vector models from the
    [Google Image Modules collection on TensorFlow Hub](https://tfhub.dev/google/collections/image/1).

*   Custom models that meet the
    [model compatibility requirements](#model-compatibility-requirements).

## Run inference in C++

```c++
// Initialization.
ImageEmbedderOptions options:
options.mutable_model_file_with_metadata()->set_file_name(model_file);
options.set_l2_normalize(true);
std::unique_ptr<ImageEmbedder> image_embedder = ImageEmbedder::CreateFromOptions(options).value();

// Run inference on two images.
const EmbeddingResult result_1 = image_embedder->Embed(*frame_buffer_1);
const EmbeddingResult result_2 = image_embedder->Embed(*frame_buffer_2);

// Compute cosine similarity.
double similarity = ImageEmbedder::CosineSimilarity(
    result_1.embeddings[0].feature_vector()
    result_2.embeddings[0].feature_vector());
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_embedder.h)
for more options to configure `ImageEmbedder`.

## Run inference in Python

### Step 1: Install TensorFlow Lite Support Pypi package.

You can install the TensorFlow Lite Support Pypi package using the following
command:

```sh
pip install tflite-support
```

### Step 2: Using the model

```python
from tflite_support.task import vision

# Initialization.
image_embedder = vision.ImageEmbedder.create_from_file(model_file)

# Run inference on two images.
image_1 = vision.TensorImage.create_from_file('/path/to/image1.jpg')
result_1 = image_embedder.embed(image_1)
image_2 = vision.TensorImage.create_from_file('/path/to/image2.jpg')
result_2 = image_embedder.embed(image_2)

# Compute cosine similarity.
feature_vector_1 = result_1.embeddings[0].feature_vector
feature_vector_2 = result_2.embeddings[0].feature_vector
similarity = image_embedder.cosine_similarity(
    result_1.embeddings[0].feature_vector, result_2.embeddings[0].feature_vector)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_embedder.py)
for more options to configure `ImageEmbedder`.

## Example results

Cosine similarity between normalized feature vectors return a score between -1
and 1. Higher is better, i.e. a cosine similarity of 1 means the two vectors are
identical.

```
Cosine similarity: 0.954312
```

Try out the simple
[CLI demo tool for ImageEmbedder](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imageembedder)
with your own model and test data.

## Model compatibility requirements

The `ImageEmbedder` API expects a TFLite model with optional, but strongly
recommended
[TFLite Model Metadata](https://www.tensorflow.org/lite/convert/metadata).

The compatible image embedder models should meet the following requirements:

*   An input image tensor (kTfLiteUInt8/kTfLiteFloat32)

    -   image input of size `[batch x height x width x channels]`.
    -   batch inference is not supported (`batch` is required to be 1).
    -   only RGB inputs are supported (`channels` is required to be 3).
    -   if type is kTfLiteFloat32, NormalizationOptions are required to be
        attached to the metadata for input normalization.

*   At least one output tensor (kTfLiteUInt8/kTfLiteFloat32)

    -   with `N` components corresponding to the `N` dimensions of the returned
        feature vector for this output layer.
    -   Either 2 or 4 dimensions, i.e. `[1 x N]` or `[1 x 1 x 1 x N]`.
