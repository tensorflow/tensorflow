# Integrate image searchers

Image search allows searching for similar images in a database of images. It
works by embedding the search query into a high-dimensional vector representing
the semantic meaning of the query, followed by similarity search in a
predefined, custom index using
[ScaNN](https://github.com/google-research/google-research/tree/master/scann)
(Scalable Nearest Neighbors).

As opposed to
[image classification](https://www.tensorflow.org/lite/inference_with_metadata/task_library/image_classifier),
expanding the number of items that can be recognized doesn't require re-training
the entire model. New items can be added simply re-building the index. This also
enables working with larger (100k+ items) databases of images.

Use the Task Library `ImageSearcher` API to deploy your custom image searcher
into your mobile apps.

## Key features of the ImageSearcher API

*   Takes a single image as input, performs embedding extraction and
    nearest-neighbor search in the index.

*   Input image processing, including rotation, resizing, and color space
    conversion.

*   Region of interest of the input image.

## Prerequisites

Before using the `ImageSearcher` API, an index needs to be built based on the
custom corpus of images to search into. This can be achieved using
[Model Maker Searcher API](https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/searcher)
by following and adapting the
[tutorial](https://www.tensorflow.org/lite/models/modify/model_maker/text_searcher).

For this you will need:

*   a TFLite image embedder model such as
    [mobilenet v3](https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/metadata/1).
    See more pretrained embedder models (a.k.a feature vector models) from the
    [Google Image Modules collection on TensorFlow Hub](https://tfhub.dev/google/collections/image/1).
*   your corpus of images.

After this step, you should have a standalone TFLite searcher model (e.g.
`mobilenet_v3_searcher.tflite`), which is the original image embedder model with
the index attached into the
[TFLite Model Metadata](https://www.tensorflow.org/lite/models/convert/metadata).

## Run inference in Java

### Step 1: Import Gradle dependency and other settings

Copy the `.tflite` searcher model file to the assets directory of the Android
module where the model will be run. Specify that the file should not be
compressed, and add the TensorFlow Lite library to the module’s `build.gradle`
file:

```java
android {
    // Other settings

    // Specify tflite index file should not be compressed for the app apk
    aaptOptions {
        noCompress "tflite"
    }

}

dependencies {
    // Other dependencies

    // Import the Task Vision Library dependency (NNAPI is included)
    implementation 'org.tensorflow:tensorflow-lite-task-vision:0.4.0'
    // Import the GPU delegate plugin Library for GPU inference
    implementation 'org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.0'
}
```

### Step 2: Using the model

```java
// Initialization
ImageSearcherOptions options =
    ImageSearcherOptions.builder()
        .setBaseOptions(BaseOptions.builder().useGpu().build())
        .setSearcherOptions(
            SearcherOptions.builder().setL2Normalize(true).build())
        .build();
ImageSearcher imageSearcher =
    ImageSearcher.createFromFileAndOptions(context, modelFile, options);

// Run inference
List<NearestNeighbor> results = imageSearcher.search(image);
```

See the
[source code and javadoc](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/java/src/java/org/tensorflow/lite/task/vision/searcher/ImageSearcher.java)
for more options to configure the `ImageSearcher`.

## Run inference in C++

```c++
// Initialization
ImageSearcherOptions options;
options.mutable_base_options()->mutable_model_file()->set_file_name(model_file);
options.mutable_embedding_options()->set_l2_normalize(true);
std::unique_ptr<ImageSearcher> image_searcher = ImageSearcher::CreateFromOptions(options).value();

// Run inference
const SearchResult result = image_searcher->Search(*frame_buffer).value();
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/cc/task/vision/image_searcher.h)
for more options to configure `ImageSearcher`.

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

# Initialization
image_searcher = vision.ImageSearcher.create_from_file(model_file)

# Run inference
image = vision.TensorImage.create_from_file(image_file)
result = image_searcher.search(image)
```

See the
[source code](https://github.com/tensorflow/tflite-support/blob/master/tensorflow_lite_support/python/task/vision/image_searcher.py)
for more options to configure `ImageSearcher`.

## Example results

```
Results:
 Rank#0:
  metadata: burger
  distance: 0.13452
 Rank#1:
  metadata: car
  distance: 1.81935
 Rank#2:
  metadata: bird
  distance: 1.96617
 Rank#3:
  metadata: dog
  distance: 2.05610
 Rank#4:
  metadata: cat
  distance: 2.06347
```

Try out the simple
[CLI demo tool for ImageSearcher](https://github.com/tensorflow/tflite-support/tree/master/tensorflow_lite_support/examples/task/vision/desktop#imagesearcher)
with your own model and test data.
