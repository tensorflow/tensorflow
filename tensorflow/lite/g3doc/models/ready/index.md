# Pre-built models for TensorFlow Lite

There are a variety of pre-built, open source models you can use immediately
with TensorFlow Lite to accomplish many machine learning tasks. Using pre-built
TensorFlow Lite models lets you add machine learning functionality to your
mobile and edge device application quickly, without having to build and train a
model. This guide helps you find and decide on pre-built models for use with
TensorFlow Lite.

You can start browsing TensorFlow Lite models right away based on general use
cases in the [TensorFlow Lite Examples](../../examples) section, or browse a
larger set of models on [TensorFlow Hub](https://tfhub.dev/s?deployment-
format=lite).

**Important:** TensorFlow Hub lists both regular TensorFlow models and
TensorFlow Lite format models. These model formats are not interchangeable.
TensorFlow models can be converted into TensorFlow Lite models, but that process
is not reversible.


## Find a model for your application

Finding an existing TensorFlow Lite model for your use case can be tricky
depending on what you are trying to accomplish. Here are a few recommended ways
to discover models for use with TensorFlow Lite:

**By example:** The fastest way to find and start using models with TensorFlow
Lite is to browse the [TensorFlow Lite Examples](../../examples) section to find
models that perform a task which is similar to your use case. This short catalog
of examples provides models for common use cases with explanations of the models
and sample code to get you started running and using them.

**By data input type:** Aside from looking at examples similar to your use
case, another way to discover models for your own use is to consider the type of
data you want to process, such as audio, text, images, or video data. Machine
learning models are frequently designed for use with one of these types of data,
so looking for models that handle the data type you want to use can help you
narrow down what models to consider. On [TensorFlow
Hub](https://tfhub.dev/s?deployment-format=lite), you can use the **Problem
domain** filter to view model data types and narrow your list.

Note: Processing video with machine learning models can frequently be
accomplished with models that are designed for processing single images,
depending on how fast and how many inferences you need to perform for your use
case. If you intend to use video for your use case, consider using single-frame
video sampling with a model built for fast processing of individual images.

The following lists links to TensorFlow Lite models on [TensorFlow
Hub](https://tfhub.dev/s?deployment-format=lite) for common use cases:

-   [Image classification](https://tfhub.dev/s?deployment-format=lite&module-type=image-classification)
    models
-   [Object detection](https://tfhub.dev/s?deployment-format=lite&module-type=image-object-detection)
    models
-   [Text classification](https://tfhub.dev/s?deployment-format=lite&module-type=text-classification)
    models
-   [Text embedding](https://tfhub.dev/s?deployment-format=lite&module-type=text-embedding)
    models
-   [Audio speech synthesis](https://tfhub.dev/s?deployment-format=lite&module-type=audio-speech-synthesis)
    models
-   [Audio embedding](https://tfhub.dev/s?deployment-format=lite&module-type=audio-embedding)
    models


## Choose between similar models

If your application follows a common use case such as image classification or
object detection, you may find yourself deciding between multiple TensorFlow
Lite models, with varying binary size, data input size, inference speed, and
prediction accuracy ratings. When deciding between a number of models, you
should narrow your options based first on your most limiting constraint: size of
model, size of data, inference speed, or accuracy.

Key Point: Generally, when choosing between similar models, pick the smallest
model to allow for the broadest device compatibility and fast inference times.

If you are not sure what your most limiting constraint is, assume it is the
size of the model and pick the smallest model available. Picking a small model
gives you the most flexibility in terms of the devices where you can
successfully deploy and run the model. Smaller models also typically produce
faster inferences, and speedier predictions generally create better end-user
experiences. Smaller models typically have lower accuracy rates, so you may need
to pick larger models if prediction accuracy is your primary concern.


## Sources for models

Use the [TensorFlow Lite Examples](../../examples)
section and [TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) as your
first destinations for finding and selecting models for use with TensorFlow
Lite. These sources generally have up to date, curated models for use with
TensorFlow Lite, and frequently include sample code to accelerate your
development process.

### TensorFlow models

It is possible to [convert](https://www.tensorflow.org/lite/convert) regular
TensorFlow models to TensorFlow Lite format. For more information about
converting models, see the [TensorFlow Lite
Converter](https://www.tensorflow.org/lite/convert) documentation. You can find
TensorFlow models on [TensorFlow Hub](https://tfhub.dev/) and in the
[TensorFlow Model Garden](https://github.com/tensorflow/models).
