# Image classification

<img src="../images/image.png" class="attempt-right">

Use a pre-trained and optimized model to identify hundreds of classes of
objects, including people, activities, animals, plants, and places.

## Get started

If you are unfamiliar with the concept of image classification, you should start
by reading <a href="#what_is_image_classification">What is image
classification?</a>

If you understand image classification, you’re new to TensorFlow Lite, and
you’re working with Android or iOS, we recommend following the corresponding
tutorial that will walk you through our sample code.

<a class="button button-primary" href="android">Android</a>
<a class="button button-primary" href="ios">iOS</a>

We also provide <a href="example_applications">example applications</a> you can
use to get started.

If you are using a platform other than Android or iOS, or you are already
familiar with the <a href="../../apis">TensorFlow Lite APIs</a>, you can
download our starter image classification model and the accompanying labels.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Download
starter model and labels</a>

Once you have the starter model running on your target device, you can
experiment with different models to find the optimal balance between
performance, accuracy, and model size. For guidance, see
<a href="#choose_a_different_model">Choose a different model</a>.

If you are using a platform other than Android or iOS, or you are already
familiar with the <a href="../../apis.md">TensorFlow Lite APIs</a>, you can
download our starter image classification model and the accompanying labels.

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">Download
starter model and labels</a>

### Example applications

We have example applications for image classification for both Android and iOS.

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android
example</a>
<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios">iOS
example</a>

The following screenshot shows the Android image classification example:

<img src="images/android_banana.png" alt="Screenshot of Android example" width="30%">

## What is image classification?

A common use of machine learning is to identify what an image represents. For
example, we might want to know what type of animal appears in the following
photograph.

<img src="images/dog.png" alt="dog" width="50%">

The task of predicting what an image represents is called _image
classification_. An image classification model is trained to recognize various
classes of images. For example, a model might be trained to recognize photos
representing three different types of animals: rabbits, hamsters, and dogs.

When we subsequently provide a new image as input to the model, it will output
the probabilities of the image representing each of the types of animal it was
trained on. An example output might be as follows:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Animal type</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Rabbit</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>Hamster</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">Dog</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

Based on the output, we can see that the classification model has predicted that
the image has a high probability of representing a dog.

Note: Image classification can only tell you the probability that an image
represents one or more of the classes that the model was trained on. It cannot
tell you the position or identity of objects within the image. If you need to
identify objects and their positions within images, you should use an
<a href="../object_detection/overview.md">object detection</a> model.

### Training, labels, and inference

During training, an image classification model is fed images and their
associated _labels_. Each label is the name of a distinct concept, or class,
that the model will learn to recognize.

Given sufficient training data (often hundreds or thousands of images per
label), an image classification model can learn to predict whether new images
belong to any of the classes it has been trained on. This process of prediction
is called _inference_.

To perform inference, an image is passed as input to a model. The model will
then output an array of probabilities between 0 and 1. With our example model,
this process might look like the following:

<table style="width: 60%">
  <tr style="border-top: 0px;">
    <td style="width: 40%"><img src="images/dog.png" alt="dog"></td>
    <td style="width: 20%; font-size: 2em; vertical-align: middle; text-align: center;">→</td>
    <td style="width: 40%; vertical-align: middle; text-align: center;">[0.07, 0.02, 0.91]</td>
</table>

Each number in the output corresponds to a label in our training data.
Associating our output with the three labels the model was trained on, we can
see the model has predicted a high probability that the image represents a dog.

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Label</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>rabbit</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>hamster</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">dog</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

You might notice that the sum of all the probabilities (for rabbit, hamster, and
dog) is equal to 1. This is a common type of output for models with multiple
classes (see
<a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a>
for more information).

### Ambiguous results

Since the probabilities will always sum to 1, if the image is not confidently
recognized as belonging to any of the classes the model was trained on you may
see the probability distributed throughout the labels without any one value
being significantly larger.

For example, the following might indicate an ambiguous result:

<table style="width: 40%;">
  <thead>
    <tr>
      <th>Label</th>
      <th>Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>rabbit</td>
      <td>0.31</td>
    </tr>
    <tr>
      <td>hamster</td>
      <td>0.35</td>
    </tr>
    <tr>
      <td>dog</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>

### Uses and limitations

The image classification models that we provide are useful for single-label
classification, which means predicting which single label the image is most
likely to represent. They are trained to recognize 1000 classes of image. For a
full list of classes, see the labels file in the
<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model
zip</a>.

If you want to train a model to recognize new classes, see
<a href="#customize_model">Customize model</a>.

For the following use cases, you should use a different type of model:

<ul>
  <li>Predicting the type and position of one or more objects within an image (see <a href="object_detection">object detection</a>)</li>
  <li>Predicting the composition of an image, for example subject versus background (see <a href="segmentation">segmentation</a>)</li>
</ul>

Once you have the starter model running on your target device, you can
experiment with different models to find the optimal balance between
performance, accuracy, and model size. For guidance, see
<a href="#choose_a_different_model">Choose a different model</a>.

## Choose a different model

There are a large number of image classification models available on our
<a href="../hosted.md">List of hosted models</a>. You should aim to choose the
optimal model for your application based on performance, accuracy and model
size. There are trade-offs between each of them.

### Performance

We measure performance in terms of the amount of time it takes for a model to
run inference on a given piece of hardware. The less time, the faster the model.

The performance you require depends on your application. Performance can be
important for applications like real-time video, where it may be important to
analyze each frame in the time before the next frame is drawn (e.g. inference
must be faster than 33ms to perform real-time inference on a 30fps video
stream).

Our quantized Mobilenet models’ performance ranges from 3.7ms to 80.3 ms.

### Accuracy

We measure accuracy in terms of how often the model correctly classifies an
image. For example, a model with a stated accuracy of 60% can be expected to
classify an image correctly an average of 60% of the time.

Our <a href="../hosted.md">List of hosted models</a> provides Top-1 and Top-5
accuracy statistics. Top-1 refers to how often the correct label appears as the
label with the highest probability in the model’s output. Top-5 refers to how
often the correct label appears in the top 5 highest probabilities in the
model’s output.

Our quantized Mobilenet models’ Top-5 accuracy ranges from 64.4 to 89.9%.

### Size

The size of a model on-disk varies with its performance and accuracy. Size may
be important for mobile development (where it might impact app download sizes)
or when working with hardware (where available storage might be limited).

Our quantized Mobilenet models’ size ranges from 0.5 to 3.4 Mb.

### Architecture

There are several different architectures of models available on
<a href="../hosted.md">List of hosted models</a>, indicated by the model’s name.
For example, you can choose between Mobilenet, Inception, and others.

The architecture of a model impacts its performance, accuracy, and size. All of
our hosted models are trained on the same data, meaning you can use the provided
statistics to compare them and choose which is optimal for your application.

Note: The image classification models we provide accept varying sizes of input. For some models, this is indicated in the filename. For example, the Mobilenet_V1_1.0_224 model accepts an input of 224x224 pixels. <br /><br />All of the models require three color channels per pixel (red, green, and blue). Quantized models require 1 byte per channel, and float models require 4 bytes per channel.<br /><br />Our <a href="android.md">Android</a> and <a href="ios">iOS</a> code samples demonstrate how to process full-sized camera images into the required format for each model.

## Customize model

The pre-trained models we provide are trained to recognize 1000 classes of
image. For a full list of classes, see the labels file in the
<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">model
zip</a>.

You can use a technique known as _transfer learning_ to re-train a model to
recognize classes not in the original set. For example, you could re-train the
model to distinguish between different species of tree, despite there being no
trees in the original training data. To do this, you will need a set of training
images for each of the new labels you wish to train.

Learn how to perform transfer learning in the
<a href="https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/">TensorFlow
for Poets</a> codelab.
