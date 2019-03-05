
# TensorFlow Lite guide

TensorFlow Lite is TensorFlow’s lightweight solution for mobile and embedded
devices. It enables on-device machine learning inference with low latency and a
small binary size. TensorFlow Lite also supports hardware acceleration with the
[Android Neural Networks
API](https://developer.android.com/ndk/guides/neuralnetworks/index.html).

TensorFlow Lite uses many techniques for achieving low latency such as
optimizing the kernels for mobile apps, pre-fused activations, and quantized
kernels that allow smaller and faster (fixed-point math) models.

Most of our TensorFlow Lite documentation is [on
GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)
for the time being.

## What does TensorFlow Lite contain?

TensorFlow Lite supports a set of core operators, both quantized and
float, which have been tuned for mobile platforms. They incorporate pre-fused
activations and biases to further enhance performance and quantized
accuracy. Additionally, TensorFlow Lite also supports using custom operations in
models.

TensorFlow Lite defines a new model file format, based on
[FlatBuffers](https://google.github.io/flatbuffers/). FlatBuffers is an
efficient open-source cross-platform serialization library. It is similar to
[protocol buffers](https://developers.google.com/protocol-buffers/?hl=en), but
the primary difference is that FlatBuffers does not need a parsing/unpacking
step to a secondary representation before you can access data, often coupled
with per-object memory allocation. Also, the code footprint of FlatBuffers is an
order of magnitude smaller than protocol buffers.

TensorFlow Lite has a new mobile-optimized interpreter, which has the key goals
of keeping apps lean and fast. The interpreter uses a static graph ordering and
a custom (less-dynamic) memory allocator to ensure minimal load, initialization,
and execution latency.

TensorFlow Lite provides an interface to leverage hardware acceleration, if
available on the device. It does so via the
[Android Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/index.html),
available on Android 8.1 (API level 27) and higher.

## Why do we need a new mobile-specific library?

Machine Learning is changing the computing paradigm, and we see an emerging
trend of new use cases on mobile and embedded devices. Consumer expectations are
also trending toward natural, human-like interactions with their devices, driven
by the camera and voice interaction models.

There are several factors which are fueling interest in this domain:

- Innovation at the silicon layer is enabling new possibilities for hardware
  acceleration, and frameworks such as the Android Neural Networks API make it
  easy to leverage these.

- Recent advances in real-time computer-vision and spoken language understanding
  have led to mobile-optimized benchmark models being open sourced
  (e.g. MobileNets, SqueezeNet).

- Widely-available smart appliances create new possibilities for
  on-device intelligence.

- Interest in stronger user data privacy paradigms where user data does not need
  to leave the mobile device.

- Ability to serve ‘offline’ use cases, where the device does not need to be
  connected to a network.

We believe the next wave of machine learning applications will have significant
processing on mobile and embedded devices.

## TensorFlow Lite highlights

TensorFlow Lite provides:

- A set of core operators, both quantized and float, many of which have been
  tuned for mobile platforms.  These can be used to create and run custom
  models.  Developers can also write their own custom operators and use them in
  models.

- A new [FlatBuffers](https://google.github.io/flatbuffers/)-based
  model file format.

- On-device interpreter with kernels optimized for faster execution on mobile.

- TensorFlow converter to convert TensorFlow-trained models to the TensorFlow
  Lite format.

- Smaller in size: TensorFlow Lite is smaller than 300KB when all supported
  operators are linked and less than 200KB when using only the operators needed
  for supporting InceptionV3 and Mobilenet.

- **Pre-tested models:**

    All of the following models are guaranteed to work out of the box:

    - Inception V3, a popular model for detecting the dominant objects
      present in an image.

    - [MobileNets](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md),
      a family of mobile-first computer vision models designed to effectively
      maximize accuracy while being mindful of the restricted resources for an
      on-device or embedded application. They are small, low-latency, low-power
      models parameterized to meet the resource constraints of a variety of use
      cases. They can be built upon for classification, detection, embeddings
      and segmentation. MobileNet models are smaller but [lower in
      accuracy](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)
      than Inception V3.

    - On Device Smart Reply, an on-device model which provides one-touch
      replies for an incoming text message by suggesting contextually relevant
      messages. The model was built specifically for memory constrained devices
      such as watches & phones and it has been successfully used to surface
      [Smart Replies on Android
      Wear](https://research.googleblog.com/2017/02/on-device-machine-intelligence.html)
      to all first-party and third-party apps.

    Also see the complete list of
    [TensorFlow Lite's supported models](hosted_models.md),
    including the model sizes, performance numbers, and downloadable model files.

- Quantized versions of the MobileNet model, which runs faster than the
  non-quantized (float) version on CPU.

- New Android demo app to illustrate the use of TensorFlow Lite with a quantized
  MobileNet model for object classification.

- Java and C++ API support


## Getting Started

We recommend you try out TensorFlow Lite with the pre-tested models indicated
above. If you have an existing model, you will need to test whether your model
is compatible with both the converter and the supported operator set.  To test
your model, see the
[documentation on GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite).

### Retrain Inception-V3 or MobileNet for a custom data set

The pre-trained models mentioned above have been trained on the ImageNet data
set, which consists of 1000 predefined classes. If those classes are not
relevant or useful for your use case, you will need to retrain those
models. This technique is called transfer learning, which starts with a model
that has been already trained on a problem and will then be retrained on a
similar problem. Deep learning from scratch can take days, but transfer learning
can be done fairly quickly. In order to do this, you'll need to generate your
custom data set labeled with the relevant classes.

The [TensorFlow for Poets](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/)
codelab walks through this process step-by-step. The retraining code supports
retraining for both floating point and quantized inference.

## TensorFlow Lite Architecture

The following diagram shows the architectural design of TensorFlow Lite:

<img src="https://www.tensorflow.org/images/tflite-architecture.jpg"
     alt="TensorFlow Lite architecture diagram"
     style="max-width:600px;">

Starting with a trained TensorFlow model on disk, you'll convert that model to
the TensorFlow Lite file format (`.tflite`) using the TensorFlow Lite
Converter. Then you can use that converted file in your mobile application.

Deploying the TensorFlow Lite model file uses:

- Java API: A convenience wrapper around the C++ API on Android.

- C++ API: Loads the TensorFlow Lite Model File and invokes the Interpreter. The
  same library is available on both Android and iOS.

- Interpreter: Executes the model using a set of kernels. The interpreter
  supports selective kernel loading; without kernels it is only 100KB, and 300KB
  with all the kernels loaded. This is a significant reduction from the 1.5M
  required by TensorFlow Mobile.

- On select Android devices, the Interpreter will use the Android Neural
  Networks API for hardware acceleration, or default to CPU execution if none
  are available.

You can also implement custom kernels using the C++ API that can be used by the
Interpreter.

## Future Work

In future releases, TensorFlow Lite will support more models and built-in
operators, contain performance improvements for both fixed point and floating
point models, improvements to the tools to enable easier developer workflows and
support for other smaller devices and more. As we continue development, we hope
that TensorFlow Lite will greatly simplify the developer experience of targeting
a model for small devices.

Future plans include using specialized machine learning hardware to get the best
possible performance for a particular model on a particular device.

## Next Steps

The TensorFlow Lite [GitHub repository](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite).
contains additional docs, code samples, and demo applications.
