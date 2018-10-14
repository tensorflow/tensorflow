
# Android Demo App

An example Android application using TensorFLow Lite is available
[on GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo).
The demo is a sample camera app that classifies images continuously
using either a quantized Mobilenet model or a floating point Inception-v3 model.
To run the demo, a device running Android 5.0 ( API 21) or higher is required.

In the demo app, inference is done using the TensorFlow Lite Java API. The demo
app classifies frames in real-time, displaying the top most probable
classifications. It also displays the time taken to detect the object.

There are three ways to get the demo app to your device:

* Download the [prebuilt binary APK](http://download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk).
* Use Android Studio to build the application.
* Download the source code for TensorFlow Lite and the demo and build it using
  bazel.


## Download the pre-built binary

The easiest way to try the demo is to download the
[pre-built binary APK](https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk)

Once the APK is installed, click the app icon to start the program. The first
time the app is opened, it asks for runtime permissions to access the device
camera. The demo app opens the back-camera of the device and recognizes objects
in the camera's field of view. At the bottom of the image (or at the left
of the image if the device is in landscape mode), it displays top three objects
classified and the classification latency.


## Build in Android Studio with TensorFlow Lite AAR from JCenter

Use Android Studio to try out changes in the project code and compile the demo
app:

* Install the latest version of
  [Android Studio](https://developer.android.com/studio/index.html).
* Make sure the Android SDK version is greater than 26 and NDK version is greater
  than 14 (in the Android Studio settings).
* Import the `tensorflow/contrib/lite/java/demo` directory as a new
  Android Studio project.
* Install all the Gradle extensions it requests.

Now you can build and run the demo app. 

The build process downloads the quantized [Mobilenet TensorFlow Lite model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip), and unzips it into the assets directory: `tensorflow/contrib/lite/java/demo/app/src/main/assets/`.

Some additional details are available on the
[TF Lite Android App page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/README.md).

### Using other models

To use a different model:
* Download the floating point [Inception-v3 model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip).
* Unzip and copy `inceptionv3_non_slim_2015.tflite` to the assets directory. 
* Change the chosen classifier in [Camera2BasicFragment.java](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/Camera2BasicFragment.java)<br>
  from: `classifier = new ImageClassifierQuantizedMobileNet(getActivity());`<br>
  to: `classifier = new ImageClassifierFloatInception(getActivity());`.


## Build TensorFlow Lite and the demo app from source

### Clone the TensorFlow repo

```sh
git clone https://github.com/tensorflow/tensorflow
```

### Install Bazel

If `bazel` is not installed on your system, see
[Installing Bazel](https://bazel.build/versions/master/docs/install.html).

Note: Bazel does not currently support Android builds on Windows. Windows users
should download the
[prebuilt binary](https://storage.googleapis.com/download.tensorflow.org/deps/tflite/TfLiteCameraDemo.apk).

### Install Android NDK and SDK

The Android NDK is required to build the native (C/C++) TensorFlow Lite code. The
current recommended version is *14b* and can be found on the
[NDK Archives](https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads)
page.

The Android SDK and build tools can be
[downloaded separately](https://developer.android.com/tools/revisions/build-tools.html)
or used as part of
[Android Studio](https://developer.android.com/studio/index.html). To build the
TensorFlow Lite Android demo, build tools require API >= 23 (but it will run on
devices with API >= 21).

In the root of the TensorFlow repository, update the `WORKSPACE` file with the
`api_level` and location of the SDK and NDK. If you installed it with
Android Studio, the SDK path can be found in the SDK manager. The default NDK
path is:`{SDK path}/ndk-bundle.` For example:

```
android_sdk_repository (
    name = "androidsdk",
    api_level = 23,
    build_tools_version = "23.0.2",
    path = "/home/xxxx/android-sdk-linux/",
)

android_ndk_repository(
    name = "androidndk",
    path = "/home/xxxx/android-ndk-r10e/",
    api_level = 19,
)
```

Some additional details are available on the
[TF Lite Android App page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/README.md).

### Build the source code

To build the demo app, run `bazel`:

```
bazel build --cxxopt=--std=c++11 //tensorflow/contrib/lite/java/demo/app/src/main:TfLiteCameraDemo
```

Caution: Because of an bazel bug, we only support building the Android demo app
within a Python 2 environment.


## About the demo

The demo app is resizing each camera image frame (224 width * 224 height) to
match the quantized MobileNets model (299 * 299 for Inception-v3). The resized
image is converted—row by row—into a
[ByteBuffer](https://developer.android.com/reference/java/nio/ByteBuffer.html).
Its size is  1 * 224 * 224 * 3 bytes, where 1 is the number of images in a batch.
224 * 224 (299 * 299) is the width and height of the image. 3 bytes represents
the 3 colors of a pixel.

This demo uses the TensorFlow Lite Java inference API
for models which take a single input and provide a single output. This outputs a
two-dimensional array, with the first dimension being the category index and the
second dimension being the confidence of classification. Both models have 1001
unique categories and the app sorts the probabilities of all the categories and
displays the top three. The model file must be downloaded and bundled within the
assets directory of the app.
