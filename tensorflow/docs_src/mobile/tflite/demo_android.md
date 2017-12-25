# TensorFlow Lite Demo for Android

The TensorFlow Lite demo is a camera app that continuously classifies whatever
it sees from your device's back camera, using a quantized MobileNet model.

You'll need an Android device running Android 5.0 or higher to run the demo.

To get you started working with TensorFlow Lite on Android, we'll walk you
through building and deploying our TensorFlow demo app in Android Studio.

It's also possible to build the demo app with Bazel, but we only recommend
this for advanced users who are very familiar with the Bazel build
environment. For more information on that, see our page [on Github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite#building-tensorflow-lite-and-the-demo-app-from-source).

## Build and deploy with Android Studio

1. Clone the TensorFlow repository from GitHub if you haven't already:

        git clone https://github.com/tensorflow/tensorflow

2. Install the latest version of Android Studio from [here](https://developer.android.com/studio/index.html).

3. From the **Welcome to Android Studio** screen, use the **Import Project
   (Gradle, Eclipse ADT, etc)** option to import the
   `tensorflow/contrib/lite/java/demo` directory as an existing Android Studio
   Project.

    Android Studio may prompt you to install Gradle upgrades and other tool
    versions; you should accept these upgrades.

4. Download the TensorFlow Lite MobileNet model from [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_224_android_quant_2017_11_08.zip).

    Unzip this and copy the `mobilenet_quant_v1_224.tflite` file to the assets
    directory: `tensorflow/contrib/lite/java/demo/app/src/main/assets/`

5. Build and run the app in Android Studio.

You'll have to grant permissions for the app to use the device's camera. Point
the camera at various objects and enjoy seeing how the model classifies things!
