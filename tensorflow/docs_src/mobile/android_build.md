# Building TensorFlow on Android

To get you started working with TensorFlow on Android, we'll walk through two
ways to build our TensorFlow mobile demos and deploying them on an Android
device. The first is Android Studio, which lets you build and deploy in an
IDE. The second is building with Bazel and deploying with ADB on the command
line.

Why choose one or the other of these methods?

The simplest way to use TensorFlow on Android is to use Android Studio. If you
aren't planning to customize your TensorFlow build at all, or if you want to use
Android Studio's editor and other features to build an app and just want to add
TensorFlow to it, we recommend using Android Studio.

If you are using custom ops, or have some other reason to build TensorFlow from
scratch, scroll down and see our instructions
for [building the demo with Bazel](#build_the_demo_using_bazel).

## Build the demo using Android Studio

**Prerequisites**

If you haven't already, do the following two things:

- Install [Android Studio](https://developer.android.com/studio/index.html),
  following the instructions on their website.

- Clone the TensorFlow repository from Github:

        git clone https://github.com/tensorflow/tensorflow

**Building**

1. Open Android Studio, and from the Welcome screen, select **Open an existing
   Android Studio project**.

2. From the **Open File or Project** window that appears, navigate to and select
    the `tensorflow/examples/android` directory from wherever you cloned the
    TensorFlow Github repo.  Click OK.

    If it asks you to do a Gradle Sync, click OK.

    You may also need to install various platforms and tools, if you get
    errors like "Failed to find target with hash string 'android-23' and similar.

3. Open the `build.gradle` file (you can go to **1:Project** in the side panel
    and find it under the **Gradle Scripts** zippy under **Android**). Look for
    the `nativeBuildSystem` variable and set it to `none` if it isn't already:

        // set to 'bazel', 'cmake', 'makefile', 'none'
        def nativeBuildSystem = 'none'

4. Click the *Run* button (the green arrow) or select *Run > Run 'android'* from the
    top menu. You may need to rebuild the project using *Build > Rebuild Project*.

    If it asks you to use Instant Run, click **Proceed Without Instant Run**.

    Also, you need to have an Android device plugged in with developer options
    enabled at this
    point. See [here](https://developer.android.com/studio/run/device.html) for
    more details on setting up developer devices.

This installs three apps on your phone that are all part of the TensorFlow
Demo. See [Android Sample Apps](#android_sample_apps) for more information about
them.

## Adding TensorFlow to your apps using Android Studio

To add TensorFlow to your own apps on Android, the simplest way is to add the
following lines to your Gradle build file:

    allprojects {
        repositories {
            jcenter()
        }
	}

    dependencies {
        compile 'org.tensorflow:tensorflow-android:+'
    }

This automatically downloads the latest stable version of TensorFlow as an AAR
and installs it in your project.

## Build the demo using Bazel

Another way to use TensorFlow on Android is to build an APK
using [Bazel](https://bazel.build/) and load it onto your device
using [ADB](https://developer.android.com/studio/command-line/adb.html). This
requires some knowledge of build systems and Android developer tools, but we'll
guide you through the basics here.

- First, follow our instructions for @{$install/install_sources$installing from sources}.
  This will also guide you through installing Bazel and cloning the
  TensorFlow code.

- Download the Android [SDK](https://developer.android.com/studio/index.html)
  and [NDK](https://developer.android.com/ndk/downloads/index.html) if you do
  not already have them. You need at least version 12b of the NDK, and 23 of the
  SDK.

- In your copy of the TensorFlow source, update the
  [WORKSPACE](https://github.com/tensorflow/tensorflow/blob/master/WORKSPACE)
  file with the location of your SDK and NDK, where it says &lt;PATH_TO_NDK&gt;
  and &lt;PATH_TO_SDK&gt;.

- Run Bazel to build the demo APK:

        bazel build -c opt //tensorflow/examples/android:tensorflow_demo

- Use [ADB](https://developer.android.com/studio/command-line/adb.html#move) to
  install the APK onto your device:

        adb install -r bazel-bin/tensorflow/examples/android/tensorflow_demo.apk

Note: In general when compiling for Android with Bazel you need
`--config=android` on the Bazel command line, though in this case this
particular example is Android-only, so you don't need it here.

This installs three apps on your phone that are all part of the TensorFlow
Demo. See [Android Sample Apps](#android_sample_apps) for more information about
them.

## Android Sample Apps

The
[Android example code](https://www.tensorflow.org/code/tensorflow/examples/android/) is
a single project that builds and installs three sample apps which all use the
same underlying code. The sample apps all take video input from a phone's
camera:

- **TF Classify** uses the Inception v3 model to label the objects it’s pointed
  at with classes from Imagenet. There are only 1,000 categories in Imagenet,
  which misses most everyday objects and includes many things you’re unlikely to
  encounter often in real life, so the results can often be quite amusing. For
  example there’s no ‘person’ category, so instead it will often guess things it
  does know that are often associated with pictures of people, like a seat belt
  or an oxygen mask. If you do want to customize this example to recognize
  objects you care about, you can use
  the
  [TensorFlow for Poets codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/index.html#0) as
  an example for how to train a model based on your own data.

- **TF Detect** uses a multibox model to try to draw bounding boxes around the
  locations of people in the camera. These boxes are annotated with the
  confidence for each detection result. Results will not be perfect, as this
  kind of object detection is still an active research topic.  The demo also
  includes optical tracking for when objects move between frames, which runs
  more frequently than the TensorFlow inference. This improves the user
  experience since the apparent frame rate is faster, but it also gives the
  ability to estimate which boxes refer to the same object between frames, which
  is important for counting objects over time.

- **TF Stylize** implements a real-time style transfer algorithm on the camera
  feed. You can select which styles to use and mix between them using the
  palette at the bottom of the screen, and also switch out the resolution of the
  processing to go higher or lower rez.

When you build and install the demo, you'll see three app icons on your phone,
one for each of the demos. Tapping on them should open up the app and let you
explore what they do. You can enable profiling statistics on-screen by tapping
the volume up button while they’re running.

### Android Inference Library

Because Android apps need to be written in Java, and core TensorFlow is in C++,
TensorFlow has a JNI library to interface between the two. Its interface is aimed
only at inference, so it provides the ability to load a graph, set up inputs,
and run the model to calculate particular outputs. You can see the full
documentation for the minimal set of methods in
[TensorFlowInferenceInterface.java](https://www.tensorflow.org/code/tensorflow/contrib/android/java/org/tensorflow/contrib/android/TensorFlowInferenceInterface.java)

The demos applications use this interface, so they’re a good place to look for
example usage. You can download prebuilt binary jars
at
[ci.tensorflow.org](https://ci.tensorflow.org/view/Nightly/job/nightly-android/).
