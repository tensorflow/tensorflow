<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>

Installation Requirements for Android project:

1.) Use Ubuntu 16.04 or higher.

Installation for android client:

1.) Clone this repository by using.

```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
```

Note that `--recurse-submodules` is necessary to prevent some issues with protobuf compilation.

2.) Install Bazel

Bazel is the primary build system for TensorFlow. To build with Bazel,
it and the Android NDK and SDK must be installed on your system.

Get the recommended Bazel version listed in [os_setup.html](https://www.tensorflow.org/versions/master/get_started/os_setup.html#source)

3.) Download [Android Studio](https://developer.android.com/studio/index.html). Build
        tools API >= 23 is required to build the TF Android demo.
        
4.) Open Android Studio then open SDK Manager from the Configure dropdown.

<div align="center">
  <img src="https://skonda.in/wp-content/uploads/2016/05/Android-Studio-Welcome-screen.png"><br><br>
</div>

- Install Android SDK 6.0.1
- Install Android NDK

5.) Make note where android installs SDK and NDK. 
- Usually /home/<user>/Android/Sdk and /home/<user>/Android/Sdk/ndk-bundle respectively
- You can check by opening SDK manager, going to the SDK tools tab and clicking the installed SDK and the path will be on the top of the window.

6.) Open Android Studio project located in tensorflow/examples/android

7.) Install any packages required by the project.

8.) Plug in any android device in developer mode into your machine.

9.) Run a debug build of the application using your android device as the test device.

10.) Done! Yay!

-----------------

***TENSORFLOW***

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow also includes TensorBoard, a data visualization toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

## Installation
*See [Download and Setup](tensorflow/g3doc/get_started/os_setup.md) for instructions on how to install our release binaries or how to build from source.*

##For more information

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow Model Zoo](https://github.com/tensorflow/models)
* [TensorFlow MOOC on Udacity](https://www.udacity.com/course/deep-learning--ud730)

The TensorFlow community has created amazing things with TensorFlow, please see the [resources section of tensorflow.org](https://www.tensorflow.org/versions/master/resources#community) for an incomplete list.
