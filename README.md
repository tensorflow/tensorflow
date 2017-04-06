##Installation Requirements for Android project:

1.) Use Ubuntu 16.04 or higher.

##Installation for android client:

**1.) Clone this repository by using.**

```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
```

Note that `--recurse-submodules` is necessary to prevent some issues with protobuf compilation.

**2.) Install Bazel**

Bazel is the primary build system for TensorFlow. To build with Bazel,
it and the Android NDK and SDK must be installed on your system.

Get the recommended Bazel version listed in [os_setup.html](https://www.tensorflow.org/versions/master/get_started/os_setup.html#source)

**3.) Download [Android Studio](https://developer.android.com/studio/index.html).** Build
        tools API >= 23 is required to build the TF Android demo.
        
**4.) Open Android Studio then open SDK Manager from the Configure dropdown.**

<div align="center">
  <img src="https://skonda.in/wp-content/uploads/2016/05/Android-Studio-Welcome-screen.png"><br><br>
</div>

- Install Android SDK 6.0.1
- Install Android NDK

**5.) Make note where android installs SDK and NDK.**
- Usually /home/<user>/Android/Sdk and /home/<user>/Android/Sdk/ndk-bundle respectively
- You can check by opening SDK manager, going to the SDK tools tab and clicking the installed SDK and the path will be on the top of the window.

**6.) Open Android Studio project located in tensorflow/examples/android**

**7.) Install any packages required by the project.**

**8.) Plug in any android device in developer mode into your machine.**

**9.) Run a debug build of the application using your android device as the test device.**

***10.) Done! Yay!***

-----------------
