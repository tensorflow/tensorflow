TensorFlow-Android-Inference
============================
This directory contains CMake support for building the Android Java Inference
interface to the TensorFlow native APIs.

See [tensorflow/contrib/android](..) for more details about the library, and
instructions for building with Bazel.

Usage
-----
Add TensorFlow-Android-Inference as a dependency of your Android application

* settings.gradle

```
include ':TensorFlow-Android-Inference'
findProject(":TensorFlow-Android-Inference").projectDir = 
            new File("${/path/to/tensorflow_repo}/contrib/android/cmake")
```

* application's build.gradle (adding dependency):

```
debugCompile project(path: ':tensorflow_inference', configuration: 'debug')
releaseCompile project(path: ':tensorflow_inference', configuration: 'release')
```
Note: this makes native code in the lib traceable from your app.

Dependencies
------------
TensorFlow-Android-Inference depends on the TensorFlow static libs already built
in your local TensorFlow repo directory. For Linux/Mac OS, build_all_android.sh
is used in build.gradle to build it. It DOES take time to build the core libs;
so, by default, it is commented out to avoid confusion (otherwise
Android Studio would appear to hang during opening the project).
To enable it, refer to the comment in

* build.gradle

Output
------
- TensorFlow-Inference-debug.aar
- TensorFlow-Inference-release.aar

File libtensorflow_inference.so should be packed under jni/${ANDROID_ABI}/
in the above aar, and it is transparent to the app as it will access them via
equivalent java APIs.

