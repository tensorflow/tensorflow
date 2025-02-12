# TensorFlow Lite in Google Play services C API (Beta)

Beta: TensorFlow Lite in Google Play services C API is currently in Beta.

TensorFlow Lite in Google Play services runtime allows you to run machine
learning (ML) models without statically bundling TensorFlow Lite libraries into
your app. This guide provide instructions on how to use the C APIs for Google
Play services.

Before working with the TensorFlow Lite in Google Play services C API, make sure
you have the [CMake](https://cmake.org/) build tool installed.

## Update your build configuration

Add the following dependencies to your app project code to access the Play
services API for TensorFlow Lite:

```
implementation "com.google.android.gms:play-services-tflite-java:16.2.0-beta02"
```

Then, enable the
[Prefab](https://developer.android.com/build/dependencies#build-system-configuration)
feature to access the C API from your CMake script by updating the android block
of your module's build.gradle file:

```
buildFeatures {
  prefab = true
}
```

You finally need to add the package `tensorflowlite_jni_gms_client` imported
from the AAR as a dependency in your CMake script:

```
find_package(tensorflowlite_jni_gms_client REQUIRED CONFIG)

target_link_libraries(tflite-jni # your JNI lib target
        tensorflowlite_jni_gms_client::tensorflowlite_jni_gms_client
        android # other deps for your target
        log)

# Also add -DTFLITE_IN_GMSCORE -DTFLITE_WITH_STABLE_ABI
# to the C/C++ compiler flags.

add_compile_definitions(TFLITE_IN_GMSCORE)
add_compile_definitions(TFLITE_WITH_STABLE_ABI)
```

## Initialize the TensorFlow Lite runtime

Before calling the TensorFlow Lite Native API you must initialize the
`TfLiteNative` runtime in your Java/Kotlin code.

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <pre class="prettyprint">
Task tfLiteInitializeTask = TfLiteNative.initialize(context);
      </pre>
      </section>
      <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
val tfLiteInitializeTask: Task<Void> = TfLiteNative.initialize(context)
        </pre>
      </section>
  </devsite-selector>
</div>

Using the Google Play services Task API, `TfLiteNative.initialize`
asynchronously loads the TFLite runtime from Google Play services into your
application's runtime process. Use `addOnSuccessListener()` to make sure the
`TfLite.initialize()` task completes before executing code that accesses
TensorFlow Lite APIs. Once the task has completed successfully, you can invoke
all the available TFLite Native APIs.

## Native code implementation

To use TensorFlow Lite in Google Play services with your native code, you can do
one of the following:

-   declare new JNI functions to call native functions from your Java code
-   Call the TensorFlow Lite Native API from your existing native C code.

JNI functions:

You can declare a new JNI function to make the TensorFlow Lite runtime declared
in Java/Kotlin accessible to your native code as follow:

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <pre class="prettyprint">
package com.google.samples.gms.tflite.c;

public class TfLiteJni {
  static {
    System.loadLibrary("tflite-jni");
  }
  public TfLiteJni() { /**/ };
  public native void loadModel(AssetManager assetManager, String assetName);
  public native float[] runInference(float[] input);
}
      </pre>
      </section>
      <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
package com.google.samples.gms.tflite.c

class TfLiteJni() {
  companion object {
    init {
      System.loadLibrary("tflite-jni")
    }
  }
  external fun loadModel(assetManager: AssetManager, assetName: String)
  external fun runInference(input: FloatArray): FloatArray
}
        </pre>
      </section>
  </devsite-selector>
</div>

Matching the following `loadModel` and `runInference` native functions:

```
#ifdef __cplusplus
extern "C" {
#endif

void Java_com_google_samples_gms_tflite_c_loadModel(
  JNIEnv *env, jobject tflite_jni, jobject asset_manager, jstring asset_name){}
  //...
}

jfloatArray Java_com_google_samples_gms_tflite_c_TfLiteJni_runInference(
  JNIEnv* env, jobject tfliteJni, jfloatArray input) {
  //...
}

#ifdef __cplusplus
}  // extern "C".
#endif
```

You can then call your C functions from your Java/Kotlin code:

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <pre class="prettyprint">
tfLiteHandleTask.onSuccessTask(unused -> {
    TfLiteJni jni = new TfLiteJni();
    jni.loadModel(getAssets(), "add.bin");
    //...
});
    </pre>
    </section>
    <section>
      <h3>Kotlin</h3>
      <pre class="prettyprint">
tfLiteHandleTask.onSuccessTask {
    val jni = TfLiteJni()
    jni.loadModel(assets, "add.bin")
    // ...
}
      </pre>
    </section>
  </devsite-selector>
</div>

### TensorFlow Lite in C code

Include the appropriate API header file to include the TfLite with Google Play
services API:

```
#include "tensorflow/lite/c/c_api.h"
```

You can then use the regular TensorFlow Lite C API:

```
auto model = TfLiteModelCreate(model_asset, model_asset_length);
// ...
auto options = TfLiteInterpreterOptionsCreate();
// ...
auto interpreter = TfLiteInterpreterCreate(model, options);
```

The TensorFlow Lite with Google Play services Native API headers provide the
same API as the regular
[TensorFlow Lite C API](https://www.tensorflow.org/lite/api_docs/c), excluding
features that are deprecated or experimental. For now the functions and types
from the `c_api.h`, `c_api_types.h` and `common.h` headers are available. Please
note that functions from the `c_api_experimental.h` header are not supported.
The documentation can be found
[online](https://www.tensorflow.org/lite/api_docs/c).

You can use functions specific to TensorFlow Lite with Google Play Services by
including `tflite.h`.
