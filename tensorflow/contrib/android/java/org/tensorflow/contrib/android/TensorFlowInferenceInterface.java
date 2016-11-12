/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.contrib.android;

import android.content.res.AssetManager;
import android.util.Log;
import java.util.Random;

/**
 * JNI wrapper class for the Tensorflow native code.
 *
 * See tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowImageClassifier.java
 * for an example usage.
 * */
public class TensorFlowInferenceInterface {
  private static final String TAG = "TensorFlowInferenceInterface";

  /**
   * A unique identifier used to associate the Java TensorFlowInferenceInterface
   * with its associated native variables.
   * It is accessed via native reflection so any refactoring must also be accompanied
   * by a change to tensorflow_inference_jni.cc.
   */
  private final long id;

  public TensorFlowInferenceInterface() {
    id = new Random().nextLong();

    // Fallback to loading from the default libtensorflow_inference.so
    // only if the app hasn't already loaded a library containing the
    // native TF bindings.
    try {
      testLoaded();
      Log.i(TAG, "Native methods already loaded.");
    } catch (UnsatisfiedLinkError e1) {
      Log.i(TAG, "Loading tensorflow_inference.");
      try {
        System.loadLibrary("tensorflow_inference");
      } catch (UnsatisfiedLinkError e2) {
        throw new RuntimeException(
            "Native TF methods not found; check that the correct native"
                + " libraries are present and loaded.");
      }
    }
  }

  /**
   * Creates a native TensorFlow session for the given model.
   *
   * @param assetManager The AssetManager to use to load the model file.
   * @param model The filepath to the GraphDef proto representing the model.
   * @return The native status returned by TensorFlow. 0 indicates success.
   */
  public native int initializeTensorFlow(AssetManager assetManager, String model);

  /**
   * Runs inference between the previously registered input nodes (via fillNode*)
   * and the requested output nodes. Output nodes can then be queried with the
   * readNode* methods.
   *
   * @param outputNames A list of output nodes which should be filled by the inference pass.
   * @return The native status returned by TensorFlow. 0 indicates success.
   */
  public native int runInference(String[] outputNames);

  /**
   * Cleans up the native variables associated with this Object. initializeTensorFlow() can then
   * be called again to initialize a new session.
   *
   */
  public native void close();

  // Methods for creating a native Tensor and filling it with values.
  public native void fillNodeFloat(String inputName, int[] dims, float[] values);

  public native void fillNodeInt(String inputName, int[] dims, int[] values);

  public native void fillNodeDouble(String inputName, int[] dims, double[] values);

  public native void readNodeFloat(String outputName, float[] values);
  public native void readNodeInt(String outputName, int[] values);
  public native void readNodeDouble(String outputName, double[] values);

  /**
   * Canary method solely for determining if the tensorflow_inference native library should be
   * loaded. If the method is already present, assume that another library is providing the
   * implementations for this class.
   */
  private native void testLoaded();
}
