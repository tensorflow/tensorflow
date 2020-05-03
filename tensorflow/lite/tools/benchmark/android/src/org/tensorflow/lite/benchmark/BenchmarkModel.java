/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.benchmark;

/** Helper class for running a native TensorFlow Lite benchmark. */
class BenchmarkModel {
  static {
    System.loadLibrary("tensorflowlite_benchmark");
  }

  // Executes a standard TensorFlow Lite benchmark according to the provided args.
  //
  // Note that {@code args} will be split by the native execution code.
  public static void run(String args) {
    nativeRun(args);
  }

  private static native void nativeRun(String args);
}
