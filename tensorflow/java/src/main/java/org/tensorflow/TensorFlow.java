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

package org.tensorflow;

/** Static utility methods describing the TensorFlow runtime. */
public final class TensorFlow {
  /** Returns the version of the underlying TensorFlow runtime. */
  public static native String version();

  /**
   * All the TensorFlow operations available in this address space.
   *
   * @return A serialized representation of an <a
   *     href="https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto">OpList</a>
   *     protocol buffer, which lists all the available TensorFlow operations.
   */
  public static native byte[] registeredOpList();

  /**
   * Load the dynamic library in filename and register the operations and kernels present in that
   * library.
   *
   * @param filename Path of the dynamic library containing operations and kernels to load.
   * @return Serialized bytes of the <a
   *     href="https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto">OpList</a>
   *     protocol buffer message defining the operations defined in the library.
   * @throws UnsatisfiedLinkError if filename cannot be loaded.
   */
  public static byte[] loadLibrary(String filename) {
    long h = 0;
    try {
      h = libraryLoad(filename);
    } catch (RuntimeException e) {
      throw new UnsatisfiedLinkError(e.getMessage());
    }
    try {
      return libraryOpList(h);
    } finally {
      libraryDelete(h);
    }
  }

  private static native long libraryLoad(String filename);

  private static native void libraryDelete(long handle);

  private static native byte[] libraryOpList(long handle);

  private TensorFlow() {}

  /** Load the TensorFlow runtime C library. */
  static void init() {
    NativeLibrary.load();
  }

  static {
    init();
  }
}
