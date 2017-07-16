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

/** Type of elements in a {@link Tensor}. */
public enum DataType {
  /** 32-bit single precision floating point. */
  FLOAT(1),

  /** 64-bit double precision floating point. */
  DOUBLE(2),

  /** 32-bit signed integer. */
  INT32(3),

  /** 8-bit unsigned integer. */
  UINT8(4),

  /**
   * A sequence of bytes.
   *
   * <p>TensorFlow uses the STRING type for an arbitrary sequence of bytes.
   */
  STRING(7),

  /** 64-bit signed integer. */
  INT64(9),

  /** Boolean. */
  BOOL(10);

  private final int value;

  // The integer value must match the corresponding TF_* value in the TensorFlow C API.
  DataType(int value) {
    this.value = value;
  }

  /** Corresponding value of the TF_DataType enum in the TensorFlow C API. */
  int c() {
    return value;
  }
  
  // Cached to avoid copying it
  final private static DataType[] values = values();

  static DataType fromC(int c) {
    for (DataType t : values) {
      if (t.value == c)
        return t;
    }
    throw new IllegalArgumentException(
        "DataType " + c + " is not recognized in Java (version " + TensorFlow.version() + ")");
  }
}
