/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite;

/** Represents the type of elements in a TensorFlow Lite {@link Tensor} as an enum. */
public enum DataType {
  /** 32-bit single precision floating point. */
  FLOAT32(1),

  /** 32-bit signed integer. */
  INT32(2),

  /** 8-bit unsigned integer. */
  UINT8(3),

  /** 64-bit signed integer. */
  INT64(4),

  /** Strings. */
  STRING(5),

  /** Bool. */
  BOOL(6),

  /** 16-bit signed integer. */
  INT16(7),

  /** 8-bit signed integer. */
  INT8(9);

  private final int value;

  DataType(int value) {
    this.value = value;
  }

  /** Returns the size of an element of this type, in bytes, or -1 if element size is variable. */
  public int byteSize() {
    switch (this) {
      case FLOAT32:
      case INT32:
        return 4;
      case INT16:
        return 2;
      case INT8:
      case UINT8:
        return 1;
      case INT64:
        return 8;
      case BOOL:
        // Boolean size is JVM-dependent.
        return -1;
      case STRING:
        return -1;
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + this + " is not supported yet");
  }

  /** Corresponding value of the TfLiteType enum in the TensorFlow Lite C API. */
  int c() {
    return value;
  }

  // Cached to avoid copying it
  private static final DataType[] values = values();
}
