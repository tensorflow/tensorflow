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

/** Type of elements in a {@link TfLiteTensor}. */
enum DataType {
  /** 32-bit single precision floating point. */
  FLOAT32(1),

  /** 32-bit signed integer. */
  INT32(2),

  /** 8-bit unsigned integer. */
  UINT8(3),

  /** 64-bit signed integer. */
  INT64(4),

  /** A {@link ByteBuffer}. */
  BYTEBUFFER(999);

  private final int value;

  DataType(int value) {
    this.value = value;
  }

  /** Corresponding value of the kTfLite* enum in the TensorFlow Lite CC API. */
  int getNumber() {
    return value;
  }

  /** Converts an integer to the corresponding type. */
  static DataType fromNumber(int c) {
    for (DataType t : values) {
      if (t.value == c) {
        return t;
      }
    }
    throw new IllegalArgumentException(
        "DataType " + c + " is not recognized in Java (version " + TensorFlowLite.version() + ")");
  }

  /** Returns byte size of the type. */
  int elemByteSize() {
    switch (this) {
      case FLOAT32:
        return 4;
      case INT32:
        return 4;
      case UINT8:
        return 1;
      case INT64:
        return 8;
      case BYTEBUFFER:
        return 1;
    }
    throw new IllegalArgumentException("DataType " + this + " is not supported yet");
  }

  // Cached to avoid copying it
  private static final DataType[] values = values();
}
