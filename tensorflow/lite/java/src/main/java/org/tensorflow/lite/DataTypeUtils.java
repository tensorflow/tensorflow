/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

/**
 * Utility methods for DataType.
 */
class DataTypeUtils {

  private DataTypeUtils() {}

  /** Gets string names of the data type. */
  static String toStringName(DataType dataType) {
    switch (dataType) {
      case FLOAT32:
        return "float";
      case INT32:
        return "int";
      case INT16:
        return "short";
      case INT8:
      case UINT8:
        return "byte";
      case INT64:
        return "long";
      case BOOL:
        return "bool";
      case STRING:
        return "string";
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + dataType + " is not supported yet");
  }

  /** Converts a C TfLiteType enum value to the corresponding type. */
  static DataType fromC(int c) {
    switch (c) {
      case 1:
        return DataType.FLOAT32;
      case 2:
        return DataType.INT32;
      case 3:
        return DataType.UINT8;
      case 4:
        return DataType.INT64;
      case 5:
        return DataType.STRING;
      case 6:
        return DataType.BOOL;
      case 7:
        return DataType.INT16;
      case 9:
        return DataType.INT8;
      default: // continue below to handle unsupported C types.
    }
    throw new IllegalArgumentException(
        "DataType error: DataType " + c + " is not recognized in Java.");
  }
}
