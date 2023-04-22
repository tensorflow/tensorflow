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

import java.util.HashMap;
import java.util.Map;

import org.tensorflow.types.UInt8;

/** Represents the type of elements in a {@link Tensor} as an enum. */
public enum DataType {
  /** 32-bit single precision floating point. */
  FLOAT(1, 4),

  /** 64-bit double precision floating point. */
  DOUBLE(2, 8),

  /** 32-bit signed integer. */
  INT32(3, 4),

  /** 8-bit unsigned integer. */
  UINT8(4, 1),

  /**
   * A sequence of bytes.
   *
   * <p>TensorFlow uses the STRING type for an arbitrary sequence of bytes.
   */
  STRING(7, -1),

  /** 64-bit signed integer. */
  INT64(9, 8),

  /** Boolean. */
  BOOL(10, 1);

  private final int value;
  
  private final int byteSize;

  /**
   * @param value must match the corresponding TF_* value in the TensorFlow C API.
   * @param byteSize size of an element of this type, in bytes, -1 if unknown
   */
  DataType(int value, int byteSize) {
    this.value = value;
    this.byteSize = byteSize;
  }

  /**
   * Returns the size of an element of this type, in bytes, or -1 if element size is variable.
   */
  public int byteSize() {
    return byteSize;
  }

  /** Corresponding value of the TF_DataType enum in the TensorFlow C API. */
  int c() {
    return value;
  }

  // Cached to avoid copying it
  private static final DataType[] values = values();

  static DataType fromC(int c) {
    for (DataType t : values) {
      if (t.value == c) {
        return t;
      }
    }
    throw new IllegalArgumentException(
        "DataType " + c + " is not recognized in Java (version " + TensorFlow.version() + ")");
  }

  /**
   * Returns the DataType of a Tensor whose elements have the type specified by class {@code c}.
   *
   * @param c The class describing the TensorFlow type of interest.
   * @return The {@code DataType} enum corresponding to {@code c}.
   * @throws IllegalArgumentException if objects of {@code c} do not correspond to a TensorFlow
   *     datatype.
   */
  public static DataType fromClass(Class<?> c) {
    DataType dtype = typeCodes.get(c);
    if (dtype == null) {
      throw new IllegalArgumentException(
          c.getName() + " objects cannot be used as elements in a TensorFlow Tensor");
    }
    return dtype;
  }

  private static final Map<Class<?>, DataType> typeCodes = new HashMap<>();

  static {
    typeCodes.put(Float.class, DataType.FLOAT);
    typeCodes.put(Double.class, DataType.DOUBLE);
    typeCodes.put(Integer.class, DataType.INT32);
    typeCodes.put(UInt8.class, DataType.UINT8);
    typeCodes.put(Long.class, DataType.INT64);
    typeCodes.put(Boolean.class, DataType.BOOL);
    typeCodes.put(String.class, DataType.STRING);
  }
}
