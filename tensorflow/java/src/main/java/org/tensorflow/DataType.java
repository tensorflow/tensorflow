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
import org.tensorflow.types.TFBool;
import org.tensorflow.types.TFDouble;
import org.tensorflow.types.TFFloat;
import org.tensorflow.types.TFInt32;
import org.tensorflow.types.TFInt64;
import org.tensorflow.types.TFString;
import org.tensorflow.types.TFType;
import org.tensorflow.types.TFUInt8;

/**
 * Represents the type of elements in a {@link Tensor} as an enum.
 *
 * @see org.tensorflow.types
 */
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
  private static final DataType[] values = values();

  static DataType fromC(int c) {
    for (DataType t : values) {
      if (t.value == c) return t;
    }
    throw new IllegalArgumentException(
        "DataType " + c + " is not recognized in Java (version " + TensorFlow.version() + ")");
  }

  /**
   * Returns the DataType value corresponding to a TensorFlow type class.
   *
   * @param c The class describing the TensorFlow type of interest.
   */
  public static DataType fromClass(Class<? extends TFType> c) {
    DataType dtype = typeCodes.get(c);
    if (dtype == null) {
      throw new IllegalArgumentException("" + c + " is not a TensorFlow type.");
    }
    return dtype;
  }

  private static final Map<Class<?>, DataType> typeCodes = new HashMap<>();

  static {
    typeCodes.put(TFFloat.class, DataType.FLOAT);
    typeCodes.put(TFDouble.class, DataType.DOUBLE);
    typeCodes.put(TFInt32.class, DataType.INT32);
    typeCodes.put(TFUInt8.class, DataType.UINT8);
    typeCodes.put(TFInt64.class, DataType.INT64);
    typeCodes.put(TFBool.class, DataType.BOOL);
    typeCodes.put(TFString.class, DataType.STRING);
  }

  /**
   * Returns the zero value of type described by {@code c}, or null if the type (e.g., string) is
   * not numeric and therefore has no zero value.
   *
   * @param c The class describing the TensorFlow type of interest.
   * TODO(andrewmyers): Not clear we want this at all; probably this is the wrong
   * place for it if we do want it.
   */
  public static Object zeroValue(Class<? extends TFType> c) {
    return zeros.get(c);
  }

  private static final Map<Class<?>, Object> zeros = new HashMap<>();

  static {
    zeros.put(TFFloat.class, 0.0f);
    zeros.put(TFDouble.class, 0.0);
    zeros.put(TFInt32.class, 0);
    zeros.put(TFUInt8.class, (byte) 0);
    zeros.put(TFInt64.class, 0L);
    zeros.put(TFBool.class, false);
    zeros.put(TFString.class, null); // no zero value
  }
}
