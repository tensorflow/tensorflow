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

package org.tensorflow.types;

/** Represents an 8-bit unsigned integer. */
public class UInt8 extends Number {

  private static final long serialVersionUID = 1L;
  
  // This class is only used for generic parameterization and is not instantiable. Thus,
  // it is safe to implement the Number abstract methods with all zeros, as they will
  // never be invoked.

  @Override
  public double doubleValue() {
    return 0.0;
  }

  @Override
  public float floatValue() {
    return 0.0f;
  }

  @Override
  public int intValue() {
    return 0;
  }

  @Override
  public long longValue() {
    return 0L;
  }

  private UInt8() {}
}
