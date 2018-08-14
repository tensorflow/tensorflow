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
package org.tensorflow.types;

/**
 * Base class for Tensorflow real numbers data types.
 * 
 * <p>Real number Tensorflow data types must extend from this class to satisfy compile-time type safety enforced by
 * operations that only accept real number tensors as some of their operands. This is asserted by extending the 
 * generic parameter of those operands from {@link Number}.
 * 
 * <p>Since data types listed under this pacakge are only used for generic parameterization, there is no real data 
 * manipulation involved. So it is safe to implement the {@link Number} abstract methods by returning zero in all cases.
 */
abstract class TFNumber extends Number {

  private static final long serialVersionUID = 1L;

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
}
