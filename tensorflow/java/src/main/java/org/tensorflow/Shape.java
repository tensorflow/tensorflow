// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package org.tensorflow;

import java.util.Arrays;

/** The possibly partially known shape of a tensor produced by an operation. */
public final class Shape {

  /** Create a Shape representing an unknown number of dimensions. */
  public static Shape unknown() {
    return new Shape(null);
  }

  /** Create a Shape representing a scalar value. */
  public static Shape scalar() {
    return new Shape(new long[0]);
  }

  /**
   * Create a Shape representing an N-dimensional value.
   *
   * <p>Creates a Shape representing an N-dimensional value (N being at least 1), with the provided
   * size for each dimension. A -1 indicates that the size of the corresponding dimension is
   * unknown. For example:
   *
   * <pre>{@code
   * // A 2-element vector.
   * Shape vector = Shape.create(2);
   *
   * // A 2x3 matrix.
   * Shape matrix = Shape.create(2, 3);
   *
   * // A matrix with 4 columns but an unknown number of rows.
   * // This is typically used to indicate the shape of tensors that represent
   * // a variable-sized batch of values. The Shape below might represent a
   * // variable-sized batch of 4-element vectors.
   * Shape batch = Shape.create(-1, 4);
   * }</pre>
   */
  public static Shape make(long firstDimensionSize, long... otherDimensionSizes) {
    long[] shape = new long[otherDimensionSizes.length + 1];
    shape[0] = firstDimensionSize;
    System.arraycopy(otherDimensionSizes, 0, shape, 1, otherDimensionSizes.length);
    return new Shape(shape);
  }

  /**
   * Number of dimensions represented by this shape.
   *
   * @return -1 if the number of dimensions is unknown, 0 if the shape represents a scalar, 1 for a
   *     vector, 2 for a matrix etc.
   */
  public int numDimensions() {
    return shape == null ? -1 : shape.length;
  }

  /**
   * The size of the i-th dimension.
   *
   * @return The size of the requested dimension or -1 if it is unknown.
   */
  public long size(int i) {
    return shape[i];
  }

  /** Succint description of the shape meant for debugging. */
  @Override public String toString() {
    if (shape == null) {
      return "<unknown>";
    }
    return Arrays.toString(shape).replace("-1", "?");
  }

  // Package-private constructor.
  Shape(long[] shape) {
    this.shape = shape;
  }

  // Package-private accessor.
  // The idea is that the public API does not expose the internal array.
  long[] asArray() {
    return shape;
  }

  private long[] shape;
}
