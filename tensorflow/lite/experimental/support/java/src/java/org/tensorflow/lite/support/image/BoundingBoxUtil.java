/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.image;

import static org.tensorflow.lite.support.common.SupportPreconditions.checkArgument;

import android.graphics.RectF;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * Helper class for converting values that represents bounding boxes into rectangles.
 *
 * <p>The class provides a static function to create bounding boxes as {@link RectF} from different
 * types of configurations.
 *
 * <p>Generally, a bounding box could be represented by 4 float values, but the values could be
 * interpreted in many ways. We now support 3 {@link Type} of configurations, and the order of
 * elements in each type is configurable as well.
 */
public final class BoundingBoxUtil {

  /** Denotes how a bounding box is represented. */
  public enum Type {
    /**
     * Represents the bounding box by using the combination of boundaries, {left, top, right,
     * bottom}. The default order is {left, top, right, bottom}. Other orders can be indicated by an
     * index array.
     */
    BOUNDARIES,
    /**
     * Represents the bounding box by using the upper_left corner, width and height. The default
     * order is {upper_left_x, upper_left_y, width, height}. Other orders can be indicated by an
     * index array.
     */
    UPPER_LEFT,
    /**
     * Represents the bounding box by using the center of the box, width and height. The default
     * order is {center_x, center_y, width, height}. Other orders can be indicated by an index
     * array.
     */
    CENTER,
  }

  /** Denotes if the coordinates are actual pixels or relative ratios. */
  public enum CoordinateType {
    /** The coordinates are relative ratios in range [0, 1]. */
    RATIO,
    /** The coordinates are actual pixel values. */
    PIXEL
  }

  /**
   * Creates a list of bounding boxes from a {@link TensorBuffer} which represents bounding boxes.
   *
   * @param tensor holds the data representing some boxes.
   * @param valueIndex denotes the order of the elements defined in each bounding box type. An empty
   *     index array represent the default order of each bounding box type. For example, to denote
   *     the default order of BOUNDARIES, {left, top, right, bottom}, the index should be {0, 1, 2,
   *     3}. To denote the order {left, right, top, bottom}, the order should be {0, 2, 1, 3}.
   *     <p>The index array can be applied to all bounding box types to adjust the order of their
   *     corresponding underlying elements.
   * @param boundingBoxAxis specifies the index of the dimension that represents bounding box. The
   *     size of that dimension is required to be 4. Index here starts from 0. For example, if the
   *     tensor has shape 4x10, the axis for bounding boxes is likely to be 0. For shape 10x4, the
   *     axis is likely to be 1 (or -1, equivalently).
   * @param type defines how values should be converted into boxes. See {@link Type}
   * @param coordinateType defines how values are interpreted to coordinates. See {@link
   *     CoordinateType}
   * @param height the height of the image which the boxes belong to. Only has effects when {@code
   *     coordinateType} is {@link CoordinateType#RATIO}
   * @param width the width of the image which the boxes belong to. Only has effects when {@code
   *     coordinateType} is {@link CoordinateType#RATIO}
   * @return A list of bounding boxes that the {@code tensor} represents. All dimensions except
   *     {@code boundingBoxAxis} will be collapsed with order kept. For example, given {@code
   *     tensor} with shape {1, 4, 10, 2} and {@code boundingBoxAxis = 1}, The result will be a list
   *     of 20 bounding boxes.
   * @throws IllegalArgumentException if size of bounding box dimension (set by {@code
   *     boundingBoxAxis}) is not 4.
   * @throws IllegalArgumentException if {@code boundingBoxAxis} is not in {@code (-(D+1), D)} where
   *     {@code D} is the number of dimensions of the {@code tensor}.
   * @throws IllegalArgumentException if {@code tensor} has data type other than {@link
   *     DataType#FLOAT32}.
   */
  public static List<RectF> convert(
      TensorBuffer tensor,
      int[] valueIndex,
      int boundingBoxAxis,
      Type type,
      CoordinateType coordinateType,
      int height,
      int width) {
    int[] shape = tensor.getShape();
    checkArgument(
        boundingBoxAxis >= -shape.length && boundingBoxAxis < shape.length,
        String.format(
            "Axis %d is not in range (-(D+1), D), where D is the number of dimensions of input"
                + " tensor (shape=%s)",
            boundingBoxAxis, Arrays.toString(shape)));
    if (boundingBoxAxis < 0) {
      boundingBoxAxis = shape.length + boundingBoxAxis;
    }
    checkArgument(
        shape[boundingBoxAxis] == 4,
        String.format(
            "Size of bounding box dimension %d is not 4. Got %d in shape %s",
            boundingBoxAxis, shape[boundingBoxAxis], Arrays.toString(shape)));
    checkArgument(
        valueIndex.length == 4,
        String.format(
            "Bounding box index array length %d is not 4. Got index array %s",
            valueIndex.length, Arrays.toString(valueIndex)));
    checkArgument(
        tensor.getDataType() == DataType.FLOAT32,
        "Bounding Boxes only create from FLOAT32 buffers. Got: " + tensor.getDataType().name());
    List<RectF> boundingBoxList = new ArrayList<>();
    // Collapse dimensions to {a, 4, b}. So each bounding box could be represent as (i, j), and its
    // four values are (i, k, j), where 0 <= k < 4. We can compute the 4 flattened index by
    // i * 4b + k * b + j.
    int a = 1;
    for (int i = 0; i < boundingBoxAxis; i++) {
      a *= shape[i];
    }
    int b = 1;
    for (int i = boundingBoxAxis + 1; i < shape.length; i++) {
      b *= shape[i];
    }
    float[] values = new float[4];
    ByteBuffer byteBuffer = tensor.getBuffer();
    byteBuffer.rewind();
    FloatBuffer floatBuffer = byteBuffer.asFloatBuffer();
    for (int i = 0; i < a; i++) {
      for (int j = 0; j < b; j++) {
        for (int k = 0; k < 4; k++) {
          values[k] = floatBuffer.get((i * 4 + k) * b + j);
        }
        boundingBoxList.add(
            convertOneBoundingBox(values, valueIndex, type, coordinateType, height, width));
      }
    }
    byteBuffer.rewind();
    return boundingBoxList;
  }

  private static RectF convertOneBoundingBox(
      float[] values,
      int[] valueIndex,
      Type type,
      CoordinateType coordinateType,
      int height,
      int width) {
    float[] orderedValues = new float[4];
    for (int i = 0; i < 4; i++) {
      orderedValues[i] = values[valueIndex[i]];
    }
    return convertOneBoundingBox(orderedValues, type, coordinateType, height, width);
  }

  private static RectF convertOneBoundingBox(
      float[] values, Type type, CoordinateType coordinateType, int height, int width) {
    switch (type) {
      case BOUNDARIES:
        return convertFromBoundaries(values, coordinateType, height, width);
      case UPPER_LEFT:
      case CENTER:
        // TODO(b/150824448): convertFrom{UpperLeft, Center}
        throw new IllegalArgumentException("BoundingBox.Type " + type + " is not yet supported.");
    }
    throw new IllegalArgumentException("Cannot recognize BoundingBox.Type " + type);
  }

  private static RectF convertFromBoundaries(
      float[] values, CoordinateType coordinateType, int height, int width) {
    if (coordinateType == CoordinateType.RATIO) {
      return new RectF(
          values[0] * width, values[1] * height, values[2] * width, values[3] * height);
    } else {
      return new RectF(values[0], values[1], values[2], values[3]);
    }
  }

  // Private constructor to prevent initialization.
  private BoundingBoxUtil() {}
}
