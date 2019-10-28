/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.support.label;

import android.content.Context;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPrecondtions;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * TensorLabel is an util wrapper for TensorBuffers with meaningful labels on an axis.
 *
 * <p>For example, an image classification model may have an output tensor with shape as {1, 10},
 * where 1 is the batch size and 10 is the number of categories. In fact, on the 2nd axis, we could
 * label each sub-tensor with the name or description of each corresponding category. {@link
 * TensorLabel} could help converting the plain Tensor in {@link TensorBuffer} into a map from
 * predefined labels to sub-tensors. In this case, if provided 10 labels for the 2nd axis, {@link
 * TensorLabel} could convert the original {1, 10} Tensor to a 10 element map, each value of which
 * is Tensor in shape {} (scalar). Usage example:
 *
 * <pre>
 *   TensorBuffer outputTensor = ...;
 *   {@literal List<String>} labels = FileUtil.loadLabels(context, labelFilePath);
 *   // labels the first axis with size greater than one
 *   TensorLabel labeled = new TensorLabel(labels, outputTensor);
 *   // If each sub-tensor has effectively size 1, we can directly get a float value
 *   {@literal Map<String, Float>} probabilities = labeled.getMapWithFloatValue();
 *   // Or get sub-tensors, when each sub-tensor has elements more than 1
 *   {@literal Map<String, TensorBuffer>} subTensors = labeled.getMapWithTensorBuffer();
 * </pre>
 *
 * <p>Note: currently we only support tensor-to-map conversion for the first label with size greater
 * than 1.
 *
 * @see org.tensorflow.lite.support.common.FileUtil#loadLabels(Context, String) to load labels from
 *     a label file (plain text file whose each line is a label) in assets simply.
 */
public class TensorLabel {
  private final Map<Integer, List<String>> axisLabels;
  private final TensorBuffer tensorBuffer;
  private final int[] shape;

  /**
   * Creates a TensorLabel object which is able to label on the axes of multi-dimensional tensors.
   *
   * @param axisLabels A map, whose key is axis id (starting from 0) and value is corresponding
   *     labels. Note: The size of labels should be same with the size of the tensor on that axis.
   * @param tensorBuffer The TensorBuffer to be labeled.
   * @throws NullPointerException if {@code axisLabels} or {@code tensorBuffer} is null, or any
   *     value in {@code axisLabels} is null.
   * @throws IllegalArgumentException if any key in {@code axisLabels} is out of range (compared to
   *     the shape of {@code tensorBuffer}, or any value (labels) has different size with the {@code
   *     tensorBuffer} on the given dimension.
   */
  public TensorLabel(
      @NonNull Map<Integer, List<String>> axisLabels, @NonNull TensorBuffer tensorBuffer) {
    SupportPrecondtions.checkNotNull(axisLabels, "Axis labels cannot be null.");
    SupportPrecondtions.checkNotNull(tensorBuffer, "Tensor Buffer cannot be null.");
    this.axisLabels = axisLabels;
    this.tensorBuffer = tensorBuffer;
    this.shape = tensorBuffer.getShape();
    for (Map.Entry<Integer, List<String>> entry : axisLabels.entrySet()) {
      int axis = entry.getKey();
      SupportPrecondtions.checkArgument(
          axis >= 0 && axis < shape.length, "Invalid axis id: " + axis);
      SupportPrecondtions.checkNotNull(entry.getValue(), "Label list is null on axis " + axis);
      SupportPrecondtions.checkArgument(
          shape[axis] == entry.getValue().size(),
          "Label number " + entry.getValue().size() + " mismatch the shape on axis " + axis);
    }
  }

  /**
   * Creates a TensorLabel object which is able to label on one axis of multi-dimensional tensors.
   *
   * <p>Note: The labels are applied on the first axis whose size is larger than 1. For example, if
   * the shape of the tensor is [1, 10, 3], the labels will be applied on axis 1 (id starting from
   * 0), and size of {@code axisLabels} should be 10 as well.
   *
   * @param axisLabels A list of labels, whose size should be same with the size of the tensor on
   *     the to-be-labeled axis.
   * @param tensorBuffer The TensorBuffer to be labeled.
   */
  public TensorLabel(@NonNull List<String> axisLabels, @NonNull TensorBuffer tensorBuffer) {
    this(makeMap(getFirstAxisWithSizeGreaterThanOne(tensorBuffer), axisLabels), tensorBuffer);
  }

  /**
   * Gets the map with a pair of the label and the corresponding TensorBuffer. Only allow the
   * mapping on the first axis with size greater than 1 currently.
   */
  @NonNull
  public Map<String, TensorBuffer> getMapWithTensorBuffer() {
    int labeledAxis = getFirstAxisWithSizeGreaterThanOne(tensorBuffer);

    Map<String, TensorBuffer> labelToTensorMap = new LinkedHashMap<>();
    SupportPrecondtions.checkArgument(
        axisLabels.containsKey(labeledAxis),
        "get a <String, TensorBuffer> map requires the labels are set on the first non-1 axis.");
    List<String> labels = axisLabels.get(labeledAxis);

    DataType dataType = tensorBuffer.getDataType();
    int typeSize = tensorBuffer.getTypeSize();
    int flatSize = tensorBuffer.getFlatSize();

    // Gets the underlying bytes that could be used to generate the sub-array later.
    ByteBuffer byteBuffer = tensorBuffer.getBuffer();
    byteBuffer.rewind();

    // Note: computation below is only correct when labeledAxis is the first axis with size greater
    // than 1.
    int subArrayLength = flatSize / shape[labeledAxis] * typeSize;
    int i = 0;
    SupportPrecondtions.checkNotNull(labels, "Label list should never be null");
    for (String label : labels) {
      // Gets the corresponding TensorBuffer.
      byteBuffer.position(i * subArrayLength);
      ByteBuffer subBuffer = byteBuffer.slice();
      // ByteBuffer.slice doesn't keep order. Modify it to align with the original one.
      subBuffer.order(byteBuffer.order()).limit(subArrayLength);
      TensorBuffer labelBuffer = TensorBuffer.createDynamic(dataType);
      labelBuffer.loadBuffer(subBuffer, Arrays.copyOfRange(shape, labeledAxis + 1, shape.length));
      labelToTensorMap.put(label, labelBuffer);
      i += 1;
    }
    return labelToTensorMap;
  }

  /**
   * Gets a map that maps label to float. Only allow the mapping on the first axis with size greater
   * than 1, and the axis should be effectively the last axis (which means every sub tensor
   * specified by this axis should have a flat size of 1).
   *
   * @throws IllegalArgumentException if size of a sub tensor on each label is not 1.
   */
  @NonNull
  public Map<String, Float> getMapWithFloatValue() {
    int labeledAxis = getFirstAxisWithSizeGreaterThanOne(tensorBuffer);
    SupportPrecondtions.checkArgument(
        labeledAxis == shape.length - 1,
        "get a <String, Scalar> map is only valid when the only labeled axis is the last one.");
    List<String> labels = axisLabels.get(labeledAxis);
    float[] data = tensorBuffer.getFloatArray();
    SupportPrecondtions.checkArgument(labels.size() == data.length);
    Map<String, Float> result = new LinkedHashMap<>();
    int i = 0;
    for (String label : labels) {
      result.put(label, data[i]);
      i += 1;
    }
    return result;
  }

  private static int getFirstAxisWithSizeGreaterThanOne(@NonNull TensorBuffer tensorBuffer) {
    int[] shape = tensorBuffer.getShape();
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] > 1) {
        return i;
      }
    }
    throw new IllegalArgumentException(
        "Cannot find an axis to label. A valid axis to label should have size larger than 1.");
  }

  // Helper function to wrap the List<String> to a one-entry map.
  private static Map<Integer, List<String>> makeMap(int axis, List<String> labels) {
    Map<Integer, List<String>> map = new LinkedHashMap<>();
    map.put(axis, labels);
    return map;
  }
}
