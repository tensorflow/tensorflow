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

package org.tensorflow.lite.support.image;

import android.graphics.PointF;
import android.graphics.RectF;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import org.tensorflow.lite.support.common.Operator;
import org.tensorflow.lite.support.common.SequentialProcessor;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.image.ops.TensorOperatorWrapper;

/**
 * ImageProcessor is a helper class for preprocessing and postprocessing {@link TensorImage}. It
 * could transform a {@link TensorImage} to another by executing a chain of {@link ImageOperator}.
 *
 * <p>Example Usage:
 *
 * <pre>
 *   ImageProcessor processor = new ImageProcessor.Builder()
 *       .add(new ResizeOp(224, 224, ResizeMethod.NEAREST_NEIGHBOR)
 *       .add(new Rot90Op())
 *       .add(new NormalizeOp(127.5f, 127.5f))
 *       .build();
 *   TensorImage anotherTensorImage = processor.process(tensorImage);
 * </pre>
 *
 * <p><b>WARNING:</b> Instances of an {@code ImageProcessor} are <b>not</b> thread-safe with {@link
 * #updateNumberOfRotations}. Updating the number of rotations and then processing images (using
 * {@link #process}) must be protected from concurrent access. It is recommended to create separate
 * {@code ImageProcessor} instances for each thread. If multiple threads access a {@code
 * ImageProcessor} concurrently, it must be synchronized externally.
 *
 * @see ImageProcessor.Builder to build a {@link ImageProcessor} instance
 * @see ImageProcessor#process(TensorImage) to apply the processor on a {@link TensorImage}
 */
public class ImageProcessor extends SequentialProcessor<TensorImage> {
  private ImageProcessor(Builder builder) {
    super(builder);
  }

  /**
   * Transforms a point from coordinates system of the result image back to the one of the input
   * image.
   *
   * @param point the point from the result coordinates system.
   * @param inputImageHeight the height of input image.
   * @param inputImageWidth the width of input image.
   * @return the point with the coordinates from the coordinates system of the input image.
   */
  public PointF inverseTransform(PointF point, int inputImageHeight, int inputImageWidth) {
    List<Integer> widths = new ArrayList<>();
    List<Integer> heights = new ArrayList<>();
    int currentWidth = inputImageWidth;
    int currentHeight = inputImageHeight;
    for (Operator<TensorImage> op : operatorList) {
      widths.add(currentWidth);
      heights.add(currentHeight);
      ImageOperator imageOperator = (ImageOperator) op;
      int newHeight = imageOperator.getOutputImageHeight(currentHeight, currentWidth);
      int newWidth = imageOperator.getOutputImageWidth(currentHeight, currentWidth);
      currentHeight = newHeight;
      currentWidth = newWidth;
    }
    ListIterator<Operator<TensorImage>> opIterator = operatorList.listIterator(operatorList.size());
    ListIterator<Integer> widthIterator = widths.listIterator(widths.size());
    ListIterator<Integer> heightIterator = heights.listIterator(heights.size());
    while (opIterator.hasPrevious()) {
      ImageOperator imageOperator = (ImageOperator) opIterator.previous();
      int height = heightIterator.previous();
      int width = widthIterator.previous();
      point = imageOperator.inverseTransform(point, height, width);
    }
    return point;
  }

  /**
   * Transforms a rectangle from coordinates system of the result image back to the one of the input
   * image.
   *
   * @param rect the rectangle from the result coordinates system.
   * @param inputImageHeight the height of input image.
   * @param inputImageWidth the width of input image.
   * @return the rectangle with the coordinates from the coordinates system of the input image.
   */
  public RectF inverseTransform(RectF rect, int inputImageHeight, int inputImageWidth) {
    // when rotation is involved, corner order may change - top left changes to bottom right, .etc
    PointF p1 =
        inverseTransform(new PointF(rect.left, rect.top), inputImageHeight, inputImageWidth);
    PointF p2 =
        inverseTransform(new PointF(rect.right, rect.bottom), inputImageHeight, inputImageWidth);
    return new RectF(
        Math.min(p1.x, p2.x), Math.min(p1.y, p2.y), Math.max(p1.x, p2.x), Math.max(p1.y, p2.y));
  }

  /**
   * The Builder to create an ImageProcessor, which could be executed later.
   *
   * @see #add(TensorOperator) to add a general TensorOperator
   * @see #add(ImageOperator) to add an ImageOperator
   * @see #build() complete the building process and get a built Processor
   */
  public static class Builder extends SequentialProcessor.Builder<TensorImage> {
    public Builder() {
      super();
    }

    /**
     * Adds an {@link ImageOperator} into the Operator chain.
     *
     * @param op the Operator instance to be executed then
     */
    public Builder add(ImageOperator op) {
      super.add(op);
      return this;
    }

    /**
     * Adds a {@link TensorOperator} into the Operator chain. In execution, the processor calls
     * {@link TensorImage#getTensorBuffer()} to transform the {@link TensorImage} by transforming
     * the underlying {@link org.tensorflow.lite.support.tensorbuffer.TensorBuffer}.
     *
     * @param op the Operator instance to be executed then
     */
    public Builder add(TensorOperator op) {
      return add(new TensorOperatorWrapper(op));
    }

    /** Completes the building process and gets the {@link ImageProcessor} instance. */
    @Override
    public ImageProcessor build() {
      return new ImageProcessor(this);
    }
  }

  /**
   * Updates the number of rotations for the first {@link Rot90Op} in this {@link ImageProcessor}.
   *
   * <p><b>WARNING:</b>this method is <b>not</b> thread-safe. Updating the number of rotations and
   * then processing images (using {@link #process}) must be protected from concurrent access with
   * additional synchronization.
   *
   * @param k the number of rotations
   * @throws IllegalStateException if {@link Rot90Op} has not been added to this {@link
   *     ImageProcessor}
   */
  public void updateNumberOfRotations(int k) {
    updateNumberOfRotations(k, /*occurrence=*/ 0);
  }

  /**
   * Updates the number of rotations for the {@link Rot90Op} specified by {@code occurrence} in this
   * {@link ImageProcessor}.
   *
   * <p><b>WARNING:</b>this method is <b>not</b> thread-safe. Updating the number of rotations and
   * then processing images (using {@link #process}) must be protected from concurrent access with
   * additional synchronization.
   *
   * @param k the number of rotations
   * @param occurrence the index of perticular {@link Rot90Op} in this {@link ImageProcessor}. For
   *     example, if the second {@link Rot90Op} needs to be updated, {@code occurrence} should be
   *     set to 1.
   * @throws IndexOutOfBoundsException if {@code occurrence} is negative or is not less than the
   *     number of {@link Rot90Op} in this {@link ImageProcessor}
   * @throws IllegalStateException if {@link Rot90Op} has not been added to this {@link
   *     ImageProcessor}
   */
  public synchronized void updateNumberOfRotations(int k, int occurrence) {
    SupportPreconditions.checkState(
        operatorIndex.containsKey(Rot90Op.class.getName()),
        "The Rot90Op has not been added to the ImageProcessor.");

    List<Integer> indexes = operatorIndex.get(Rot90Op.class.getName());
    SupportPreconditions.checkElementIndex(occurrence, indexes.size(), "occurrence");

    // The index of the Rot90Op to be replaced in operatorList.
    int index = indexes.get(occurrence);
    Rot90Op newRot = new Rot90Op(k);
    operatorList.set(index, newRot);
  }
}
