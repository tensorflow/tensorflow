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

import android.graphics.Bitmap;
import java.util.Arrays;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * Implements some stateless image conversion methods.
 *
 * This class is an internal helper for {@link org.tensorflow.lite.support.image}.
 */
class ImageConversions {

  /**
   * Converts an Image in a TensorBuffer to a Bitmap, whose memory is already allocated.
   *
   * Notice: We only support ARGB_8888 at this point.
   *
   * @param buffer The TensorBuffer object representing the image. It should be an UInt8 buffer with
   * 3 dimensions: width, height, channel. Size of each dimension should be positive and the size of
   * channels should be 3 (representing R, G, B).
   * @param bitmap The destination of the conversion. Needs to be created in advance, needs to be
   * mutable, and needs to have the same width and height with the buffer.
   * @throws IllegalArgumentException 1) if the {@code buffer} is not uint8 (e.g. a float buffer),
   * or has an invalid shape. 2) if the {@code bitmap} is not mutable. 3) if the {@code bitmap} has
   * different height or width with the buffer.
   */
  static void convertTensorBufferToBitmap(TensorBuffer buffer, Bitmap bitmap) {
    if (buffer.getDataType() != DataType.UINT8) {
      // We will add support to FLOAT format conversion in the future, as it may need other configs.
      throw new UnsupportedOperationException(String.format(
          "Converting TensorBuffer of type %s to Bitmap is not supported yet.",
          buffer.getDataType()));
    }
    int[] shape = buffer.getShape();
    if (shape.length != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3) {
      throw new IllegalArgumentException(String.format(
          "Buffer shape %s is not valid. 3D TensorBuffer with shape [w, h, 3] is required",
          Arrays.toString(shape)));
    }
    int h = shape[0];
    int w = shape[1];
    if (bitmap.getWidth() != w || bitmap.getHeight() != h) {
      throw new IllegalArgumentException(String.format(
          "Given bitmap has different width or height %s with the expected ones %s.",
          Arrays.toString(new int[]{bitmap.getWidth(), bitmap.getHeight()}),
          Arrays.toString(new int[]{w, h})));
    }
    if (!bitmap.isMutable()) {
      throw new IllegalArgumentException("Given bitmap is not mutable");
    }
    // TODO(b/138904567): Find a way to avoid creating multiple intermediate buffers every time.
    int[] intValues = new int[w * h];
    int[] rgbValues = buffer.getIntArray();
    for (int i = 0, j = 0; i < intValues.length; i++) {
      byte r = (byte) rgbValues[j++];
      byte g = (byte) rgbValues[j++];
      byte b = (byte) rgbValues[j++];
      intValues[i] = ((r << 16) | (g << 8) | b);
    }
    bitmap.setPixels(intValues, 0, w, 0, 0, w, h);
  }

  /**
   * Converts an Image in a Bitmap to a TensorBuffer (3D Tensor: Width-Height-Channel) whose memory
   * is already allocated, or could be dynamically allocated.
   *
   * @param bitmap The Bitmap object representing the image. Currently we only support ARGB_8888
   * config.
   * @param buffer The destination of the conversion. Needs to be created in advance. If it's
   * fixed-size, its flat size should be w*h*3.
   * @throws IllegalArgumentException if the buffer is fixed-size, but the size doesn't match.
   */
  static void convertBitmapToTensorBuffer(Bitmap bitmap, TensorBuffer buffer) {
    int w = bitmap.getWidth();
    int h = bitmap.getHeight();
    int[] intValues = new int[w * h];
    bitmap.getPixels(intValues, 0, w, 0, 0, w, h);
    // TODO(b/138904567): Find a way to avoid creating multiple intermediate buffers every time.
    int[] rgbValues = new int[w * h * 3];
    for (int i = 0, j = 0; i < intValues.length; i++) {
      rgbValues[j++] = ((intValues[i] >> 16) & 0xFF);
      rgbValues[j++] = ((intValues[i] >> 8) & 0xFF);
      rgbValues[j++] = (intValues[i] & 0xFF);
    }
    int[] shape = new int[] {h, w, 3};
    buffer.loadArray(rgbValues, shape);
  }

  // Hide the constructor as the class is static.
  private ImageConversions() {}
}
