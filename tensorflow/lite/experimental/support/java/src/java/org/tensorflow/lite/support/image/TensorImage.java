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
import android.graphics.Bitmap.Config;
import java.nio.ByteBuffer;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.SupportPreconditions;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * TensorImage is the wrapper class for Image object. When using image processing utils in
 * TFLite.support library, it's common to convert image objects in variant types to TensorImage at
 * first.
 *
 * <p>We are adopting a little bit complex strategy to keep data here. In short, a TensorImage
 * object may have 2 potential sources of truth: the real and updated image could be in a Bitmap, or
 * a TensorBuffer, or both. It's mainly for performance, avoiding redundant data conversions.
 *
 * <p>IMPORTANT: The container doesn't own its data. Callers should not modify data objects those
 * are passed to {@link ImageContainer#set(Bitmap)} or {@link ImageContainer#set(TensorBuffer)}.
 *
 * <p>IMPORTANT: All methods are not proved thread-safe. Note: This class still a WIP. Currently, it
 * supports only RGB color space in uint8 (0-255). When getting Bitmap, value of A channel is always
 * set by 0.
 *
 * @see ImageProcessor which is often used for transforming a {@link TensorImage}.
 */
// TODO(b/138906681): Support basic Image properties (ColorType, DataType)
// TODO(b/138907116): Support loading images from TensorBuffer with properties.
// TODO(b/138905544): Support directly loading RGBBytes, YUVBytes and other types if necessary.
public class TensorImage {

  private final ImageContainer container;

  /**
   * Initialize a TensorImage object.
   *
   * Note: The data type of this TensorImage is UINT8, which means it could naturally accept Bitmaps
   * whose pixel value range is [0, 255]. However, any image with float value pixels will not be
   * loaded correctly. In those cases, please use {@link TensorImage(DataType)}.
   */
  public TensorImage() {
    this(DataType.UINT8);
  }

  /**
   * Initializes a TensorImage object with data type specified.
   *
   * <p>Note: The shape of a TensorImage is not fixed. It is determined when {@code load} methods
   * called, and could be change later.
   *
   * @param dataType the expected internal data type of underlying tensor. The type is always fixed
   *     during the lifetime of the {@link TensorImage}. To convert the data type, use {@link
   *     TensorImage#createFrom(TensorImage, DataType)} to create a copy and convert data type at
   *     the same time.
   * @throws IllegalArgumentException if {@code dataType} is neither {@link DataType#UINT8} nor
   *     {@link DataType#FLOAT32}.
   */
  public TensorImage(DataType dataType) {
    SupportPreconditions.checkArgument(
        dataType == DataType.UINT8 || dataType == DataType.FLOAT32,
        "Illegal data type for TensorImage: Only FLOAT32 and UINT8 are accepted");
    container = new ImageContainer(dataType);
  }

  /**
   * Creates a deep-copy of a given {@link TensorImage} and converts internal tensor data type.
   *
   * <p>If the given {@code dataType} is different with {@code src.getDataType()}, an implicit data
   * conversion will be applied. Converting data from {@link DataType#FLOAT32} to {@link
   * DataType#UINT8} may involve default float->int conversion and value clamping, because {@link
   * DataType#UINT8} stores value from 0 to 255 (inclusively).
   *
   * @param src the TensorImage to copy from.
   * @param dataType the expected data type of newly created {@link TensorImage}.
   * @return a TensorImage whose data is copied from {@code src} and data type is {@code dataType}.
   */
  @NonNull
  public static TensorImage createFrom(@NonNull TensorImage src, DataType dataType) {
    TensorImage dst = new TensorImage(dataType);
    if (src.container.isBufferUpdated) {
      dst.container.set(TensorBuffer.createFrom(src.getTensorBuffer(), dataType));
    } else if (src.container.isBitmapUpdated) {
      Bitmap srcBitmap = src.getBitmap();
      dst.container.set(srcBitmap.copy(srcBitmap.getConfig(), srcBitmap.isMutable()));
    }
    return dst;
  }

  /**
   * Loads a Bitmap image object into TensorImage.
   *
   * Important: When loading a bitmap, DO NOT MODIFY the bitmap from the caller side anymore. The
   * {@code TensorImage} object will rely on the bitmap. It will probably modify the bitmap as well.
   * In this method, we perform a zero-copy approach for that bitmap, by simply holding its
   * reference. Use {@code bitmap.copy(bitmap.getConfig(), true)} to create a copy if necessary.
   *
   * Note: To get the best performance, please load images in the same shape to avoid memory
   * re-allocation.
   *
   * @throws IllegalArgumentException if {@code bitmap} is not in ARGB_8888.
   */
  public void load(@NonNull Bitmap bitmap) {
    SupportPreconditions.checkNotNull(bitmap, "Cannot load null bitmap.");
    SupportPreconditions.checkArgument(
        bitmap.getConfig().equals(Config.ARGB_8888), "Only supports loading ARGB_8888 bitmaps.");
    container.set(bitmap);
  }

  /**
   * Loads a float array as RGB pixels into TensorImage, representing the pixels inside.
   *
   * <p>Note: If the TensorImage has data type {@link DataType#UINT8}, numeric casting and clamping
   * will be applied.
   *
   * @param pixels The RGB pixels representing the image.
   * @param shape The shape of the image, should either in form (h, w, 3), or in form (1, h, w, 3).
   */
  public void load(@NonNull float[] pixels, @NonNull int[] shape) {
    checkImageTensorShape(shape);
    TensorBuffer buffer = TensorBuffer.createDynamic(getDataType());
    buffer.loadArray(pixels, shape);
    load(buffer);
  }

  /**
   * Loads an uint8 array as RGB pixels into TensorImage, representing the pixels inside.
   *
   * <p>Note: If the TensorImage has data type {@link DataType#UINT8}, all pixel values will clamp
   * into [0, 255].
   *
   * @param pixels The RGB pixels representing the image.
   * @param shape The shape of the image, should either in form (h, w, 3), or in form (1, h, w, 3).
   */
  public void load(@NonNull int[] pixels, @NonNull int[] shape) {
    checkImageTensorShape(shape);
    TensorBuffer buffer = TensorBuffer.createDynamic(getDataType());
    buffer.loadArray(pixels, shape);
    load(buffer);
  }

  /**
   * Loads a TensorBuffer containing pixel values. The color layout should be RGB.
   *
   * @param buffer The TensorBuffer to load. Its shape should be either (h, w, 3) or (1, h, w, 3).
   */
  public void load(TensorBuffer buffer) {
    checkImageTensorShape(buffer.getShape());
    container.set(buffer);
  }

  /**
   * Returns a bitmap representation of this TensorImage.
   *
   * <p>Important: It's only a reference. DO NOT MODIFY. We don't create a copy here for performance
   * concern, but if modification is necessary, please make a copy.
   *
   * @return a reference to a Bitmap representing the image in ARGB_8888 config. A is always 0.
   * @throws IllegalStateException if the TensorImage never loads data, or if the TensorImage is
   *     holding a float-value image in {@code TensorBuffer}.
   */
  @NonNull
  public Bitmap getBitmap() {
    return container.getBitmap();
  }

  /**
   * Returns a ByteBuffer representation of this TensorImage.
   *
   * <p>Important: It's only a reference. DO NOT MODIFY. We don't create a copy here for performance
   * concern, but if modification is necessary, please make a copy.
   *
   * <p>It's essentially a short cut for {@code getTensorBuffer().getBuffer()}.
   *
   * @return a reference to a ByteBuffer which holds the image data.
   * @throws IllegalStateException if the TensorImage never loads data.
   */
  @NonNull
  public ByteBuffer getBuffer() {
    return container.getTensorBuffer().getBuffer();
  }

  /**
   * Returns a ByteBuffer representation of this TensorImage.
   *
   * <p>Important: It's only a reference. DO NOT MODIFY. We don't create a copy here for performance
   * concern, but if modification is necessary, please make a copy.
   *
   * @return a reference to a TensorBuffer which holds the image data.
   * @throws IllegalStateException if the TensorImage never loads data.
   */
  @NonNull
  public TensorBuffer getTensorBuffer() {
    return container.getTensorBuffer();
  }

  /**
   * Gets the current data type.
   *
   * @return a data type. Currently only UINT8 and FLOAT32 are possible.
   */
  public DataType getDataType() {
    return container.getDataType();
  }

  // Requires tensor shape [h, w, 3] or [1, h, w, 3].
  static void checkImageTensorShape(int[] shape) {
    SupportPreconditions.checkArgument(
        (shape.length == 3 || (shape.length == 4 && shape[0] == 1))
            && shape[shape.length - 3] > 0
            && shape[shape.length - 2] > 0
            && shape[shape.length - 1] == 3,
        "Only supports image shape in (h, w, c) or (1, h, w, c), and channels representing R, G, B"
            + " in order.");
  }

  // Handles RGB image data storage strategy of TensorBuffer.
  private static class ImageContainer {

    private TensorBuffer bufferImage;
    private boolean isBufferUpdated;
    private Bitmap bitmapImage;
    private boolean isBitmapUpdated;

    private final DataType dataType;

    private static final int ARGB_8888_ELEMENT_BYTES = 4;

    ImageContainer(DataType dataType) {
      this.dataType = dataType;
    }

    // Internal method to set the image source-of-truth with a bitmap. The bitmap has to be
    // ARGB_8888.
    void set(Bitmap bitmap) {
      bitmapImage = bitmap;
      isBufferUpdated = false;
      isBitmapUpdated = true;
    }

    // Internal method to set the image source-of-truth with a TensorBuffer.
    void set(TensorBuffer buffer) {
      bufferImage = buffer;
      isBitmapUpdated = false;
      isBufferUpdated = true;
    }

    public DataType getDataType() {
      return dataType;
    }

    // Internal method to update the internal Bitmap data by TensorBuffer data.
    @NonNull
    Bitmap getBitmap() {
      if (isBitmapUpdated) {
        return bitmapImage;
      }
      if (!isBufferUpdated) {
        throw new IllegalStateException("Both buffer and bitmap data are obsolete.");
      }
      if (bufferImage.getDataType() != DataType.UINT8) {
        throw new IllegalStateException(
            "TensorImage is holding a float-value image which is not able to convert a Bitmap.");
      }
      int requiredAllocation = bufferImage.getFlatSize() * ARGB_8888_ELEMENT_BYTES;
      // Create a new bitmap and reallocate memory for it.
      if (bitmapImage == null || bitmapImage.getAllocationByteCount() < requiredAllocation) {
        int[] shape = bufferImage.getShape();
        int h = shape[shape.length - 3];
        int w = shape[shape.length - 2];
        bitmapImage = Bitmap.createBitmap(w, h, Config.ARGB_8888);
      }
      ImageConversions.convertTensorBufferToBitmap(bufferImage, bitmapImage);
      isBitmapUpdated = true;
      return bitmapImage;
    }

    // Internal method to update the internal TensorBuffer data by Bitmap data.
    @NonNull
    TensorBuffer getTensorBuffer() {
      if (isBufferUpdated) {
        return bufferImage;
      }
      SupportPreconditions.checkArgument(
          isBitmapUpdated, "Both buffer and bitmap data are obsolete.");
      int requiredFlatSize = bitmapImage.getWidth() * bitmapImage.getHeight() * 3;
      if (bufferImage == null
          || (!bufferImage.isDynamic() && bufferImage.getFlatSize() != requiredFlatSize)) {
        bufferImage = TensorBuffer.createDynamic(dataType);
      }
      ImageConversions.convertBitmapToTensorBuffer(bitmapImage, bufferImage);
      isBufferUpdated = true;
      return bufferImage;
    }
  }
}
