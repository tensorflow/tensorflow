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

package org.tensorflow.lite;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.EnumSet;
import javax.imageio.ImageIO;

/** Utility for interacting with test-specific data. */
public abstract class TestUtils {

  private static final float DEFAULT_IMAGE_MEAN = 127.5f;
  private static final float DEFAULT_IMAGE_STD = 127.5f;

  public static MappedByteBuffer getTestFileAsBuffer(String path) {
    try (FileChannel fileChannel =
        (FileChannel)
            Files.newByteChannel(new File(path).toPath(), EnumSet.of(StandardOpenOption.READ))) {
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileChannel.size());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static boolean supportsFilePaths() {
    return true;
  }

  public static ByteBuffer getTestImageAsByteBuffer(String path) {
    File imageFile = new File(path);
    try {
      BufferedImage image = ImageIO.read(imageFile);
      return toByteBuffer(image);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static ByteBuffer getTestImageAsFloatByteBuffer(String path) {
    File imageFile = new File(path);
    try {
      BufferedImage image = ImageIO.read(imageFile);
      return toFloatByteBuffer(image);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  private static ByteBuffer toByteBuffer(BufferedImage image) {
    ByteBuffer imgData =
        ByteBuffer.allocateDirect(image.getHeight() * image.getWidth() * 3)
            .order(ByteOrder.nativeOrder());
    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int val = image.getRGB(x, y);
        imgData.put((byte) ((val >> 16) & 0xFF));
        imgData.put((byte) ((val >> 8) & 0xFF));
        imgData.put((byte) (val & 0xFF));
      }
    }
    return imgData;
  }

  private static ByteBuffer toFloatByteBuffer(BufferedImage image) {
    return toFloatByteBuffer(image, DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD);
  }

  private static ByteBuffer toFloatByteBuffer(
      BufferedImage image, float imageMean, float imageStd) {
    ByteBuffer imgData =
        ByteBuffer.allocateDirect(image.getHeight() * image.getWidth() * 3 * 4)
            .order(ByteOrder.nativeOrder());
    for (int y = 0; y < image.getHeight(); y++) {
      for (int x = 0; x < image.getWidth(); x++) {
        int pixelValue = image.getRGB(x, y);
        imgData.putFloat((((pixelValue >> 16) & 0xFF) - imageMean) / imageStd);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) - imageMean) / imageStd);
        imgData.putFloat(((pixelValue & 0xFF) - imageMean) / imageStd);
      }
    }
    return imgData;
  }

  private TestUtils() {}
}
