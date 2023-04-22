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

package org.tensorflow.demo.env;

import android.graphics.Bitmap;
import android.text.TextUtils;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Size class independent of a Camera object.
 */
public class Size implements Comparable<Size>, Serializable {

  // 1.4 went out with this UID so we'll need to maintain it to preserve pending queries when
  // upgrading.
  public static final long serialVersionUID = 7689808733290872361L;

  public final int width;
  public final int height;

  public Size(final int width, final int height) {
    this.width = width;
    this.height = height;
  }

  public Size(final Bitmap bmp) {
    this.width = bmp.getWidth();
    this.height = bmp.getHeight();
  }

  /**
   * Rotate a size by the given number of degrees.
   * @param size Size to rotate.
   * @param rotation Degrees {0, 90, 180, 270} to rotate the size.
   * @return Rotated size.
   */
  public static Size getRotatedSize(final Size size, final int rotation) {
    if (rotation % 180 != 0) {
      // The phone is portrait, therefore the camera is sideways and frame should be rotated.
      return new Size(size.height, size.width);
    }
    return size;
  }

  public static Size parseFromString(String sizeString) {
    if (TextUtils.isEmpty(sizeString)) {
      return null;
    }

    sizeString = sizeString.trim();

    // The expected format is "<width>x<height>".
    final String[] components = sizeString.split("x");
    if (components.length == 2) {
      try {
        final int width = Integer.parseInt(components[0]);
        final int height = Integer.parseInt(components[1]);
        return new Size(width, height);
      } catch (final NumberFormatException e) {
        return null;
      }
    } else {
      return null;
    }
  }

  public static List<Size> sizeStringToList(final String sizes) {
    final List<Size> sizeList = new ArrayList<Size>();
    if (sizes != null) {
      final String[] pairs = sizes.split(",");
      for (final String pair : pairs) {
        final Size size = Size.parseFromString(pair);
        if (size != null) {
          sizeList.add(size);
        }
      }
    }
    return sizeList;
  }

  public static String sizeListToString(final List<Size> sizes) {
    String sizesString = "";
    if (sizes != null && sizes.size() > 0) {
      sizesString = sizes.get(0).toString();
      for (int i = 1; i < sizes.size(); i++) {
        sizesString += "," + sizes.get(i).toString();
      }
    }
    return sizesString;
  }

  public final float aspectRatio() {
    return (float) width / (float) height;
  }

  @Override
  public int compareTo(final Size other) {
    return width * height - other.width * other.height;
  }

  @Override
  public boolean equals(final Object other) {
    if (other == null) {
      return false;
    }

    if (!(other instanceof Size)) {
      return false;
    }

    final Size otherSize = (Size) other;
    return (width == otherSize.width && height == otherSize.height);
  }

  @Override
  public int hashCode() {
    return width * 32713 + height;
  }

  @Override
  public String toString() {
    return dimensionsAsString(width, height);
  }

  public static final String dimensionsAsString(final int width, final int height) {
    return width + "x" + height;
  }
}
