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

#ifndef TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_
#define TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_

#include "tensorflow/tools/android/test/jni/object_tracking/geom.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image-inl.h"
#include "tensorflow/tools/android/test/jni/object_tracking/image.h"
#include "tensorflow/tools/android/test/jni/object_tracking/utils.h"

namespace tf_tracking {

typedef uint8_t Code;

class IntegralImage : public Image<uint32_t> {
 public:
  explicit IntegralImage(const Image<uint8_t>& image_base)
      : Image<uint32_t>(image_base.GetWidth(), image_base.GetHeight()) {
    Recompute(image_base);
  }

  IntegralImage(const int width, const int height)
      : Image<uint32_t>(width, height) {}

  void Recompute(const Image<uint8_t>& image_base) {
    SCHECK(image_base.GetWidth() == GetWidth() &&
          image_base.GetHeight() == GetHeight(), "Dimensions don't match!");

    // Sum along first row.
    {
      int x_sum = 0;
      for (int x = 0; x < image_base.GetWidth(); ++x) {
        x_sum += image_base[0][x];
        (*this)[0][x] = x_sum;
      }
    }

    // Sum everything else.
    for (int y = 1; y < image_base.GetHeight(); ++y) {
      uint32_t* curr_sum = (*this)[y];

      // Previously summed pointers.
      const uint32_t* up_one = (*this)[y - 1];

      // Current value pointer.
      const uint8_t* curr_delta = image_base[y];

      uint32_t row_till_now = 0;

      for (int x = 0; x < GetWidth(); ++x) {
        // Add the one above and the one to the left.
        row_till_now += *curr_delta;
        *curr_sum = *up_one + row_till_now;

        // Scoot everything along.
        ++curr_sum;
        ++up_one;
        ++curr_delta;
      }
    }

    SCHECK(VerifyData(image_base), "Images did not match!");
  }

  bool VerifyData(const Image<uint8_t>& image_base) {
    for (int y = 0; y < GetHeight(); ++y) {
      for (int x = 0; x < GetWidth(); ++x) {
        uint32_t curr_val = (*this)[y][x];

        if (x > 0) {
          curr_val -= (*this)[y][x - 1];
        }

        if (y > 0) {
          curr_val -= (*this)[y - 1][x];
        }

        if (x > 0 && y > 0) {
          curr_val += (*this)[y - 1][x - 1];
        }

        if (curr_val != image_base[y][x]) {
          LOGE("Mismatch! %d vs %d", curr_val, image_base[y][x]);
          return false;
        }

        if (GetRegionSum(x, y, x, y) != curr_val) {
          LOGE("Mismatch!");
        }
      }
    }

    return true;
  }

  // Returns the sum of all pixels in the specified region.
  inline uint32_t GetRegionSum(const int x1, const int y1, const int x2,
                               const int y2) const {
    SCHECK(x1 >= 0 && y1 >= 0 &&
          x2 >= x1 && y2 >= y1 && x2 < GetWidth() && y2 < GetHeight(),
          "indices out of bounds! %d-%d / %d, %d-%d / %d, ",
          x1, x2, GetWidth(), y1, y2, GetHeight());

    const uint32_t everything = (*this)[y2][x2];

    uint32_t sum = everything;
    if (x1 > 0 && y1 > 0) {
      // Most common case.
      const uint32_t left = (*this)[y2][x1 - 1];
      const uint32_t top = (*this)[y1 - 1][x2];
      const uint32_t top_left = (*this)[y1 - 1][x1 - 1];

      sum = everything - left - top + top_left;
      SCHECK(sum >= 0, "Both: %d - %d - %d + %d => %d! indices: %d %d %d %d",
            everything, left, top, top_left, sum, x1, y1, x2, y2);
    } else if (x1 > 0) {
      // Flush against top of image.
      // Subtract out the region to the left only.
      const uint32_t top = (*this)[y2][x1 - 1];
      sum = everything - top;
      SCHECK(sum >= 0, "Top: %d - %d => %d!", everything, top, sum);
    } else if (y1 > 0) {
      // Flush against left side of image.
      // Subtract out the region above only.
      const uint32_t left = (*this)[y1 - 1][x2];
      sum = everything - left;
      SCHECK(sum >= 0, "Left: %d - %d => %d!", everything, left, sum);
    }

    SCHECK(sum >= 0, "Negative sum!");

    return sum;
  }

  // Returns the 2bit code associated with this region, which represents
  // the overall gradient.
  inline Code GetCode(const BoundingBox& bounding_box) const {
    return GetCode(bounding_box.left_, bounding_box.top_,
                   bounding_box.right_, bounding_box.bottom_);
  }

  inline Code GetCode(const int x1, const int y1,
                      const int x2, const int y2) const {
    SCHECK(x1 < x2 && y1 < y2, "Bounds out of order!! TL:%d,%d BR:%d,%d",
           x1, y1, x2, y2);

    // Gradient computed vertically.
    const int box_height = (y2 - y1) / 2;
    const int top_sum = GetRegionSum(x1, y1, x2, y1 + box_height);
    const int bottom_sum = GetRegionSum(x1, y2 - box_height, x2, y2);
    const bool vertical_code = top_sum > bottom_sum;

    // Gradient computed horizontally.
    const int box_width = (x2 - x1) / 2;
    const int left_sum = GetRegionSum(x1, y1, x1 + box_width, y2);
    const int right_sum = GetRegionSum(x2 - box_width, y1, x2, y2);
    const bool horizontal_code = left_sum > right_sum;

    const Code final_code = (vertical_code << 1) | horizontal_code;

    SCHECK(InRange(final_code, static_cast<Code>(0), static_cast<Code>(3)),
          "Invalid code! %d", final_code);

    // Returns a value 0-3.
    return final_code;
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(IntegralImage);
};

}  // namespace tf_tracking

#endif  // TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_INTEGRAL_IMAGE_H_
