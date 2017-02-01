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

#ifndef THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_
#define THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_

#include <stdint.h>

#include "tensorflow/examples/android/jni/object_tracking/geom.h"
#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"


namespace tf_tracking {

inline void GetUV(const uint8_t* const input, Image<uint8_t>* const u,
                  Image<uint8_t>* const v) {
  const uint8_t* pUV = input;

  for (int row = 0; row < u->GetHeight(); ++row) {
    uint8_t* u_curr = (*u)[row];
    uint8_t* v_curr = (*v)[row];
    for (int col = 0; col < u->GetWidth(); ++col) {
#ifdef __APPLE__
      *u_curr++ = *pUV++;
      *v_curr++ = *pUV++;
#else
      *v_curr++ = *pUV++;
      *u_curr++ = *pUV++;
#endif
    }
  }
}

// Marks every point within a circle of a given radius on the given boolean
// image true.
template <typename U>
inline static void MarkImage(const int x, const int y, const int radius,
                             Image<U>* const img) {
  SCHECK(img->ValidPixel(x, y), "Marking invalid pixel in image! %d, %d", x, y);

  // Precomputed for efficiency.
  const int squared_radius = Square(radius);

  // Mark every row in the circle.
  for (int d_y = 0; d_y <= radius; ++d_y) {
    const int squared_y_dist = Square(d_y);

    const int min_y = MAX(y - d_y, 0);
    const int max_y = MIN(y + d_y, img->height_less_one_);

    // The max d_x of the circle must be strictly greater or equal to
    // radius - d_y for any positive d_y. Thus, starting from radius - d_y will
    // reduce the number of iterations required as compared to starting from
    // either 0 and counting up or radius and counting down.
    for (int d_x = radius - d_y; d_x <= radius; ++d_x) {
      // The first time this critera is met, we know the width of the circle at
      // this row (without using sqrt).
      if (squared_y_dist + Square(d_x) >= squared_radius) {
        const int min_x = MAX(x - d_x, 0);
        const int max_x = MIN(x + d_x, img->width_less_one_);

        // Mark both above and below the center row.
        bool* const top_row_start = (*img)[min_y] + min_x;
        bool* const bottom_row_start = (*img)[max_y] + min_x;

        const int x_width = max_x - min_x + 1;
        memset(top_row_start, true, sizeof(*top_row_start) * x_width);
        memset(bottom_row_start, true, sizeof(*bottom_row_start) * x_width);

        // This row is marked, time to move on to the next row.
        break;
      }
    }
  }
}

#ifdef __ARM_NEON
void CalculateGNeon(
    const float* const vals_x, const float* const vals_y,
    const int num_vals, float* const G);
#endif

// Puts the image gradient matrix about a pixel into the 2x2 float array G.
// vals_x should be an array of the window x gradient values, whose indices
// can be in any order but are parallel to the vals_y entries.
// See http://robots.stanford.edu/cs223b04/algo_tracking.pdf for more details.
inline void CalculateG(const float* const vals_x, const float* const vals_y,
                       const int num_vals, float* const G) {
#ifdef __ARM_NEON
  CalculateGNeon(vals_x, vals_y, num_vals, G);
  return;
#endif

  // Non-accelerated version.
  for (int i = 0; i < num_vals; ++i) {
    G[0] += Square(vals_x[i]);
    G[1] += vals_x[i] * vals_y[i];
    G[3] += Square(vals_y[i]);
  }

  // The matrix is symmetric, so this is a given.
  G[2] = G[1];
}

inline void CalculateGInt16(const int16_t* const vals_x,
                            const int16_t* const vals_y, const int num_vals,
                            int* const G) {
  // Non-accelerated version.
  for (int i = 0; i < num_vals; ++i) {
    G[0] += Square(vals_x[i]);
    G[1] += vals_x[i] * vals_y[i];
    G[3] += Square(vals_y[i]);
  }

  // The matrix is symmetric, so this is a given.
  G[2] = G[1];
}


// Puts the image gradient matrix about a pixel into the 2x2 float array G.
// Looks up interpolated pixels, then calls above method for implementation.
inline void CalculateG(const int window_radius, const float center_x,
                       const float center_y, const Image<int32_t>& I_x,
                       const Image<int32_t>& I_y, float* const G) {
  SCHECK(I_x.ValidPixel(center_x, center_y), "Problem in calculateG!");

  // Hardcoded to allow for a max window radius of 5 (9 pixels x 9 pixels).
  static const int kMaxWindowRadius = 5;
  SCHECK(window_radius <= kMaxWindowRadius,
        "Window %d > %d!", window_radius, kMaxWindowRadius);

  // Diameter of window is 2 * radius + 1 for center pixel.
  static const int kWindowBufferSize =
      (kMaxWindowRadius * 2 + 1) * (kMaxWindowRadius * 2 + 1);

  // Preallocate buffers statically for efficiency.
  static int16_t vals_x[kWindowBufferSize];
  static int16_t vals_y[kWindowBufferSize];

  const int src_left_fixed = RealToFixed1616(center_x - window_radius);
  const int src_top_fixed = RealToFixed1616(center_y - window_radius);

  int16_t* vals_x_ptr = vals_x;
  int16_t* vals_y_ptr = vals_y;

  const int window_size = 2 * window_radius + 1;
  for (int y = 0; y < window_size; ++y) {
    const int fp_y = src_top_fixed + (y << 16);

    for (int x = 0; x < window_size; ++x) {
      const int fp_x = src_left_fixed + (x << 16);

      *vals_x_ptr++ = I_x.GetPixelInterpFixed1616(fp_x, fp_y);
      *vals_y_ptr++ = I_y.GetPixelInterpFixed1616(fp_x, fp_y);
    }
  }

  int32_t g_temp[] = {0, 0, 0, 0};
  CalculateGInt16(vals_x, vals_y, window_size * window_size, g_temp);

  for (int i = 0; i < 4; ++i) {
    G[i] = g_temp[i];
  }
}

inline float ImageCrossCorrelation(const Image<float>& image1,
                                   const Image<float>& image2,
                                   const int x_offset, const int y_offset) {
  SCHECK(image1.GetWidth() == image2.GetWidth() &&
         image1.GetHeight() == image2.GetHeight(),
        "Dimension mismatch! %dx%d vs %dx%d",
        image1.GetWidth(), image1.GetHeight(),
        image2.GetWidth(), image2.GetHeight());

  const int num_pixels = image1.GetWidth() * image1.GetHeight();
  const float* data1 = image1.data();
  const float* data2 = image2.data();
  return ComputeCrossCorrelation(data1, data2, num_pixels);
}

// Copies an arbitrary region of an image to another (floating point)
// image, scaling as it goes using bilinear interpolation.
inline void CopyArea(const Image<uint8_t>& image,
                     const BoundingBox& area_to_copy,
                     Image<float>* const patch_image) {
  VLOG(2) << "Copying from: " << area_to_copy << std::endl;

  const int patch_width = patch_image->GetWidth();
  const int patch_height = patch_image->GetHeight();

  const float x_dist_between_samples = patch_width > 0 ?
      area_to_copy.GetWidth() / (patch_width - 1) : 0;

  const float y_dist_between_samples = patch_height > 0 ?
      area_to_copy.GetHeight() / (patch_height - 1) : 0;

  for (int y_index = 0; y_index < patch_height; ++y_index) {
    const float sample_y =
        y_index * y_dist_between_samples + area_to_copy.top_;

    for (int x_index = 0; x_index < patch_width; ++x_index) {
      const float sample_x =
          x_index * x_dist_between_samples + area_to_copy.left_;

      if (image.ValidInterpPixel(sample_x, sample_y)) {
        // TODO(andrewharp): Do area averaging when downsampling.
        (*patch_image)[y_index][x_index] =
            image.GetPixelInterp(sample_x, sample_y);
      } else {
        (*patch_image)[y_index][x_index] = -1.0f;
      }
    }
  }
}


// Takes a floating point image and normalizes it in-place.
//
// First, negative values will be set to the mean of the non-negative pixels
// in the image.
//
// Then, the resulting will be normalized such that it has mean value of 0.0 and
// a standard deviation of 1.0.
inline void NormalizeImage(Image<float>* const image) {
  const float* const data_ptr = image->data();

  // Copy only the non-negative values to some temp memory.
  float running_sum = 0.0f;
  int num_data_gte_zero = 0;
  {
    float* const curr_data = (*image)[0];
    for (int i = 0; i < image->data_size_; ++i) {
      if (curr_data[i] >= 0.0f) {
        running_sum += curr_data[i];
        ++num_data_gte_zero;
      } else {
        curr_data[i] = -1.0f;
      }
    }
  }

  // If none of the pixels are valid, just set the entire thing to 0.0f.
  if (num_data_gte_zero == 0) {
    image->Clear(0.0f);
    return;
  }

  const float corrected_mean = running_sum / num_data_gte_zero;

  float* curr_data = (*image)[0];
  for (int i = 0; i < image->data_size_; ++i) {
    const float curr_val = *curr_data;
    *curr_data++ = curr_val < 0 ? 0 : curr_val - corrected_mean;
  }

  const float std_dev = ComputeStdDev(data_ptr, image->data_size_, 0.0f);

  if (std_dev > 0.0f) {
    curr_data = (*image)[0];
    for (int i = 0; i < image->data_size_; ++i) {
      *curr_data++ /= std_dev;
    }

#ifdef SANITY_CHECKS
    LOGV("corrected_mean: %1.2f  std_dev: %1.2f", corrected_mean, std_dev);
    const float correlation =
        ComputeCrossCorrelation(image->data(),
                                image->data(),
                                image->data_size_);

    if (std::abs(correlation - 1.0f) > EPSILON) {
      LOG(ERROR) << "Bad image!" << std::endl;
      LOG(ERROR) << *image << std::endl;
    }

    SCHECK(std::abs(correlation - 1.0f) < EPSILON,
           "Correlation wasn't 1.0f:  %.10f", correlation);
#endif
  }
}

}  // namespace tf_tracking

#endif  // THIRD_PARTY_TENSORFLOW_EXAMPLES_ANDROID_JNI_OBJECT_TRACKING_IMAGE_UTILS_H_
