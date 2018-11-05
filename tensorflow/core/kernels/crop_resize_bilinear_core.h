/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_CROP_RESIZE_BILINEAR_CORE_H_
#define TENSORFLOW_CORE_KERNELS_CROP_RESIZE_BILINEAR_CORE_H_

// only include intrinsics when the appropriate flags call for it,
// since these headers only exists on x86 platforms.
#ifdef __SSE4_1__
#include <smmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>

namespace tensorflow {
namespace {

// Compute the interpolation indices only once.
struct CachedInterpolation {
  int lower;  // Lower source index used in the interpolation
  int upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

bool compute_single_interpolation_weight(const int in_size,
                                         const float out2in_scale,
                                         const float out2in_start,
                                         const bool clip, const int i,
                                         int* lower, int* upper, float* lerp) {
  const float in = i * out2in_scale + out2in_start;
  *lower = (int)floor(in);
  *upper = (int)ceil(in);
  *lerp = (float)(in - (float)*lower);
  if (clip) {
    if (*lower < 0)
      *lower = 0;
    else if (*lower >= in_size)
      *lower = in_size - 1;
    if (*upper < 0)
      *upper = 0;
    else if (*upper >= in_size)
      *upper = in_size - 1;
    return true;
  } else {
    return (*lower >= 0 && *upper < in_size) ? true : false;
  }
}
/**
 * Compute interpolation values for output indexes in range
 * [out_start,out_start+out_size-1].
 * Returns true if all output indexes have lower and upper (input) indexes
 * within range [0,in_size-1].
 */
bool compute_interpolation_weights(const int min_i, const int max_i,
                                   const int in_size, const float out2in_scale,
                                   const float out2in_start, const bool clip,
                                   CachedInterpolation* interpolation) {
  bool rval = true;
  int num_i = max_i - min_i + 1;
  for (int i = 0; i < num_i; ++i) {
    if (!compute_single_interpolation_weight(
            in_size, out2in_scale, out2in_start, clip, i + min_i,
            &interpolation[i].lower, &interpolation[i].upper,
            &interpolation[i].lerp)) {
      rval = false;
    }
  }
  return rval;
}
/**
 * Compatibility method for resize_bilinear_op.cc
 */
void compute_interpolation_weights(const int out_size, const int in_size,
                                   const float out2in_scale,
                                   CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  const bool clip = true;
  if (!compute_interpolation_weights(0, out_size - 1, in_size, out2in_scale,
                                     0.0f, clip, interpolation)) {
    // Should never happen, check for it anyway
    printf(
        "Warning! Interpolation values have lower,upper indexes outside of "
        "range [0,in_size-1]\n");
  }
}
/**
 * Compute minimum and maximum (output) i where both lower and upper (input) is
 * in range [0,in_size-1]
 * If no values of i satisfy condition, min_i = in_size, max_i = -1 and method
 * returns false.
 * Returns true if min_i >= max_i.
 */
bool compute_minmax_indexes(const int out_size, const int in_size,
                            const float out2in_scale, const float out2in_start,
                            int* min_i, int* max_i) {
  *min_i = out_size;
  *max_i = -1;
  int lower, upper;
  float lerp;
  for (int i = 0; i < out_size; ++i) {
    if (compute_single_interpolation_weight(in_size, out2in_scale, out2in_start,
                                            false, i, &lower, &upper, &lerp)) {
      if (i < *min_i) *min_i = i;
      if (i > *max_i) *max_i = i;
    }
  }
  return (*min_i <= *max_i) ? true : false;
}
/**
 * Compute interpolation weights for crop_and_resize_op.cc
 * Also computes extrapolation areas.
 * Returns true if at least one point requires interpolation, false otherwise.
 */
bool compute_interpolation_weights(
    const int out_size, const int in_size,
    const float x1,  // lower bounding box, crop region starts at in_size*x1
    const float x2,  // upper bounding box, crop region ends at in_size*x2
    int* min_i, int* max_i, std::vector<CachedInterpolation>* interpolation) {
  float out2in_start = out_size > 1
                           ? (float)(in_size - 1) * (float)x1
                           : (float)(in_size - 1) * (float)(x1 + x2) / 2.0f;
  float out2in_scale =
      out_size > 1
          ? (float)(x2 - x1) * (float)(in_size - 1) / (float)(out_size - 1)
          : 0.0f;
  if (compute_minmax_indexes(out_size, in_size, out2in_scale, out2in_start,
                             min_i, max_i)) {
    interpolation->resize(*max_i - *min_i + 1);
    bool all_inputs_ok = compute_interpolation_weights(
        *min_i, *max_i, in_size, out2in_scale, out2in_start, false,
        interpolation->data());
    if (!all_inputs_ok) {
      // should never happen, purpose of compute_minmax_indexes is to ensure
      // that all inputs are ok.
      printf(
          "Error! compute_interpolation_weights returned input indexes outside "
          "valid range - SEGV will likely ensue.\n");
    }
    return true;
  } else {
    return false;
  }
}

/**
 * Cast float v to type U with range clamping.
 *
 * If v<min_val, return value is clamped to u_min_val. similarly if v>max_val,
 * return value is clamped to u_max_val.
 */
template <typename U>
U cast_to(float v, float min_val, float max_val, U u_min_val, U u_max_val);
template <typename U>
U cast_to(float v, float min_val, float max_val, U u_min_val, U u_max_val) {
  if (v < min_val)
    return u_min_val;
  else if (v > max_val)
    return u_max_val;
  else
    return static_cast<U>(v);
}
/**
 * no-op cast from float to float.
 */
template <>
float cast_to<float>(float v, float min_val, float max_val, float u_min_val,
                     float u_max_val) {
  return v;
}

float compute_lerp(const float top_left, const float top_right,
                   const float bottom_left, const float bottom_right,
                   const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 * Accepts input tensors of type T and produces output tensors of type U.
 * Optionally flips horizontal and/or vertical axis.
 */
template <typename T, typename U>
void crop_resize_single_image(const T* image, const int64 in_height,
                              const int64 in_width, const int64 out_height,
                              const int64 out_width, const int channels,
                              const int min_ix, const int max_ix,
                              const CachedInterpolation* xs, const int min_iy,
                              const int max_iy, const CachedInterpolation* ys,
                              const float extrapolated_value, const bool flip_x,
                              const bool flip_y,
                              U* output) TF_ATTRIBUTE_NOINLINE;
template <typename T, typename U>
void crop_resize_single_image(const T* image, const int64 in_height,
                              const int64 in_width, const int64 out_height,
                              const int64 out_width, const int channels,
                              const int min_ix, const int max_ix,
                              const CachedInterpolation* xs, const int min_iy,
                              const int max_iy, const CachedInterpolation* ys,
                              const float extrapolated_value, const bool flip_x,
                              const bool flip_y, U* output) {
  const int64 in_row_size = in_width * channels;
  const int64 out_row_size = out_width * channels;
  U u_min_val = std::numeric_limits<U>::min();
  U u_max_val = std::numeric_limits<U>::max();
  float min_val = static_cast<float>(u_min_val);
  float max_val = static_cast<float>(u_max_val);
  U uEx =
      cast_to<U>(extrapolated_value, min_val, max_val, u_min_val, u_max_val);
  // low y extrapolation zone
  if (min_iy > 0) {
    U* p = flip_y ? output + out_row_size * (out_height - min_iy) : output;
    int64 nn = out_row_size * (int64)min_iy;
    for (int64 i = 0; i < nn; ++i) p[i] = uEx;
  }
  // high y extrapolation zone
  if (max_iy < out_height - 1) {
    U* p = flip_y ? output : output + out_row_size * (max_iy + 1);
    int64 nn = out_row_size * (int64)(out_height - 1 - max_iy);
    for (int64 i = 0; i < nn; ++i) p[i] = uEx;
  }
  // low x extrapolation zone
  if (min_ix > 0) {
    for (int iy = min_iy; iy <= max_iy; ++iy) {
      int xx0 = flip_x ? (out_width - min_ix) * channels : 0;
      int nxx = min_ix * channels;
      U* p = output + xx0 +
             out_row_size * (int64)(flip_y ? out_height - 1 - iy : iy);
      for (int ix = 0; ix < nxx; ++ix) {
        p[ix] = uEx;
      }
    }
  }
  // high x extrapolation zone
  if (max_ix < out_width - 1) {
    for (int iy = min_iy; iy <= max_iy; ++iy) {
      int xx0 = flip_x ? 0 : (max_ix + 1) * channels;
      int nxx = (out_width - 1 - max_ix) * channels;
      U* p = output + xx0 +
             out_row_size * (int64)(flip_y ? out_height - 1 - iy : iy);
      for (int ix = 0; ix < nxx; ++ix) {
        p[ix] = uEx;
      }
    }
  }
  U* output_y_ptr =
      output +
      out_row_size * (int64)(flip_y ? out_height - 1 - min_iy : min_iy);
  // interpolation zone
  if (channels == 1) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
        const int64 xs_lower = xs[ix].lower;
        const int64 xs_upper = xs[ix].upper;
        const float xs_lerp = xs[ix].lerp;

        // Read channel 0.
        const float top_left0(ys_input_lower_ptr[xs_lower]);
        const float top_right0(ys_input_lower_ptr[xs_upper]);
        const float bottom_left0(ys_input_upper_ptr[xs_lower]);
        const float bottom_right0(ys_input_upper_ptr[xs_upper]);

        // Compute output.
        float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                     bottom_right0, xs_lerp, ys_lerp);
        output_y_ptr[x] =
            cast_to<U>(result0, min_val, max_val, u_min_val, u_max_val);
      }
      output_y_ptr =
          flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 2) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
        const int64 xs_lower = xs[ix].lower;
        const int64 xs_upper = xs[ix].upper;
        const float xs_lerp = xs[ix].lerp;

        // Read channel 0.
        const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
        const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
        const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
        const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

        // Read channel 1.
        const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
        const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
        const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
        const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

        // Compute output.
        float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                     bottom_right0, xs_lerp, ys_lerp);
        float result1 = compute_lerp(top_left1, top_right1, bottom_left1,
                                     bottom_right1, xs_lerp, ys_lerp);
        output_y_ptr[x * 2 + 0] =
            cast_to<U>(result0, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 2 + 1] =
            cast_to<U>(result1, min_val, max_val, u_min_val, u_max_val);
      }
      output_y_ptr =
          flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 3) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
        const int64 xs_lower = xs[ix].lower;
        const int64 xs_upper = xs[ix].upper;
        const float xs_lerp = xs[ix].lerp;

        // Read channel 0.
        const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
        const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
        const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
        const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

        // Read channel 1.
        const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
        const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
        const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
        const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

        // Read channel 2.
        const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
        const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
        const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
        const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

        // Compute output.
        float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                     bottom_right0, xs_lerp, ys_lerp);
        float result1 = compute_lerp(top_left1, top_right1, bottom_left1,
                                     bottom_right1, xs_lerp, ys_lerp);
        float result2 = compute_lerp(top_left2, top_right2, bottom_left2,
                                     bottom_right2, xs_lerp, ys_lerp);
        output_y_ptr[x * 3 + 0] =
            cast_to<U>(result0, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 3 + 1] =
            cast_to<U>(result1, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 3 + 2] =
            cast_to<U>(result2, min_val, max_val, u_min_val, u_max_val);
      }
      output_y_ptr =
          flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else if (channels == 4) {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
        const int64 xs_lower = xs[ix].lower;
        const int64 xs_upper = xs[ix].upper;
        const float xs_lerp = xs[ix].lerp;

        // Read channel 0.
        const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
        const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
        const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
        const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

        // Read channel 1.
        const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
        const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
        const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
        const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

        // Read channel 2.
        const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
        const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
        const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
        const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

        // Read channel 3.
        const float top_left3(ys_input_lower_ptr[xs_lower + 3]);
        const float top_right3(ys_input_lower_ptr[xs_upper + 3]);
        const float bottom_left3(ys_input_upper_ptr[xs_lower + 3]);
        const float bottom_right3(ys_input_upper_ptr[xs_upper + 3]);

        // Compute output.
        float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                     bottom_right0, xs_lerp, ys_lerp);
        float result1 = compute_lerp(top_left1, top_right1, bottom_left1,
                                     bottom_right1, xs_lerp, ys_lerp);
        float result2 = compute_lerp(top_left2, top_right2, bottom_left2,
                                     bottom_right2, xs_lerp, ys_lerp);
        float result3 = compute_lerp(top_left3, top_right3, bottom_left3,
                                     bottom_right3, xs_lerp, ys_lerp);
        output_y_ptr[x * 4 + 0] =
            cast_to<U>(result0, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 4 + 1] =
            cast_to<U>(result1, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 4 + 2] =
            cast_to<U>(result2, min_val, max_val, u_min_val, u_max_val);
        output_y_ptr[x * 4 + 3] =
            cast_to<U>(result3, min_val, max_val, u_min_val, u_max_val);
      }
      output_y_ptr =
          flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  } else {
    for (int y = min_iy; y <= max_iy; ++y) {
      const int iy = y - min_iy;
      const T* ys_input_lower_ptr = image + ys[iy].lower * in_row_size;
      const T* ys_input_upper_ptr = image + ys[iy].upper * in_row_size;
      const float ys_lerp = ys[iy].lerp;
      const int x0 = flip_x ? out_width - 1 - max_ix : min_ix;
      const int x1 = flip_x ? out_width - 1 - min_ix : max_ix;
      for (int x = x0; x <= x1; ++x) {
        const int ix = flip_x ? out_width - 1 - min_ix - x : x - min_ix;
        const int64 xs_lower = xs[ix].lower;
        const int64 xs_upper = xs[ix].upper;
        const float xs_lerp = xs[ix].lerp;
        for (int ichan = 0; ichan < channels; ++ichan) {
          const float top_left0(ys_input_lower_ptr[xs_lower + ichan]);
          const float top_right0(ys_input_lower_ptr[xs_upper + ichan]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + ichan]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + ichan]);
          float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                       bottom_right0, xs_lerp, ys_lerp);
          output_y_ptr[x * channels + ichan] =
              cast_to<U>(result0, min_val, max_val, u_min_val, u_max_val);
        }
      }
      output_y_ptr =
          flip_y ? output_y_ptr - out_row_size : output_y_ptr + out_row_size;
    }
  }
}

// template for method that calls either explicitly vectorized method
// or the fallback method, depending on what is appropriate for the
// machine you are running on
template <typename T, typename U>
void crop_resize_single_image_common(
    const T* image, const int64 in_height, const int64 in_width,
    const int64 out_height, const int64 out_width, const int channels,
    const int min_ix, const int max_ix, const CachedInterpolation* xs,
    const int min_iy, const int max_iy, const CachedInterpolation* ys,
    const float extrapolated_value, const bool flip_x, const bool flip_y,
    U* output) TF_ATTRIBUTE_NOINLINE;

// For now, only compile vectorized code on LINUX systems.
// to-do: Test vectorized code on other platforms (MacOS and Windows).
#if defined(__linux__) && defined(__SSE4_1__)

//
// The remaining code implements explicitly vectorized versions of a bilinear
// image resizer.
// Images with 1, 2, 3 or 4 channels are supported.
// The image resizer reads samples of type T and writes samples of type U.
// T and U can be any of the following: uint8, int8, uint16, int16, int32,
// Eigen::half, bfloat16 and float.
// There are separate codes for SSE4.1 and AVX2. Enabling AVX2 also enables
// FP16C instruction set,
// which contains instructions that convert between Eigen::half and float. The
// SSE4.1 code path emulates
// the FP16C instructions in software.
//

//
// This class loads 4 pixels with n channels, converts to fp32 and packs
// the result into n SSE vector words.
// Input data type T must be one of uint8, int8, uint16, int16, int32,
// Eigen::half, bfloat16 or float.
//

template <class T>
class VectorLoader {
 public:
#ifdef __AVX2__
  // convert 8 packed words of type T to fp32.
  // T must be one of uint8, int8, uint16, int16, int32, Eigen::half, bfloat16
  // or float.
  __m256 to_fp32(__m256i raw);
#else
  // convert 4 packed words of type T to fp32.
  // T must be one of uint8, int8, uint16, int16, int32, Eigen::half, bfloat16
  // or float.
  __m128 to_fp32(__m128i raw);
#endif

#ifdef __AVX2__
  // pack 4 pixels with 1 channel, 2 channels and 3channels respectively in
  // separate 128 bit lanes.
  // input is stored in lower portion of 4 separate sse words, v0 through v3.
  // output is stored in lower portion of v0.
  void pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  // output is stored in lower portion of v0 and v1.
  void pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  // output is stored in lower portion of v0, v1 and v2.
  void pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
#else
  // pack 4 pixels with 1 channel, 2 channels and 3channels respectively.
  // input is stored in lower portion of 4 separate sse words, v0 through v3.
  // output is stored in lower portion of v0.
  void pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  // output is stored in lower portion of v0 and v1.
  void pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  // output is stored in lower portion of v0, v1 and v2.
  void pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
#endif

#ifdef __AVX2__
  // extract right pixel for load1 and load4 cases.
  __m256i extract_right_1ch(const __m256i left);
  __m256i extract_right_2ch(const __m256i left);
  __m256i extract_right_3ch(const __m256i left);
  __m256i extract_right_4ch(const __m256i left);
#else
  __m128i extract_right_1ch(const __m128i left);
  __m128i extract_right_2ch(const __m128i left);
  __m128i extract_right_3ch(const __m128i left);
  __m128i extract_right_4ch(const __m128i left);
#endif

#ifdef __AVX2__
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 1 channel.
  // load1 case, i.e. 4 left and right inputs are loaded with a single unaligned
  // SSE load.
  void load1_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* right0);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 2 channels.
  // load1 case, i.e. 4 left and right inputs are loaded with a single unaligned
  // SSE load.
  void load1_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* right0, __m256* right1);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 3 channels.
  // load1 case, i.e. 4 left and right inputs are loaded with a single unaligned
  // SSE load.
  void load1_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* left2, __m256* right0, __m256* right1, __m256* right2);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 4 channels.
  // load1 case, i.e. 4 left and right inputs are loaded with a single unaligned
  // SSE load.
  void load1_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* left2, __m256* left3, __m256* right0, __m256* right1,
                 __m256* right2, __m256* right3);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 1 channel.
  // load2 case, i.e. 4 left inputs are loaded with first SSE load and 4 right
  // inputs are loaded with second SSE load.
  void load2_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* right0);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 2 channels.
  // load2 case, i.e. 4 left inputs are loaded with first SSE load and 4 right
  // inputs are loaded with second SSE load.
  void load2_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* right0, __m256* right1);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 3 channels.
  // load2 case, i.e. 4 left inputs are loaded with first SSE load and 4 right
  // inputs are loaded with second SSE load.
  void load2_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* left2, __m256* right0, __m256* right1, __m256* right2);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 4 channels.
  // load2 case, i.e. 4 left inputs are loaded with first SSE load and 4 right
  // inputs are loaded with second SSE load.
  void load2_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m256* left0, __m256* left1,
                 __m256* left2, __m256* left3, __m256* right0, __m256* right1,
                 __m256* right2, __m256* right3);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 1 channel.
  // load4 case, i.e. each pair of left and right inputs are loaded with a
  // separate SSE load.
  void load4_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* right0);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 2 channels.
  // load4 case, i.e. each pair of left and right inputs are loaded with a
  // separate SSE load.
  void load4_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* right0, __m256* right1);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 3 channels.
  // load4 case, i.e. each pair of left and right inputs are loaded with a
  // separate SSE load.
  void load4_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* left2, __m256* right0, __m256* right1,
                 __m256* right2);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 4 channels.
  // load4 case, i.e. each pair of left and right inputs are loaded with a
  // separate SSE load.
  void load4_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* left2, __m256* left3, __m256* right0,
                 __m256* right1, __m256* right2, __m256* right3);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 1 channel.
  // load8 case, i.e. each input is loaded with a separate SSE load.
  // 4 pixels, each with left and right input necessitates 8 separate SSE loads
  // per input row.
  void load8_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* right0);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 2 channels.
  // load8 case, i.e. each input is loaded with a separate SSE load.
  // 4 pixels, each with left and right input necessitates 8 separate SSE loads
  // per input row.
  void load8_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* right0, __m256* right1);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 3 channels.
  // load8 case, i.e. each input is loaded with a separate SSE load.
  // 4 pixels, each with left and right input necessitates 8 separate SSE loads
  // per input row.
  void load8_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* left2, __m256* right0, __m256* right1,
                 __m256* right2);
  // load top left and bottom left interpolation inputs into output argument
  // left.
  // load top right and bottom right interpolation inputs into output argument
  // right.
  // pixels have 4 channels.
  // load8 case, i.e. each input is loaded with a separate SSE load.
  // 4 pixels, each with left and right input necessitates 8 separate SSE loads
  // per input row.
  void load8_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m256* left0,
                 __m256* left1, __m256* left2, __m256* left3, __m256* right0,
                 __m256* right1, __m256* right2, __m256* right3);
#else
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 1 channel.
  // load1 case, i.e. all inputs for one input row are loaded with a single SSE
  // load.
  void load1_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* bl0,
                 __m128* tr0, __m128* br0);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 2 channels.
  // load1 case, i.e. all inputs for one input row are loaded with a single SSE
  // load.
  void load1_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* bl0, __m128* bl1, __m128* tr0, __m128* tr1,
                 __m128* br0, __m128* br1);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 3 channels.
  // load1 case, i.e. all inputs for one input row are loaded with a single SSE
  // load.
  void load1_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* tl2, __m128* bl0, __m128* bl1, __m128* bl2,
                 __m128* tr0, __m128* tr1, __m128* tr2, __m128* br0,
                 __m128* br1, __m128* br2);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 4 channels.
  // load1 case, i.e. all inputs for one input row are loaded with a single SSE
  // load.
  void load1_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* tl2, __m128* tl3, __m128* bl0, __m128* bl1,
                 __m128* bl2, __m128* bl3, __m128* tr0, __m128* tr1,
                 __m128* tr2, __m128* tr3, __m128* br0, __m128* br1,
                 __m128* br2, __m128* br3);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 1 channel.
  // load2 case, i.e. left inputs are loaded with first SSE load, right inputs
  // are loaded with second SSE load.
  void load2_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* bl0,
                 __m128* tr0, __m128* br0);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 2 channels.
  // load2 case, i.e. left inputs are loaded with first SSE load, right inputs
  // are loaded with second SSE load.
  void load2_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* bl0, __m128* bl1, __m128* tr0, __m128* tr1,
                 __m128* br0, __m128* br1);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 3 channels.
  // load2 case, i.e. left inputs are loaded with first SSE load, right inputs
  // are loaded with second SSE load.
  void load2_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* tl2, __m128* bl0, __m128* bl1, __m128* bl2,
                 __m128* tr0, __m128* tr1, __m128* tr2, __m128* br0,
                 __m128* br1, __m128* br2);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 4 channels.
  // load2 case, i.e. left inputs are loaded with first SSE load, right inputs
  // are loaded with second SSE load.
  void load2_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 const __m128i* shuffle_masks, __m128* tl0, __m128* tl1,
                 __m128* tl2, __m128* tl3, __m128* bl0, __m128* bl1,
                 __m128* bl2, __m128* bl3, __m128* tr0, __m128* tr1,
                 __m128* tr2, __m128* tr3, __m128* br0, __m128* br1,
                 __m128* br2, __m128* br3);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 1 channel.
  // load4 case, i.e. left and right inputs are loaded with a separate SSE load
  // for each pixel.
  void load4_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* bl0, __m128* tr0, __m128* br0);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 2 channels.
  // load4 case, i.e. left and right inputs are loaded with a separate SSE load
  // for each pixel.
  void load4_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* bl0, __m128* bl1, __m128* tr0,
                 __m128* tr1, __m128* br0, __m128* br1);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 3 channels.
  // load4 case, i.e. left and right inputs are loaded with a separate SSE load
  // for each pixel.
  void load4_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* tl2, __m128* bl0, __m128* bl1,
                 __m128* bl2, __m128* tr0, __m128* tr1, __m128* tr2,
                 __m128* br0, __m128* br1, __m128* br2);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 4 channels.
  // load4 case, i.e. left and right inputs are loaded with a separate SSE load
  // for each pixel.
  void load4_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* tl2, __m128* tl3, __m128* bl0,
                 __m128* bl1, __m128* bl2, __m128* bl3, __m128* tr0,
                 __m128* tr1, __m128* tr2, __m128* tr3, __m128* br0,
                 __m128* br1, __m128* br2, __m128* br3);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 1 channel.
  // load8 case, i.e. left and right inputs are loaded with separate SSE loads
  // for each pixel.
  void load8_1ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* bl0, __m128* tr0, __m128* br0);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 2 channels.
  // load8 case, i.e. left and right inputs are loaded with separate SSE loads
  // for each pixel.
  void load8_2ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* bl0, __m128* bl1, __m128* tr0,
                 __m128* tr1, __m128* br0, __m128* br1);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 3 channels.
  // load8 case, i.e. left and right inputs are loaded with separate SSE loads
  // for each pixel.
  void load8_3ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* tl2, __m128* bl0, __m128* bl1,
                 __m128* bl2, __m128* tr0, __m128* tr1, __m128* tr2,
                 __m128* br0, __m128* br1, __m128* br2);
  // load top left interpolation inputs into output argument tl.
  // load bottom left interpolation inputs into output argument bl.
  // load top right interpolation inputs into output argument tr.
  // load bottom right interpolation inputs into output argument br.
  // pixels have 4 channels.
  // load8 case, i.e. left and right inputs are loaded with separate SSE loads
  // for each pixel.
  void load8_4ch(const T* lower_ptr, const T* upper_ptr, int offset0,
                 int offset1, int offset2, int offset3, __m128* tl0,
                 __m128* tl1, __m128* tl2, __m128* tl3, __m128* bl0,
                 __m128* bl1, __m128* bl2, __m128* bl3, __m128* tr0,
                 __m128* tr1, __m128* tr2, __m128* tr3, __m128* br0,
                 __m128* br1, __m128* br2, __m128* br3);
#endif

  // there is no method that packs 4 pixels with 4 channel into four sse words.
  // nothing to do for this case, everything is already in the right position.

 private:
// helper methods
#ifdef __AVX2__
  // pack 4 pixels with 1, 2, 3 or 4 channels into lower portion of SSE vector
  // word.
  // works within SSE lanes.
  // sizeof(sample_data_type) can be 1, 2 or 4 bytes.
  void pack4_1b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_2b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_4b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_1b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_2b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_4b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_1b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_2b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
  void pack4_4b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2, __m256i* v3);
// there is no pack4_xx_4ch functions because none is needed.
// all the bytes are loaded in the right spots for this case.
#else
  // pack 4 pixels with 1, 2, 3 or 4 channels into lower portion of SSE vector
  // word.
  // sizeof(sample_data_type) can be 1, 2 or 4 bytes.
  void pack4_1b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_2b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_4b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_1b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_2b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_4b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_1b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_2b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
  void pack4_4b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2, __m128i* v3);
#endif
#ifdef __AVX2__
  __m256i extract_right_1b_(const __m256i left);
  __m256i extract_right_2b_(const __m256i left);
  __m256i extract_right_3b_(const __m256i left);
  __m256i extract_right_4b_(const __m256i left);
  __m256i extract_right_6b_(const __m256i left);
  __m256i extract_right_8b_(const __m256i left);
#else
  __m128i extract_right_1b_(const __m128i left);
  __m128i extract_right_2b_(const __m128i left);
  __m128i extract_right_3b_(const __m128i left);
  __m128i extract_right_4b_(const __m128i left);
  __m128i extract_right_6b_(const __m128i left);
  __m128i extract_right_8b_(const __m128i left);
#endif
};

#ifdef __AVX2__
template <class T>
void VectorLoader<T>::pack4_1b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  *v3 = _mm256_slli_si256(*v3, 3);
  __m256i and_mask = _mm256_setr_epi32(255, 0, 0, 0, 255, 0, 0, 0);
  *v2 = _mm256_or_si256(*v3,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v2), 2));
  *v1 = _mm256_or_si256(*v2,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v1), 1));
  *v0 = _mm256_or_si256(*v1, _mm256_and_si256(and_mask, *v0));
}
template <class T>
void VectorLoader<T>::pack4_2b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  *v3 = _mm256_slli_si256(*v3, 6);
  __m256i and_mask = _mm256_setr_epi32(65535, 0, 0, 0, 65535, 0, 0, 0);
  *v2 = _mm256_or_si256(*v3,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v2), 4));
  *v1 = _mm256_or_si256(*v2,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v1), 2));
  *v0 = _mm256_or_si256(*v1, _mm256_and_si256(and_mask, *v0));
}
template <class T>
void VectorLoader<T>::pack4_4b_1ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  *v3 = _mm256_slli_si256(*v3, 12);
  __m256i and_mask = _mm256_setr_epi32(-1, 0, 0, 0, -1, 0, 0, 0);
  *v2 = _mm256_or_si256(*v3,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v2), 8));
  *v1 = _mm256_or_si256(*v2,
                        _mm256_slli_si256(_mm256_and_si256(and_mask, *v1), 4));
  *v0 = _mm256_or_si256(*v1, _mm256_and_si256(and_mask, *v0));
}

template <class T>
void VectorLoader<T>::pack4_1b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(65535, 0, 0, 0, 65535, 0, 0, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 2));
  *v1 = _mm256_or_si256(_mm256_and_si256(*v2, and_mask),
                        _mm256_slli_si256(*v3, 2));
}
template <class T>
void VectorLoader<T>::pack4_2b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(-1, 0, 0, 0, -1, 0, 0, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 4));
  *v1 = _mm256_or_si256(_mm256_and_si256(*v2, and_mask),
                        _mm256_slli_si256(*v3, 4));
}
template <class T>
void VectorLoader<T>::pack4_4b_2ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(-1, -1, 0, 0, -1, -1, 0, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 8));
  *v1 = _mm256_or_si256(_mm256_and_si256(*v2, and_mask),
                        _mm256_slli_si256(*v3, 8));
}

template <class T>
void VectorLoader<T>::pack4_1b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(16777215, 0, 0, 0, 16777215, 0, 0, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 3));
  and_mask = _mm256_srli_si256(and_mask, 1);
  *v1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v1, 1), and_mask),
                        _mm256_slli_si256(*v2, 2));
  and_mask = _mm256_srli_si256(and_mask, 1);
  *v2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v2, 2), and_mask),
                        _mm256_slli_si256(*v3, 1));
}
template <class T>
void VectorLoader<T>::pack4_2b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(-1, 65535, 0, 0, -1, 65535, 0, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 6));
  and_mask = _mm256_srli_si256(and_mask, 2);
  *v1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v1, 2), and_mask),
                        _mm256_slli_si256(*v2, 4));
  and_mask = _mm256_srli_si256(and_mask, 2);
  *v2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v2, 4), and_mask),
                        _mm256_slli_si256(*v3, 2));
}
template <class T>
void VectorLoader<T>::pack4_4b_3ch_(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  __m256i and_mask = _mm256_setr_epi32(-1, -1, -1, 0, -1, -1, -1, 0);
  *v0 = _mm256_or_si256(_mm256_and_si256(*v0, and_mask),
                        _mm256_slli_si256(*v1, 12));
  and_mask = _mm256_srli_si256(and_mask, 4);
  *v1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v1, 4), and_mask),
                        _mm256_slli_si256(*v2, 8));
  and_mask = _mm256_srli_si256(and_mask, 4);
  *v2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(*v2, 8), and_mask),
                        _mm256_slli_si256(*v3, 4));
}

template <>
void VectorLoader<uint8>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_1b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                  __m256i* v3) {
  pack4_1b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                         __m256i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                      __m256i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_1ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_1ch_(v0, v1, v2, v3);
}

template <>
void VectorLoader<uint8>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_1b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                  __m256i* v3) {
  pack4_1b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                         __m256i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                      __m256i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_2ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_2ch_(v0, v1, v2, v3);
}

template <>
void VectorLoader<uint8>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_1b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                  __m256i* v3) {
  pack4_1b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                    __m256i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                         __m256i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                      __m256i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_3ch(__m256i* v0, __m256i* v1, __m256i* v2,
                                   __m256i* v3) {
  pack4_4b_3ch_(v0, v1, v2, v3);
}
#else
template <class T>
void VectorLoader<T>::pack4_1b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  *v3 = _mm_slli_si128(*v3, 3);
  __m128i and_mask = _mm_setr_epi32(255, 0, 0, 0);
  *v2 = _mm_or_si128(*v3, _mm_slli_si128(_mm_and_si128(and_mask, *v2), 2));
  *v1 = _mm_or_si128(*v2, _mm_slli_si128(_mm_and_si128(and_mask, *v1), 1));
  *v0 = _mm_or_si128(*v1, _mm_and_si128(and_mask, *v0));
}
template <class T>
void VectorLoader<T>::pack4_2b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  *v3 = _mm_slli_si128(*v3, 6);
  __m128i and_mask = _mm_setr_epi32(65535, 0, 0, 0);
  *v2 = _mm_or_si128(*v3, _mm_slli_si128(_mm_and_si128(and_mask, *v2), 4));
  *v1 = _mm_or_si128(*v2, _mm_slli_si128(_mm_and_si128(and_mask, *v1), 2));
  *v0 = _mm_or_si128(*v1, _mm_and_si128(and_mask, *v0));
}
template <class T>
void VectorLoader<T>::pack4_4b_1ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  *v3 = _mm_slli_si128(*v3, 12);
  __m128i and_mask = _mm_setr_epi32(-1, 0, 0, 0);
  *v2 = _mm_or_si128(*v3, _mm_slli_si128(_mm_and_si128(and_mask, *v2), 8));
  *v1 = _mm_or_si128(*v2, _mm_slli_si128(_mm_and_si128(and_mask, *v1), 4));
  *v0 = _mm_or_si128(*v1, _mm_and_si128(and_mask, *v0));
}
template <class T>
void VectorLoader<T>::pack4_1b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(65535, 0, 0, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 2));
  *v1 = _mm_or_si128(_mm_and_si128(*v2, and_mask), _mm_slli_si128(*v3, 2));
}
template <class T>
void VectorLoader<T>::pack4_2b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(-1, 0, 0, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 4));
  *v1 = _mm_or_si128(_mm_and_si128(*v2, and_mask), _mm_slli_si128(*v3, 4));
}
template <class T>
void VectorLoader<T>::pack4_4b_2ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(-1, -1, 0, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 8));
  *v1 = _mm_or_si128(_mm_and_si128(*v2, and_mask), _mm_slli_si128(*v3, 8));
}
template <class T>
void VectorLoader<T>::pack4_1b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(16777215, 0, 0, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 3));
  and_mask = _mm_srli_si128(and_mask, 1);
  *v1 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v1, 1), and_mask),
                     _mm_slli_si128(*v2, 2));
  and_mask = _mm_srli_si128(and_mask, 1);
  *v2 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v2, 2), and_mask),
                     _mm_slli_si128(*v3, 1));
}
template <class T>
void VectorLoader<T>::pack4_2b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(-1, 65535, 0, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 6));
  and_mask = _mm_srli_si128(and_mask, 2);
  *v1 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v1, 2), and_mask),
                     _mm_slli_si128(*v2, 4));
  and_mask = _mm_srli_si128(and_mask, 2);
  *v2 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v2, 4), and_mask),
                     _mm_slli_si128(*v3, 2));
}
template <class T>
void VectorLoader<T>::pack4_4b_3ch_(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  __m128i and_mask = _mm_setr_epi32(-1, -1, -1, 0);
  *v0 = _mm_or_si128(_mm_and_si128(*v0, and_mask), _mm_slli_si128(*v1, 12));
  and_mask = _mm_srli_si128(and_mask, 4);
  *v1 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v1, 4), and_mask),
                     _mm_slli_si128(*v2, 8));
  and_mask = _mm_srli_si128(and_mask, 4);
  *v2 = _mm_or_si128(_mm_and_si128(_mm_srli_si128(*v2, 8), and_mask),
                     _mm_slli_si128(*v3, 4));
}

template <>
void VectorLoader<uint8>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_1b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                  __m128i* v3) {
  pack4_1b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                         __m128i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                      __m128i* v3) {
  pack4_2b_1ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_1ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_1ch_(v0, v1, v2, v3);
}

template <>
void VectorLoader<uint8>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_1b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                  __m128i* v3) {
  pack4_1b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                         __m128i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                      __m128i* v3) {
  pack4_2b_2ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_2ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_2ch_(v0, v1, v2, v3);
}

template <>
void VectorLoader<uint8>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_1b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int8>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                  __m128i* v3) {
  pack4_1b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<uint16>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                    __m128i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int16>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<int32>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<Eigen::half>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                         __m128i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<bfloat16>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                      __m128i* v3) {
  pack4_2b_3ch_(v0, v1, v2, v3);
}
template <>
void VectorLoader<float>::pack_3ch(__m128i* v0, __m128i* v1, __m128i* v2,
                                   __m128i* v3) {
  pack4_4b_3ch_(v0, v1, v2, v3);
}
#endif

#ifdef __AVX2__
template <>
__m256i VectorLoader<uint8>::extract_right_1ch(const __m256i left) {
  return extract_right_1b_(left);
}
template <>
__m256i VectorLoader<int8>::extract_right_1ch(const __m256i left) {
  return extract_right_1b_(left);
}
template <>
__m256i VectorLoader<uint16>::extract_right_1ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<int16>::extract_right_1ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<int32>::extract_right_1ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<Eigen::half>::extract_right_1ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<bfloat16>::extract_right_1ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<float>::extract_right_1ch(const __m256i left) {
  return extract_right_4b_(left);
}

template <>
__m256i VectorLoader<uint8>::extract_right_2ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<int8>::extract_right_2ch(const __m256i left) {
  return extract_right_2b_(left);
}
template <>
__m256i VectorLoader<uint16>::extract_right_2ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<int16>::extract_right_2ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<int32>::extract_right_2ch(const __m256i left) {
  return extract_right_8b_(left);
}
template <>
__m256i VectorLoader<Eigen::half>::extract_right_2ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<bfloat16>::extract_right_2ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<float>::extract_right_2ch(const __m256i left) {
  return extract_right_8b_(left);
}

template <>
__m256i VectorLoader<uint8>::extract_right_3ch(const __m256i left) {
  return extract_right_3b_(left);
}
template <>
__m256i VectorLoader<int8>::extract_right_3ch(const __m256i left) {
  return extract_right_3b_(left);
}
template <>
__m256i VectorLoader<uint16>::extract_right_3ch(const __m256i left) {
  return extract_right_6b_(left);
}
template <>
__m256i VectorLoader<int16>::extract_right_3ch(const __m256i left) {
  return extract_right_6b_(left);
}
template <>
__m256i VectorLoader<int32>::extract_right_3ch(const __m256i left) {
  assert(false);
}
template <>
__m256i VectorLoader<Eigen::half>::extract_right_3ch(const __m256i left) {
  return extract_right_6b_(left);
}
template <>
__m256i VectorLoader<bfloat16>::extract_right_3ch(const __m256i left) {
  return extract_right_6b_(left);
}
template <>
__m256i VectorLoader<float>::extract_right_3ch(const __m256i left) {
  assert(false);
}

template <>
__m256i VectorLoader<uint8>::extract_right_4ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<int8>::extract_right_4ch(const __m256i left) {
  return extract_right_4b_(left);
}
template <>
__m256i VectorLoader<uint16>::extract_right_4ch(const __m256i left) {
  return extract_right_8b_(left);
}
template <>
__m256i VectorLoader<int16>::extract_right_4ch(const __m256i left) {
  return extract_right_8b_(left);
}
template <>
__m256i VectorLoader<int32>::extract_right_4ch(const __m256i left) {
  assert(false);
}
template <>
__m256i VectorLoader<Eigen::half>::extract_right_4ch(const __m256i left) {
  return extract_right_8b_(left);
}
template <>
__m256i VectorLoader<bfloat16>::extract_right_4ch(const __m256i left) {
  return extract_right_8b_(left);
}
template <>
__m256i VectorLoader<float>::extract_right_4ch(const __m256i left) {
  assert(false);
}
#else
template <>
__m128i VectorLoader<uint8>::extract_right_1ch(const __m128i left) {
  return extract_right_1b_(left);
}
template <>
__m128i VectorLoader<int8>::extract_right_1ch(const __m128i left) {
  return extract_right_1b_(left);
}
template <>
__m128i VectorLoader<uint16>::extract_right_1ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<int16>::extract_right_1ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<int32>::extract_right_1ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<Eigen::half>::extract_right_1ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<bfloat16>::extract_right_1ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<float>::extract_right_1ch(const __m128i left) {
  return extract_right_4b_(left);
}

template <>
__m128i VectorLoader<uint8>::extract_right_2ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<int8>::extract_right_2ch(const __m128i left) {
  return extract_right_2b_(left);
}
template <>
__m128i VectorLoader<uint16>::extract_right_2ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<int16>::extract_right_2ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<int32>::extract_right_2ch(const __m128i left) {
  return extract_right_8b_(left);
}
template <>
__m128i VectorLoader<Eigen::half>::extract_right_2ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<bfloat16>::extract_right_2ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<float>::extract_right_2ch(const __m128i left) {
  return extract_right_8b_(left);
}

template <>
__m128i VectorLoader<uint8>::extract_right_3ch(const __m128i left) {
  return extract_right_3b_(left);
}
template <>
__m128i VectorLoader<int8>::extract_right_3ch(const __m128i left) {
  return extract_right_3b_(left);
}
template <>
__m128i VectorLoader<uint16>::extract_right_3ch(const __m128i left) {
  return extract_right_6b_(left);
}
template <>
__m128i VectorLoader<int16>::extract_right_3ch(const __m128i left) {
  return extract_right_6b_(left);
}
template <>
__m128i VectorLoader<int32>::extract_right_3ch(const __m128i left) {
  assert(false);
}
template <>
__m128i VectorLoader<Eigen::half>::extract_right_3ch(const __m128i left) {
  return extract_right_6b_(left);
}
template <>
__m128i VectorLoader<bfloat16>::extract_right_3ch(const __m128i left) {
  return extract_right_6b_(left);
}
template <>
__m128i VectorLoader<float>::extract_right_3ch(const __m128i left) {
  assert(false);
}

template <>
__m128i VectorLoader<uint8>::extract_right_4ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<int8>::extract_right_4ch(const __m128i left) {
  return extract_right_4b_(left);
}
template <>
__m128i VectorLoader<uint16>::extract_right_4ch(const __m128i left) {
  return extract_right_8b_(left);
}
template <>
__m128i VectorLoader<int16>::extract_right_4ch(const __m128i left) {
  return extract_right_8b_(left);
}
template <>
__m128i VectorLoader<int32>::extract_right_4ch(const __m128i left) {
  assert(false);
}
template <>
__m128i VectorLoader<Eigen::half>::extract_right_4ch(const __m128i left) {
  return extract_right_8b_(left);
}
template <>
__m128i VectorLoader<bfloat16>::extract_right_4ch(const __m128i left) {
  return extract_right_8b_(left);
}
template <>
__m128i VectorLoader<float>::extract_right_4ch(const __m128i left) {
  assert(false);
}
#endif

#ifdef __AVX2__
template <>
__m256 VectorLoader<uint8>::to_fp32(__m256i raw) {
  raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_cvtepu8_epi32(_mm256_castsi256_si128(raw))),
      _mm_cvtepu8_epi32(_mm256_extractf128_si256(raw, 1)), 1);
  return _mm256_cvtepi32_ps(raw);
}
template <>
__m256 VectorLoader<int8>::to_fp32(__m256i raw) {
  raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_cvtepi8_epi32(_mm256_castsi256_si128(raw))),
      _mm_cvtepi8_epi32(_mm256_extractf128_si256(raw, 1)), 1);
  return _mm256_cvtepi32_ps(raw);
}
template <>
__m256 VectorLoader<uint16>::to_fp32(__m256i raw) {
  raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_cvtepu16_epi32(_mm256_castsi256_si128(raw))),
      _mm_cvtepu16_epi32(_mm256_extractf128_si256(raw, 1)), 1);
  return _mm256_cvtepi32_ps(raw);
}
template <>
__m256 VectorLoader<int16>::to_fp32(__m256i raw) {
  raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_cvtepi16_epi32(_mm256_castsi256_si128(raw))),
      _mm_cvtepi16_epi32(_mm256_extractf128_si256(raw, 1)), 1);
  return _mm256_cvtepi32_ps(raw);
}
template <>
__m256 VectorLoader<int32>::to_fp32(__m256i raw) {
  return _mm256_cvtepi32_ps(raw);
}
template <>
__m256 VectorLoader<Eigen::half>::to_fp32(__m256i raw) {
  return _mm256_insertf128_ps(
      _mm256_castps128_ps256(_mm_cvtph_ps(_mm256_castsi256_si128(raw))),
      _mm_cvtph_ps(_mm256_extractf128_si256(raw, 1)), 1);
}
template <>
__m256 VectorLoader<bfloat16>::to_fp32(__m256i raw) {
  // bfloat16 is essentially fp32 with mantissa truncated from 23 to 7 bits.
  // can convert with << 16, which we fuse with initial shuffle into epi32
  // positions.
  __m256i shuf_hi32 = _mm256_setr_epi8(
      -128, -128, 0, 1, -128, -128, 2, 3, -128, -128, 4, 5, -128, -128, 6, 7,
      -128, -128, 0, 1, -128, -128, 2, 3, -128, -128, 4, 5, -128, -128, 6, 7);
  return _mm256_castsi256_ps(_mm256_shuffle_epi8(raw, shuf_hi32));
}
template <>
__m256 VectorLoader<float>::to_fp32(__m256i raw) {
  return _mm256_castsi256_ps(raw);
}
#else
template <>
__m128 VectorLoader<uint8>::to_fp32(__m128i raw) {
  return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(raw));
}
template <>
__m128 VectorLoader<int8>::to_fp32(__m128i raw) {
  return _mm_cvtepi32_ps(_mm_cvtepi8_epi32(raw));
}
template <>
__m128 VectorLoader<uint16>::to_fp32(__m128i raw) {
  return _mm_cvtepi32_ps(_mm_cvtepu16_epi32(raw));
}
template <>
__m128 VectorLoader<int16>::to_fp32(__m128i raw) {
  return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(raw));
}
template <>
__m128 VectorLoader<int32>::to_fp32(__m128i raw) {
  return _mm_cvtepi32_ps(raw);
}
template <>
__m128 VectorLoader<Eigen::half>::to_fp32(__m128i raw) {
#ifdef __F16C__
  return _mm_cvtph_ps(raw);
#else
  // It is fairly trivial to convert from fp16 to fp32.
  // The formats are defined as follows:
  //
  // fp16 :: 15=sign_bit, 14-10=exponent, 9-0=mantissa :: exp zero offset is 15
  //      :: exponent of -15 (all 0) and +16 (all 1) are special numbers.
  // fp32 :: 31=sign_bit, 30-23=exponent, 22-0=mantissa :: exp zero offset is
  // 127
  //      :: exponent of -127 (all 0) and +128 (all 1) are special numbers.
  //
  // Assuming the fp16 values is stored in the lower 16 bits of an int32
  // 'fp16_val'.
  //
  // fp16_mantissa = fp16_val & (2^10-1)
  // fp32_mantissa = fp16_mantissa << 13
  //
  // The exponent is a little trickier.
  // For normal numbers, the following works:
  // fp16_exponent_with_10bit_left_shift = (fp16_val & ((2^5-1)<<10))
  // fp16_exponent_at_msb = fp16_exponent_with_10bit_left_shift << 17
  // The next line shifts in 1's from msb
  // fp16_exponent_at_fp32_position = fp16_exponent_at_msb >> 4
  // The next line flips the 3 bits from [msb-1,msb-4]
  // fp32_exponent = fp16_exponent_at_fp32_position ^ (7 << 27)
  // This breaks for subnormals, nan and infinity.
  // The only thing that breaks is the 3bit bit flip, which should
  // happen for normal numbers, but should not happen otherwise.
  // Since the bit flip can be done with an XOR of all 1's, we
  // can make this happen by turning the XOR mask to all zeros
  // when the fp16_exponent is either 0 or 31.
  //
  // ..move 16-bit input words to lower part of 32-bit positions.
  __m128i shuf_lo32 = _mm_setr_epi8(0, 1, -128, -128, 2, 3, -128, -128, 4, 5,
                                    -128, -128, 6, 7, -128, -128);
  __m128i fp16_val = _mm_shuffle_epi8(raw, shuf_lo32);
  // ..extract sign bit
  __m128i fp32_sign =
      _mm_slli_epi32(_mm_and_si128(fp16_val, _mm_set1_epi32(32768)), 16);
  // ..extract fp16_mantissa and shift
  __m128i fp16_mantissa = _mm_and_si128(fp16_val, _mm_set1_epi32(1023));
  __m128i fp32_mantissa = _mm_slli_epi32(fp16_mantissa, 13);
  // ..extract fp16 exponent shifted 10bits to the left
  __m128i fp16_exponent_sl10 = _mm_and_si128(fp16_val, _mm_set1_epi32(31744));
  __m128i fp16_exponent_all1_mask =
      _mm_cmpeq_epi32(fp16_exponent_sl10, _mm_set1_epi32(31 << 10));
  __m128i fp16_exponent_all0_mask =
      _mm_cmpeq_epi32(fp16_exponent_sl10, _mm_setzero_si128());
  __m128i fp16_denormal_mask =
      _mm_or_si128(fp16_exponent_all0_mask, fp16_exponent_all1_mask);
  __m128i fp32_exponent_before_xor =
      _mm_and_si128(_mm_set1_epi32(2139095040),
                    _mm_srai_epi32(_mm_slli_epi32(fp16_exponent_sl10, 17), 4));
  __m128i fp32_exponent_xor_mask =
      _mm_andnot_si128(fp16_denormal_mask, _mm_set1_epi32(7 << 27));
  __m128i fp32_exponent =
      _mm_xor_si128(fp32_exponent_xor_mask, fp32_exponent_before_xor);
  // ..or everything into one word
  __m128i fp32_val =
      _mm_or_si128(_mm_or_si128(fp32_sign, fp32_exponent), fp32_mantissa);
  return _mm_castsi128_ps(fp32_val);
#endif
}
template <>
__m128 VectorLoader<bfloat16>::to_fp32(__m128i raw) {
  // bfloat16 is essentially fp32 with mantissa truncated from 23 to 7 bits.
  // can convert with << 16, which we fuse with initial shuffle into epi32
  // positions.
  __m128i shuf_hi32 = _mm_setr_epi8(-128, -128, 0, 1, -128, -128, 2, 3, -128,
                                    -128, 4, 5, -128, -128, 6, 7);
  return _mm_castsi128_ps(_mm_shuffle_epi8(raw, shuf_hi32));
}
template <>
__m128 VectorLoader<float>::to_fp32(__m128i raw) {
  return _mm_castsi128_ps(raw);
}
#endif

#ifdef __AVX2__
template <class T>
__m256i VectorLoader<T>::extract_right_1b_(const __m256i left) {
  return _mm256_srli_si256(left, 1);
}
template <class T>
__m256i VectorLoader<T>::extract_right_2b_(const __m256i left) {
  return _mm256_srli_si256(left, 2);
}
template <class T>
__m256i VectorLoader<T>::extract_right_3b_(const __m256i left) {
  return _mm256_srli_si256(left, 3);
}
template <class T>
__m256i VectorLoader<T>::extract_right_4b_(const __m256i left) {
  return _mm256_srli_si256(left, 4);
}
template <class T>
__m256i VectorLoader<T>::extract_right_6b_(const __m256i left) {
  return _mm256_srli_si256(left, 6);
}
template <class T>
__m256i VectorLoader<T>::extract_right_8b_(const __m256i left) {
  return _mm256_srli_si256(left, 8);
}
#else
template <class T>
__m128i VectorLoader<T>::extract_right_1b_(const __m128i left) {
  return _mm_srli_si128(left, 1);
}
template <class T>
__m128i VectorLoader<T>::extract_right_2b_(const __m128i left) {
  return _mm_srli_si128(left, 2);
}
template <class T>
__m128i VectorLoader<T>::extract_right_3b_(const __m128i left) {
  return _mm_srli_si128(left, 3);
}
template <class T>
__m128i VectorLoader<T>::extract_right_4b_(const __m128i left) {
  return _mm_srli_si128(left, 4);
}
template <class T>
__m128i VectorLoader<T>::extract_right_6b_(const __m128i left) {
  return _mm_srli_si128(left, 6);
}
template <class T>
__m128i VectorLoader<T>::extract_right_8b_(const __m128i left) {
  return _mm_srli_si128(left, 8);
}
#endif

#ifdef __AVX2__
template <class T>
void VectorLoader<T>::load1_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* right0) {
  __m256i raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  *left0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[0])));
  *right0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[1])));
}
template <class T>
void VectorLoader<T>::load1_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* right0,
                                __m256* right1) {
  __m256i raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  *left0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[0])));
  *left1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[1])));
  *right0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[2])));
  *right1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[3])));
}
template <class T>
void VectorLoader<T>::load1_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* left2,
                                __m256* right0, __m256* right1,
                                __m256* right2) {
  __m256i raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  *left0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[0])));
  *left1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[1])));
  *left2 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[2])));
  *right0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[3])));
  *right1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[4])));
  *right2 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[5])));
}
template <class T>
void VectorLoader<T>::load1_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* left2,
                                __m256* left3, __m256* right0, __m256* right1,
                                __m256* right2, __m256* right3) {
  __m256i raw = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  *left0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[0])));
  *left1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[1])));
  *left2 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[2])));
  *left3 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[3])));
  *right0 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[4])));
  *right1 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[5])));
  *right2 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[6])));
  *right3 = to_fp32(
      _mm256_shuffle_epi8(raw, _mm256_broadcastsi128_si256(shuffle_masks[7])));
}
template <class T>
void VectorLoader<T>::load2_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* right0) {
  __m256i raw1 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i raw2 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 1))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 1)), 1);
  __m256i mask = _mm256_broadcastsi128_si256(shuffle_masks[0]);
  *left0 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right0 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
}
template <class T>
void VectorLoader<T>::load2_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* right0,
                                __m256* right1) {
  __m256i raw1 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i raw2 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 2))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 2)), 1);
  __m256i mask = _mm256_broadcastsi128_si256(shuffle_masks[0]);
  *left0 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right0 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[1]);
  *left1 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right1 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
}
template <class T>
void VectorLoader<T>::load2_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* left2,
                                __m256* right0, __m256* right1,
                                __m256* right2) {
  __m256i raw1 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i raw2 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 3))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 3)), 1);
  __m256i mask = _mm256_broadcastsi128_si256(shuffle_masks[0]);
  *left0 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right0 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[1]);
  *left1 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right1 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[2]);
  *left2 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right2 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
}
template <class T>
void VectorLoader<T>::load2_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m256* left0, __m256* left1, __m256* left2,
                                __m256* left3, __m256* right0, __m256* right1,
                                __m256* right2, __m256* right3) {
  __m256i raw1 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i raw2 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 4))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 4)), 1);
  __m256i mask = _mm256_broadcastsi128_si256(shuffle_masks[0]);
  *left0 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right0 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[1]);
  *left1 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right1 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[2]);
  *left2 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right2 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
  mask = _mm256_broadcastsi128_si256(shuffle_masks[3]);
  *left3 = to_fp32(_mm256_shuffle_epi8(raw1, mask));
  *right3 = to_fp32(_mm256_shuffle_epi8(raw2, mask));
}
template <class T>
void VectorLoader<T>::load4_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* right0) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = extract_right_1ch(l0);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = extract_right_1ch(l1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = extract_right_1ch(l2);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = extract_right_1ch(l3);
  }
  pack_1ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  pack_1ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
}
template <class T>
void VectorLoader<T>::load4_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* right0, __m256* right1) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = extract_right_2ch(l0);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = extract_right_2ch(l1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = extract_right_2ch(l2);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = extract_right_2ch(l3);
  }
  pack_2ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  pack_2ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
}
template <class T>
void VectorLoader<T>::load4_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* left2, __m256* right0, __m256* right1,
                                __m256* right2) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = extract_right_3ch(l0);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = extract_right_3ch(l1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = extract_right_3ch(l2);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = extract_right_3ch(l3);
  }
  pack_3ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  *left2 = to_fp32(l2);
  pack_3ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
  *right2 = to_fp32(r2);
}
template <class T>
void VectorLoader<T>::load4_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* left2, __m256* left3, __m256* right0,
                                __m256* right1, __m256* right2,
                                __m256* right3) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = extract_right_4ch(l0);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = extract_right_4ch(l1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = extract_right_4ch(l2);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = extract_right_4ch(l3);
  }
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  *left2 = to_fp32(l2);
  *left3 = to_fp32(l3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
  *right2 = to_fp32(r2);
  *right3 = to_fp32(r3);
}
template <class T>
void VectorLoader<T>::load8_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* right0) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 1))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 1)), 1);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 1)), 1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 1)), 1);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 1)), 1);
  }
  pack_1ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  pack_1ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
}
template <class T>
void VectorLoader<T>::load8_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* right0, __m256* right1) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 2))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 2)), 1);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 2)), 1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 2)), 1);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 2)), 1);
  }
  pack_2ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  pack_2ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
}
template <class T>
void VectorLoader<T>::load8_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* left2, __m256* right0, __m256* right1,
                                __m256* right2) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 3))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 3)), 1);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 3)), 1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 3)), 1);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 3)), 1);
  }
  pack_3ch(&l0, &l1, &l2, &l3);
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  *left2 = to_fp32(l2);
  pack_3ch(&r0, &r1, &r2, &r3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
  *right2 = to_fp32(r2);
}
template <class T>
void VectorLoader<T>::load8_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m256* left0, __m256* left1,
                                __m256* left2, __m256* left3, __m256* right0,
                                __m256* right1, __m256* right2,
                                __m256* right3) {
  __m256i l0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(_mm_loadu_si128((__m128i*)(lower_ptr + offset0))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0)), 1);
  __m256i r0 = _mm256_insertf128_si256(
      _mm256_castsi128_si256(
          _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 4))),
      _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 4)), 1);
  __m256i l1, r1;
  if (offset1 == offset0) {
    l1 = l0;
    r1 = r0;
  } else {
    l1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1)), 1);
    r1 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 4))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 4)), 1);
  }
  __m256i l2, r2;
  if (offset2 == offset1) {
    l2 = l1;
    r2 = r1;
  } else {
    l2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2)), 1);
    r2 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 4))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 4)), 1);
  }
  __m256i l3, r3;
  if (offset3 == offset2) {
    l3 = l2;
    r3 = r2;
  } else {
    l3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3)), 1);
    r3 = _mm256_insertf128_si256(
        _mm256_castsi128_si256(
            _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 4))),
        _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 4)), 1);
  }
  *left0 = to_fp32(l0);
  *left1 = to_fp32(l1);
  *left2 = to_fp32(l2);
  *left3 = to_fp32(l3);
  *right0 = to_fp32(r0);
  *right1 = to_fp32(r1);
  *right2 = to_fp32(r2);
  *right3 = to_fp32(r3);
}
#else
template <class T>
void VectorLoader<T>::load1_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* bl0, __m128* tr0,
                                __m128* br0) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
}
template <class T>
void VectorLoader<T>::load1_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* bl0,
                                __m128* bl1, __m128* tr0, __m128* tr1,
                                __m128* br0, __m128* br1) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
}
template <class T>
void VectorLoader<T>::load1_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* tl2,
                                __m128* bl0, __m128* bl1, __m128* bl2,
                                __m128* tr0, __m128* tr1, __m128* tr2,
                                __m128* br0, __m128* br1, __m128* br2) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[4]));
  *tr2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[5]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *bl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[4]));
  *br2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[5]));
}
template <class T>
void VectorLoader<T>::load1_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* tl2,
                                __m128* tl3, __m128* bl0, __m128* bl1,
                                __m128* bl2, __m128* bl3, __m128* tr0,
                                __m128* tr1, __m128* tr2, __m128* tr3,
                                __m128* br0, __m128* br1, __m128* br2,
                                __m128* br3) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *tl3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[4]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[5]));
  *tr2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[6]));
  *tr3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[7]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *bl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *bl3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[4]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[5]));
  *br2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[6]));
  *br3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[7]));
}
template <class T>
void VectorLoader<T>::load2_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* bl0, __m128* tr0,
                                __m128* br0) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 1));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 1));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
}
template <class T>
void VectorLoader<T>::load2_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* bl0,
                                __m128* bl1, __m128* tr0, __m128* tr1,
                                __m128* br0, __m128* br1) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 2));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 2));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
}
template <class T>
void VectorLoader<T>::load2_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* tl2,
                                __m128* bl0, __m128* bl1, __m128* bl2,
                                __m128* tr0, __m128* tr1, __m128* tr2,
                                __m128* br0, __m128* br1, __m128* br2) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 3));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tr2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *bl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 3));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *br2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
}
template <class T>
void VectorLoader<T>::load2_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, const __m128i* shuffle_masks,
                                __m128* tl0, __m128* tl1, __m128* tl2,
                                __m128* tl3, __m128* bl0, __m128* bl1,
                                __m128* bl2, __m128* bl3, __m128* tr0,
                                __m128* tr1, __m128* tr2, __m128* tr3,
                                __m128* br0, __m128* br1, __m128* br2,
                                __m128* br3) {
  __m128i raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  *tl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *tl3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  raw = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 4));
  *tr0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *tr1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *tr2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *tr3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  *bl0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *bl1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *bl2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *bl3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
  raw = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 4));
  *br0 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[0]));
  *br1 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[1]));
  *br2 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[2]));
  *br3 = to_fp32(_mm_shuffle_epi8(raw, shuffle_masks[3]));
}
template <class T>
void VectorLoader<T>::load4_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* bl0,
                                __m128* tr0, __m128* br0) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = extract_right_1ch(itl0);
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = extract_right_1ch(ibl0);
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = extract_right_1ch(itl1);
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = extract_right_1ch(ibl1);
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = extract_right_1ch(itl2);
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = extract_right_1ch(ibl2);
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = extract_right_1ch(itl3);
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = extract_right_1ch(ibl3);
  }
  pack_1ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  pack_1ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  pack_1ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  pack_1ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
}
template <class T>
void VectorLoader<T>::load4_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* bl0, __m128* bl1, __m128* tr0,
                                __m128* tr1, __m128* br0, __m128* br1) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = extract_right_2ch(itl0);
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = extract_right_2ch(ibl0);
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = extract_right_2ch(itl1);
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = extract_right_2ch(ibl1);
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = extract_right_2ch(itl2);
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = extract_right_2ch(ibl2);
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = extract_right_2ch(itl3);
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = extract_right_2ch(ibl3);
  }
  pack_2ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  pack_2ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  pack_2ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  pack_2ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
}
template <class T>
void VectorLoader<T>::load4_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* tl2, __m128* bl0, __m128* bl1,
                                __m128* bl2, __m128* tr0, __m128* tr1,
                                __m128* tr2, __m128* br0, __m128* br1,
                                __m128* br2) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = extract_right_3ch(itl0);
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = extract_right_3ch(ibl0);
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = extract_right_3ch(itl1);
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = extract_right_3ch(ibl1);
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = extract_right_3ch(itl2);
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = extract_right_3ch(ibl2);
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = extract_right_3ch(itl3);
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = extract_right_3ch(ibl3);
  }
  pack_3ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  *tl2 = to_fp32(itl2);
  pack_3ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  *tr2 = to_fp32(itr2);
  pack_3ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  *bl2 = to_fp32(ibl2);
  pack_3ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
  *br2 = to_fp32(ibr2);
}
template <class T>
void VectorLoader<T>::load4_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* tl2, __m128* tl3, __m128* bl0,
                                __m128* bl1, __m128* bl2, __m128* bl3,
                                __m128* tr0, __m128* tr1, __m128* tr2,
                                __m128* tr3, __m128* br0, __m128* br1,
                                __m128* br2, __m128* br3) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = extract_right_4ch(itl0);
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = extract_right_4ch(ibl0);
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = extract_right_4ch(itl1);
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = extract_right_4ch(ibl1);
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = extract_right_4ch(itl2);
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = extract_right_4ch(ibl2);
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = extract_right_4ch(itl3);
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = extract_right_4ch(ibl3);
  }
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  *tl2 = to_fp32(itl2);
  *tl3 = to_fp32(itl3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  *tr2 = to_fp32(itr2);
  *tr3 = to_fp32(itr3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  *bl2 = to_fp32(ibl2);
  *bl3 = to_fp32(ibl3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
  *br2 = to_fp32(ibr2);
  *br3 = to_fp32(ibr3);
}
template <class T>
void VectorLoader<T>::load8_1ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* bl0,
                                __m128* tr0, __m128* br0) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 1));
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 1));
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 1));
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 1));
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 1));
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 1));
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 1));
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 1));
  }
  pack_1ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  pack_1ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  pack_1ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  pack_1ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
}
template <class T>
void VectorLoader<T>::load8_2ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* bl0, __m128* bl1, __m128* tr0,
                                __m128* tr1, __m128* br0, __m128* br1) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 2));
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 2));
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 2));
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 2));
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 2));
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 2));
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 2));
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 2));
  }
  pack_2ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  pack_2ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  pack_2ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  pack_2ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
}
template <class T>
void VectorLoader<T>::load8_3ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* tl2, __m128* bl0, __m128* bl1,
                                __m128* bl2, __m128* tr0, __m128* tr1,
                                __m128* tr2, __m128* br0, __m128* br1,
                                __m128* br2) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 3));
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 3));
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 3));
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 3));
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 3));
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 3));
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 3));
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 3));
  }
  pack_3ch(&itl0, &itl1, &itl2, &itl3);
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  *tl2 = to_fp32(itl2);
  pack_3ch(&itr0, &itr1, &itr2, &itr3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  *tr2 = to_fp32(itr2);
  pack_3ch(&ibl0, &ibl1, &ibl2, &ibl3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  *bl2 = to_fp32(ibl2);
  pack_3ch(&ibr0, &ibr1, &ibr2, &ibr3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
  *br2 = to_fp32(ibr2);
}
template <class T>
void VectorLoader<T>::load8_4ch(const T* lower_ptr, const T* upper_ptr,
                                int offset0, int offset1, int offset2,
                                int offset3, __m128* tl0, __m128* tl1,
                                __m128* tl2, __m128* tl3, __m128* bl0,
                                __m128* bl1, __m128* bl2, __m128* bl3,
                                __m128* tr0, __m128* tr1, __m128* tr2,
                                __m128* tr3, __m128* br0, __m128* br1,
                                __m128* br2, __m128* br3) {
  __m128i itl0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0));
  __m128i itr0 = _mm_loadu_si128((__m128i*)(lower_ptr + offset0 + 4));
  __m128i ibl0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0));
  __m128i ibr0 = _mm_loadu_si128((__m128i*)(upper_ptr + offset0 + 4));
  __m128i itl1, itr1;
  __m128i ibl1, ibr1;
  if (offset1 == offset0) {
    itl1 = itl0;
    itr1 = itr0;
    ibl1 = ibl0;
    ibr1 = ibr0;
  } else {
    itl1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1));
    itr1 = _mm_loadu_si128((__m128i*)(lower_ptr + offset1 + 4));
    ibl1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1));
    ibr1 = _mm_loadu_si128((__m128i*)(upper_ptr + offset1 + 4));
  }
  __m128i itl2, itr2;
  __m128i ibl2, ibr2;
  if (offset2 == offset1) {
    itl2 = itl1;
    itr2 = itr1;
    ibl2 = ibl1;
    ibr2 = ibr1;
  } else {
    itl2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2));
    itr2 = _mm_loadu_si128((__m128i*)(lower_ptr + offset2 + 4));
    ibl2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2));
    ibr2 = _mm_loadu_si128((__m128i*)(upper_ptr + offset2 + 4));
  }
  __m128i itl3, itr3;
  __m128i ibl3, ibr3;
  if (offset3 == offset2) {
    itl3 = itl2;
    itr3 = itr2;
    ibl3 = ibl2;
    ibr3 = ibr2;
  } else {
    itl3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3));
    itr3 = _mm_loadu_si128((__m128i*)(lower_ptr + offset3 + 4));
    ibl3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3));
    ibr3 = _mm_loadu_si128((__m128i*)(upper_ptr + offset3 + 4));
  }
  *tl0 = to_fp32(itl0);
  *tl1 = to_fp32(itl1);
  *tl2 = to_fp32(itl2);
  *tl3 = to_fp32(itl3);
  *tr0 = to_fp32(itr0);
  *tr1 = to_fp32(itr1);
  *tr2 = to_fp32(itr2);
  *tr3 = to_fp32(itr3);
  *bl0 = to_fp32(ibl0);
  *bl1 = to_fp32(ibl1);
  *bl2 = to_fp32(ibl2);
  *bl3 = to_fp32(ibl3);
  *br0 = to_fp32(ibr0);
  *br1 = to_fp32(ibr1);
  *br2 = to_fp32(ibr2);
  *br3 = to_fp32(ibr3);
}
#endif

//
// This class stores 4 pixels with n channels packed into n SSE vector words.
// Pixel values are converted to type U and packed before storage.
// Output type U must be one of uint8, int8, uint16, int16, int32, Eigen::half,
// bfloat16 or float.
//

template <class U>
class VectorWriter {
 public:
  // convert 4 fp32 words to type U with.
  // this function calls clip.
  // resulting words are packed.
  // U must be one of uint8, int8, uint16, int16, int32, Eigen::half, bfloat16
  // or float.
  __m128i from_fp32(__m128 vec);

  // converts from fp32 to U by calling method from_fp32(...)
  // writes 4 pixels with 1 channel to destination.
  void write_1ch(U* destination, __m128* vec);

  // converts from fp32 to U by calling method from_fp32(...)
  // writes 4 pixels with 1 channel to destination.
  void write_2ch(U* destination, __m128* vec);

  // converts from fp32 to U by calling method from_fp32(...)
  // writes 4 pixels with 1 channel to destination.
  void write_3ch(U* destination, __m128* vec);

  // converts from fp32 to U by calling method from_fp32(...)
  // writes 4 pixels with 1 channel to destination.
  void write_4ch(U* destination, __m128* vec);

 private:
  // clip 4 fp32 words to prevent overflow when converting to type U.
  __m128 clip_(__m128 vec) {
    // default is to do nothing, since the packing intrinsics include clipping.
    return vec;
  }
  void write_1b_1ch(U* destination, __m128* vec) {
    __m128i ivec = from_fp32(vec[0]);
    _mm_store_ss((float*)(destination), _mm_castsi128_ps(ivec));
  }
  void write_2b_1ch(U* destination, __m128* vec) {
    __m128i ivec = from_fp32(vec[0]);
    _mm_store_sd((double*)(destination), _mm_castsi128_pd(ivec));
  }
  void write_4b_1ch(U* destination, __m128* vec) {
    __m128i ivec = from_fp32(vec[0]);
    _mm_storeu_si128((__m128i*)(destination), ivec);
  }
  void write_1b_2ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i mask = _mm_setr_epi32(-1, 0, 0, 0);
    ivec1 = _mm_or_si128(_mm_and_si128(mask, ivec1),
                         _mm_slli_si128(_mm_and_si128(mask, ivec2), 4));
    _mm_store_sd((double*)(destination), _mm_castsi128_pd(ivec1));
  }
  void write_2b_2ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i mask = _mm_setr_epi32(-1, -1, 0, 0);
    ivec1 = _mm_or_si128(_mm_and_si128(mask, ivec1),
                         _mm_slli_si128(_mm_and_si128(mask, ivec2), 8));
    _mm_storeu_si128((__m128i*)(destination), ivec1);
  }
  void write_4b_2ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    _mm_storeu_si128((__m128i*)(destination), ivec1);
    _mm_storeu_si128((__m128i*)(destination + 4), ivec2);
  }
  void write_1b_3ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i mask = _mm_setr_epi32(-1, 0, 0, 0);
    ivec1 = _mm_or_si128(_mm_and_si128(mask, ivec1),
                         _mm_slli_si128(_mm_and_si128(mask, ivec2), 4));
    _mm_store_sd((double*)(destination), _mm_castsi128_pd(ivec1));
    __m128i ivec3 = from_fp32(vec[2]);
    _mm_store_ss((float*)(destination + 8), _mm_castsi128_ps(ivec3));
  }
  void write_2b_3ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i mask = _mm_setr_epi32(-1, -1, 0, 0);
    ivec1 = _mm_or_si128(_mm_and_si128(mask, ivec1),
                         _mm_slli_si128(_mm_and_si128(mask, ivec2), 8));
    _mm_storeu_si128((__m128i*)(destination), ivec1);
    __m128i ivec3 = from_fp32(vec[2]);
    _mm_store_sd((double*)(destination + 8), _mm_castsi128_pd(ivec3));
  }
  void write_4b_3ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i ivec3 = from_fp32(vec[2]);
    _mm_storeu_si128((__m128i*)(destination), ivec1);
    _mm_storeu_si128((__m128i*)(destination + 4), ivec2);
    _mm_storeu_si128((__m128i*)(destination + 8), ivec3);
  }
  void write_1b_4ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i ivec3 = from_fp32(vec[2]);
    __m128i ivec4 = from_fp32(vec[3]);
    __m128i mask = _mm_setr_epi32(-1, 0, 0, 0);
    __m128i ivec = _mm_and_si128(mask, ivec1);
    ivec = _mm_or_si128(ivec, _mm_slli_si128(_mm_and_si128(mask, ivec2), 4));
    ivec = _mm_or_si128(ivec, _mm_slli_si128(_mm_and_si128(mask, ivec3), 8));
    ivec = _mm_or_si128(ivec, _mm_slli_si128(_mm_and_si128(mask, ivec4), 12));
    _mm_storeu_si128((__m128i*)(destination), ivec);
  }
  void write_2b_4ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i ivec3 = from_fp32(vec[2]);
    __m128i ivec4 = from_fp32(vec[3]);
    __m128i mask = _mm_setr_epi32(-1, -1, 0, 0);
    __m128i ivec = _mm_and_si128(mask, ivec1);
    ivec = _mm_or_si128(ivec, _mm_slli_si128(_mm_and_si128(mask, ivec2), 8));
    _mm_storeu_si128((__m128i*)(destination), ivec);
    ivec = _mm_and_si128(mask, ivec3);
    ivec = _mm_or_si128(ivec, _mm_slli_si128(_mm_and_si128(mask, ivec4), 8));
    _mm_storeu_si128((__m128i*)(destination + 8), ivec);
  }
  void write_4b_4ch(U* destination, __m128* vec) {
    __m128i ivec1 = from_fp32(vec[0]);
    __m128i ivec2 = from_fp32(vec[1]);
    __m128i ivec3 = from_fp32(vec[2]);
    __m128i ivec4 = from_fp32(vec[3]);
    _mm_storeu_si128((__m128i*)(destination), ivec1);
    _mm_storeu_si128((__m128i*)(destination + 4), ivec2);
    _mm_storeu_si128((__m128i*)(destination + 8), ivec3);
    _mm_storeu_si128((__m128i*)(destination + 12), ivec4);
  }
};

template <>
__m128 VectorWriter<int32>::clip_(__m128 vec) {
  // clip against low limit, -2147483648.
  // we round up to nearest number that can be represented as float.
  __m128 lt_val = _mm_set1_ps(-2147483520.0f);
  __m128 lt_mask = _mm_cmplt_ps(vec, lt_val);
  vec = _mm_or_ps(_mm_andnot_ps(lt_mask, vec), _mm_and_ps(lt_mask, lt_val));
  // clip against hight limit, 2147483647.
  // we round down to nearest number that can be represented as float.
  __m128 gt_val = _mm_set1_ps(2147483520.0f);
  __m128 gt_mask = _mm_cmpgt_ps(vec, gt_val);
  vec = _mm_or_ps(_mm_andnot_ps(gt_mask, vec), _mm_and_ps(gt_mask, gt_val));
  return vec;
}
template <>
__m128 VectorWriter<Eigen::half>::clip_(__m128 vec) {
  // clip against low limit, -65504.0f;
  __m128 lt_val = _mm_set1_ps(-65504.0f);
  __m128 lt_mask = _mm_cmplt_ps(vec, lt_val);
  vec = _mm_or_ps(_mm_andnot_ps(lt_mask, vec), _mm_and_ps(lt_mask, lt_val));
  // clip against hight limit, 65504.0f.
  __m128 gt_val = _mm_set1_ps(65504.0f);
  __m128 gt_mask = _mm_cmpgt_ps(vec, gt_val);
  vec = _mm_or_ps(_mm_andnot_ps(gt_mask, vec), _mm_and_ps(gt_mask, gt_val));
  return vec;
}

template <>
__m128i VectorWriter<uint8>::from_fp32(__m128 vec) {
  __m128i ivec = _mm_cvttps_epi32(vec);
  ivec = _mm_packs_epi32(ivec, ivec);
  return _mm_packus_epi16(ivec, ivec);
}
template <>
__m128i VectorWriter<int8>::from_fp32(__m128 vec) {
  __m128i ivec = _mm_cvttps_epi32(vec);
  ivec = _mm_packs_epi32(ivec, ivec);
  return _mm_packs_epi16(ivec, ivec);
}
template <>
__m128i VectorWriter<uint16>::from_fp32(__m128 vec) {
  __m128i ivec = _mm_cvttps_epi32(vec);
  return _mm_packus_epi32(ivec, ivec);
}
template <>
__m128i VectorWriter<int16>::from_fp32(__m128 vec) {
  __m128i ivec = _mm_cvttps_epi32(vec);
  return _mm_packs_epi32(ivec, ivec);
}
template <>
__m128i VectorWriter<int32>::from_fp32(__m128 vec) {
  return _mm_cvttps_epi32(clip_(vec));
}
template <>
__m128i VectorWriter<Eigen::half>::from_fp32(__m128 vec) {
#ifdef __F16C__
  return _mm_cvtps_ph(vec, _MM_FROUND_TO_ZERO);
#else
  // Emulation of _mm_cvtps_ph(vec, _MM_FROUND_TO_ZERO) intrinsic.
  //
  // fp16 :: 15=sign_bit, 14-10=exponent, 9-0=mantissa :: exp zero offset is 15
  //      :: exponent of -15 (all 0) and +16 (all 1) are special numbers.
  // fp32 :: 31=sign_bit, 30-23=exponent, 22-0=mantissa :: exp zero offset is
  // 127
  //      :: exponent of -127 (all 0) and +128 (all 1) are special numbers.
  //
  __m128i hw = _mm_castps_si128(vec);
  // ..extract fp32 exponent and mantissa
  __m128i fp16_sign_bit_msb = _mm_and_si128(_mm_set1_epi32(-2147483648), hw);
  __m128i fp32_exponent_lsb =
      _mm_and_si128(_mm_set1_epi32(255), _mm_srli_epi32(hw, 23));
  __m128i fp32_mantissa = _mm_and_si128(_mm_set1_epi32(8388607), hw);
  // ..test for NaN
  __m128i exponent_ones =
      _mm_cmpeq_epi32(fp32_exponent_lsb, _mm_set1_epi32(255));
  __m128i mantissa_zero = _mm_cmpeq_epi32(fp32_mantissa, _mm_setzero_si128());
  __m128i infinity_mask = _mm_and_si128(mantissa_zero, exponent_ones);
  // ..have to test for NaN on fp32 bits to avoid converting NaN to infinity
  __m128i NaN_mask = _mm_andnot_si128(mantissa_zero, exponent_ones);
  // ..compensate for exponent zero offset difference
  __m128i fp16_exponent_lsb =
      _mm_sub_epi32(fp32_exponent_lsb, _mm_set1_epi32(112));
  // ..clip output if fp16_exponent > 30
  __m128i saturated_mask = _mm_andnot_si128(
      exponent_ones, _mm_cmpgt_epi32(fp16_exponent_lsb, _mm_set1_epi32(30)));
  // ..generate subnormal number if fp16_exponent == 0
  // ..flush to zero if fp16_exponent < 0
  __m128i subnormal_mask =
      _mm_cmpeq_epi32(fp16_exponent_lsb, _mm_setzero_si128());
  __m128i underflow_mask =
      _mm_cmplt_epi32(fp16_exponent_lsb, _mm_setzero_si128());
  __m128i fp16_mantissa = _mm_srli_epi32(fp32_mantissa, 13);
  // ..handle abnormal values
  __m128i normal_number =
      _mm_or_si128(_mm_slli_epi32(fp16_exponent_lsb, 10), fp16_mantissa);
  __m128i subnormal_number =
      _mm_or_si128(_mm_set1_epi32(512), _mm_srli_epi32(fp16_mantissa, 1));
  __m128i saturated_number = _mm_set1_epi32(31743);
  __m128i infinity_number = _mm_set1_epi32(31744);
  __m128i NaN_number = _mm_set1_epi32(32256);
  __m128i number = _mm_andnot_si128(underflow_mask, normal_number);
  number = _mm_or_si128(_mm_andnot_si128(subnormal_mask, number),
                        _mm_and_si128(subnormal_mask, subnormal_number));
  number = _mm_or_si128(_mm_andnot_si128(saturated_mask, number),
                        _mm_and_si128(saturated_mask, saturated_number));
  number = _mm_or_si128(_mm_andnot_si128(infinity_mask, number),
                        _mm_and_si128(infinity_mask, infinity_number));
  number = _mm_or_si128(_mm_andnot_si128(NaN_mask, number),
                        _mm_and_si128(NaN_mask, NaN_number));
  // ..or in sign bit
  number = _mm_or_si128(fp16_sign_bit_msb, _mm_slli_epi32(number, 16));
  // ..move 16 bit words to lower portion of sse vector;
  __m128i shuf_from_hi32 = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -128, -128,
                                         -128, -128, -128, -128, -128, -128);
  number = _mm_shuffle_epi8(number, shuf_from_hi32);
  return number;
#endif
}
template <>
__m128i VectorWriter<bfloat16>::from_fp32(__m128 vec) {
  // casting from float to bfloat16 simply means >> 16
  // we do this with a shuffle that also moves everything to lower portion of
  // sse vector word
  __m128i shuf_from_hi32 = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15, -128, -128,
                                         -128, -128, -128, -128, -128, -128);
  return _mm_shuffle_epi8(_mm_castps_si128(vec), shuf_from_hi32);
}
template <>
__m128i VectorWriter<float>::from_fp32(__m128 vec) {
  // nothing to do in this case
  return _mm_castps_si128(vec);
}

template <>
void VectorWriter<uint8>::write_1ch(uint8* destination, __m128* vec) {
  write_1b_1ch(destination, vec);
}
template <>
void VectorWriter<int8>::write_1ch(int8* destination, __m128* vec) {
  write_1b_1ch(destination, vec);
}
template <>
void VectorWriter<uint16>::write_1ch(uint16* destination, __m128* vec) {
  write_2b_1ch(destination, vec);
}
template <>
void VectorWriter<int16>::write_1ch(int16* destination, __m128* vec) {
  write_2b_1ch(destination, vec);
}
template <>
void VectorWriter<int32>::write_1ch(int32* destination, __m128* vec) {
  write_4b_1ch(destination, vec);
}
template <>
void VectorWriter<Eigen::half>::write_1ch(Eigen::half* destination,
                                          __m128* vec) {
  write_2b_1ch(destination, vec);
}
template <>
void VectorWriter<bfloat16>::write_1ch(bfloat16* destination, __m128* vec) {
  write_2b_1ch(destination, vec);
}
template <>
void VectorWriter<float>::write_1ch(float* destination, __m128* vec) {
  _mm_storeu_si128((__m128i*)(destination), _mm_castps_si128(vec[0]));
}

template <>
void VectorWriter<uint8>::write_2ch(uint8* destination, __m128* vec) {
  write_1b_2ch(destination, vec);
}
template <>
void VectorWriter<int8>::write_2ch(int8* destination, __m128* vec) {
  write_1b_2ch(destination, vec);
}
template <>
void VectorWriter<uint16>::write_2ch(uint16* destination, __m128* vec) {
  write_2b_2ch(destination, vec);
}
template <>
void VectorWriter<int16>::write_2ch(int16* destination, __m128* vec) {
  write_2b_2ch(destination, vec);
}
template <>
void VectorWriter<int32>::write_2ch(int32* destination, __m128* vec) {
  write_4b_2ch(destination, vec);
}
template <>
void VectorWriter<Eigen::half>::write_2ch(Eigen::half* destination,
                                          __m128* vec) {
  write_2b_2ch(destination, vec);
}
template <>
void VectorWriter<bfloat16>::write_2ch(bfloat16* destination, __m128* vec) {
  write_2b_2ch(destination, vec);
}
template <>
void VectorWriter<float>::write_2ch(float* destination, __m128* vec) {
  _mm_storeu_si128((__m128i*)(destination), _mm_castps_si128(vec[0]));
  _mm_storeu_si128((__m128i*)(destination + 4), _mm_castps_si128(vec[1]));
}

template <>
void VectorWriter<uint8>::write_3ch(uint8* destination, __m128* vec) {
  write_1b_3ch(destination, vec);
}
template <>
void VectorWriter<int8>::write_3ch(int8* destination, __m128* vec) {
  write_1b_3ch(destination, vec);
}
template <>
void VectorWriter<uint16>::write_3ch(uint16* destination, __m128* vec) {
  write_2b_3ch(destination, vec);
}
template <>
void VectorWriter<int16>::write_3ch(int16* destination, __m128* vec) {
  write_2b_3ch(destination, vec);
}
template <>
void VectorWriter<int32>::write_3ch(int32* destination, __m128* vec) {
  write_4b_3ch(destination, vec);
}
template <>
void VectorWriter<Eigen::half>::write_3ch(Eigen::half* destination,
                                          __m128* vec) {
  write_2b_3ch(destination, vec);
}
template <>
void VectorWriter<bfloat16>::write_3ch(bfloat16* destination, __m128* vec) {
  write_2b_3ch(destination, vec);
}
template <>
void VectorWriter<float>::write_3ch(float* destination, __m128* vec) {
  _mm_storeu_si128((__m128i*)(destination), _mm_castps_si128(vec[0]));
  _mm_storeu_si128((__m128i*)(destination + 4), _mm_castps_si128(vec[1]));
  _mm_storeu_si128((__m128i*)(destination + 8), _mm_castps_si128(vec[2]));
}

template <>
void VectorWriter<uint8>::write_4ch(uint8* destination, __m128* vec) {
  write_1b_4ch(destination, vec);
}
template <>
void VectorWriter<int8>::write_4ch(int8* destination, __m128* vec) {
  write_1b_4ch(destination, vec);
}
template <>
void VectorWriter<uint16>::write_4ch(uint16* destination, __m128* vec) {
  write_2b_4ch(destination, vec);
}
template <>
void VectorWriter<int16>::write_4ch(int16* destination, __m128* vec) {
  write_2b_4ch(destination, vec);
}
template <>
void VectorWriter<int32>::write_4ch(int32* destination, __m128* vec) {
  write_4b_4ch(destination, vec);
}
template <>
void VectorWriter<Eigen::half>::write_4ch(Eigen::half* destination,
                                          __m128* vec) {
  write_2b_4ch(destination, vec);
}
template <>
void VectorWriter<bfloat16>::write_4ch(bfloat16* destination, __m128* vec) {
  write_2b_4ch(destination, vec);
}
template <>
void VectorWriter<float>::write_4ch(float* destination, __m128* vec) {
  _mm_storeu_si128((__m128i*)(destination), _mm_castps_si128(vec[0]));
  _mm_storeu_si128((__m128i*)(destination + 4), _mm_castps_si128(vec[1]));
  _mm_storeu_si128((__m128i*)(destination + 8), _mm_castps_si128(vec[2]));
  _mm_storeu_si128((__m128i*)(destination + 12), _mm_castps_si128(vec[3]));
}

template <class T, class U>
class CropResizeCastImage : public VectorLoader<T>, public VectorWriter<U> {
 public:
  CropResizeCastImage(const int in_height, const int in_width,
                      const int out_height, const int out_width,
                      const int channels, const int min_ix, const int max_ix,
                      const CachedInterpolation* xs, const int min_iy,
                      const int max_iy, const CachedInterpolation* ys,
                      const float extrapolated_value, const bool flip_x,
                      const bool flip_y, const bool verbose = false,
                      const int allowed_load_groups = 15)
      : verbose_(verbose),
        allowed_load_groups_(allowed_load_groups),
        in_height_(in_height),
        in_width_(in_width),
        out_height_(out_height),
        out_width_(out_width),
        channels_(channels),
        min_ix_(min_ix),
        max_ix_(max_ix),
        min_iy_(min_iy),
        max_iy_(max_iy),
        ys_(ys),
        extrapolated_value_(extrapolated_value),
        flip_x_(flip_x),
        flip_y_(flip_y),
        in_row_size_(in_width * channels),
        in_row_size_bytes_(in_width * channels * sizeof(T)),
        out_row_size_(out_width * channels),
        x0_(flip_x ? out_width - 1 - max_ix : min_ix),
        x1_(flip_x ? out_width - 1 - min_ix : max_ix),
        y0_(flip_y ? out_height - 1 - max_iy : min_iy),
        y1_(flip_y ? out_height - 1 - min_iy : max_iy) {
    if (min_ix_ <= max_ix_ && min_iy_ <= max_iy_) {
      // copy xs values, but filter out the following:
      // xs[].lower == xs[].upper AND xs[].lerp == 0
      // xs[].lower == xs[].upper AND xs[].lerp == 1
      xs_ = new CachedInterpolation[max_ix_ - min_ix_ + 1];
      for (int i = min_ix_; i <= max_ix_; ++i) {
        int ix = i - min_ix_;
        int xs_lower = xs[ix].lower / channels_;
        int xs_upper = xs[ix].upper / channels_;
        if (xs_lower == xs_upper) {
          if (xs[ix].lerp == 0.0f && xs_lower + 1 < in_width) {
            // upper weight is zero
            xs_upper = xs_lower + 1;
          } else if (xs[ix].lerp == 1.0f && xs_upper - 1 >= 0) {
            // lower weight is zero
            xs_lower = xs_upper - 1;
          }
        }
        xs_[ix].lower = xs_lower * channels_;
        xs_[ix].upper = xs_upper * channels_;
        xs_[ix].lerp = xs[ix].lerp;
      }
      _u_min_val = std::numeric_limits<U>::min();
      _u_max_val = std::numeric_limits<U>::max();
      _f_min_val = static_cast<float>(_u_min_val);
      _f_max_val = static_cast<float>(_u_max_val);
      Configure_();
    } else {
      // crop region outside of input image.
      // extrapolation only.
      general_x_ = NULL;
      load1_x_ = NULL;
      load2_x_ = NULL;
      load4_x_ = NULL;
      load8_x_ = NULL;
      load1_offsets_ = NULL;
      load2_offsets_ = NULL;
      load4_offsets_ = NULL;
      load8_offsets_ = NULL;
      load1_shuffle_masks_ = NULL;
      load2_shuffle_masks_ = NULL;
      load1_mmxs_lerp_ = NULL;
      load2_mmxs_lerp_ = NULL;
      load4_mmxs_lerp_ = NULL;
      load8_mmxs_lerp_ = NULL;
      xs_ = NULL;
    }
  }
  ~CropResizeCastImage() {
    if (general_x_ != NULL) delete[] general_x_;
    if (load1_x_ != NULL) delete[] load1_x_;
    if (load2_x_ != NULL) delete[] load2_x_;
    if (load4_x_ != NULL) delete[] load4_x_;
    if (load8_x_ != NULL) delete[] load8_x_;
    if (load1_offsets_ != NULL) delete[] load1_offsets_;
    if (load2_offsets_ != NULL) delete[] load2_offsets_;
    if (load4_offsets_ != NULL) delete[] load4_offsets_;
    if (load8_offsets_ != NULL) delete[] load8_offsets_;
    if (load1_shuffle_masks_ != NULL) delete[] load1_shuffle_masks_;
    if (load2_shuffle_masks_ != NULL) delete[] load2_shuffle_masks_;
    if (load1_mmxs_lerp_ != NULL) delete[] load1_mmxs_lerp_;
    if (load2_mmxs_lerp_ != NULL) delete[] load2_mmxs_lerp_;
    if (load4_mmxs_lerp_ != NULL) delete[] load4_mmxs_lerp_;
    if (load8_mmxs_lerp_ != NULL) delete[] load8_mmxs_lerp_;
    delete[] xs_;
  }

 private:
  // constructor arguments
  const bool verbose_;
  // this value is meant for unit testing.
  // set this to 15 for normal execution.
  // its an OR of flags for the different load group.
  //  1 -> load4from1
  //  2 -> load4from2
  //  4 -> load4from4
  //  8 -> load4from8
  const int allowed_load_groups_;
  const int in_height_, in_width_, out_height_, out_width_;
  const int channels_;
  const int min_ix_, max_ix_, min_iy_, max_iy_;
  const CachedInterpolation* ys_;
  CachedInterpolation* xs_;
  const float extrapolated_value_;
  const bool flip_x_, flip_y_;
  // computed arguments
  const int in_row_size_;
  const int in_row_size_bytes_;
  const int out_row_size_;
  const int x0_, x1_;
  const int y0_, y1_;

  // helper methods
  void ResizeRow_load1_1ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load2_1ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load4_1ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load8_1ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load1_2ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load2_2ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load4_2ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load8_2ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load1_3ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load2_3ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load4_3ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load8_3ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load1_4ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load2_4ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load4_4ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_load8_4ch_(const __m128 y_lerp, const T* ysA_input_lower_ptr,
                            const T* ysA_input_upper_ptr, U* ysA_output_ptr);
  void ResizeRow_general_(const float ys_lerp, const T* ysA_input_lower_ptr,
                          const T* ysA_input_upper_ptr, U* ysA_output_ptr);

  // configuration parameters
  int num_general_, num_load1_, num_load2_, num_load4_, num_load8_;
  int *load1_offsets_, *load2_offsets_, *load4_offsets_, *load8_offsets_;
  int *general_x_, *load1_x_, *load2_x_, *load4_x_, *load8_x_;
  __m128i *load1_shuffle_masks_, *load2_shuffle_masks_;
  __m128 *load1_mmxs_lerp_, *load2_mmxs_lerp_, *load4_mmxs_lerp_,
      *load8_mmxs_lerp_;
  float _f_min_val, _f_max_val;
  U _u_min_val, _u_max_val;
  // configuration methods
  void Configure_();
  int DetermineLoadGroup_(const int x);
  bool ComputeXIndexRange_(const int x, int* min_xidx, int* max_xidx);
  bool Load1_ok_(
      const int min_xidx,
      const int max_xidx);  // xs - pointer to first xs for this load group
  bool Load2_ok_(
      const int min_xidx,
      const int max_xidx);  // xs - pointer to first xs for this load group
  bool Load4_ok_(const int min_xidx, const int max_xidx);
  bool Load8_ok_(const int min_xidx, const int max_xidx);

 public:
  //
  // public client methods
  //

  // convenience function that determines if clipping is necessary
  // in order to prevent overflow when casting to the output type U.
  static bool clip_necessary();

  // resize image
  void Resize(const T* input_image, U* output_image);
};

template <class T, class U>
void CropResizeCastImage<T, U>::Resize(const T* input_image, U* output_image) {
  //
  U uEx = cast_to<U>(extrapolated_value_, _f_min_val, _f_max_val, _u_min_val,
                     _u_max_val);
  // extrapolate top
  if (min_iy_ > 0) {
    U* p = flip_y_ ? output_image + out_row_size_ * (out_height_ - min_iy_)
                   : output_image;
    int nn = out_row_size_ * min_iy_;
    for (int i = 0; i < nn; ++i) p[i] = uEx;
  }
  // extrapolate bottom
  if (max_iy_ < out_height_ - 1) {
    U* p =
        flip_y_ ? output_image : output_image + out_row_size_ * (max_iy_ + 1);
    int nn = out_row_size_ * (out_height_ - 1 - max_iy_);
    for (int i = 0; i < nn; ++i) p[i] = uEx;
  }
  // extrapolate left
  if (min_ix_ > 0) {
    for (int iy = min_iy_; iy <= max_iy_; ++iy) {
      int xx0 = flip_x_ ? (out_width_ - min_ix_) * channels_ : 0;
      int nxx = min_ix_ * channels_;
      U* p = output_image + xx0 +
             out_row_size_ * (flip_y_ ? out_height_ - 1 - iy : iy);
      for (int ix = 0; ix < nxx; ++ix) {
        p[ix] = uEx;
      }
    }
  }
  // extrapolate right
  if (max_ix_ < out_width_ - 1) {
    for (int iy = min_iy_; iy <= max_iy_; ++iy) {
      int xx0 = flip_x_ ? 0 : (max_ix_ + 1) * channels_;
      int nxx = (out_width_ - 1 - max_ix_) * channels_;
      U* p = output_image + xx0 +
             out_row_size_ * (flip_y_ ? out_height_ - 1 - iy : iy);
      for (int ix = 0; ix < nxx; ++ix) {
        p[ix] = uEx;
      }
    }
  }
  // interpolation region
  if (min_ix_ <= max_ix_ && min_iy_ <= max_iy_) {
    int y = y0_;
    for (y = y0_; y + 1 <= y1_; y += 2) {
      const int iyA = flip_y_ ? out_height_ - 1 - min_iy_ - y : y - min_iy_;
      const float yA_lerp = ys_[iyA].lerp;
      const __m128 ysA_lerp = _mm_set1_ps(yA_lerp);
      const T* ysA_input_lower_ptr =
          input_image + ys_[iyA].lower * in_width_ * channels_;
      const T* ysA_input_upper_ptr =
          input_image + ys_[iyA].upper * in_width_ * channels_;
      U* ysA_output_ptr = output_image + y * out_width_ * channels_;
      const int iyB =
          flip_y_ ? out_height_ - 1 - min_iy_ - (y + 1) : (y + 1) - min_iy_;
      const float yB_lerp = ys_[iyB].lerp;
      const __m128 ysB_lerp = _mm_set1_ps(yB_lerp);
      const T* ysB_input_lower_ptr =
          input_image + ys_[iyB].lower * in_width_ * channels_;
      const T* ysB_input_upper_ptr =
          input_image + ys_[iyB].upper * in_width_ * channels_;
      U* ysB_output_ptr = output_image + (y + 1) * out_width_ * channels_;
      if (channels_ == 1) {
        this->ResizeRow_load1_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load1_1ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load2_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_1ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load4_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_1ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load8_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_1ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yB_lerp, ysB_input_lower_ptr,
                                 ysB_input_upper_ptr, ysB_output_ptr);
      } else if (channels_ == 2) {
        this->ResizeRow_load1_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load1_2ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load2_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_2ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load4_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_2ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load8_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_2ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yB_lerp, ysB_input_lower_ptr,
                                 ysB_input_upper_ptr, ysB_output_ptr);
      } else if (channels_ == 3) {
        this->ResizeRow_load1_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load1_3ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load2_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_3ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load4_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_3ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load8_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_3ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yB_lerp, ysB_input_lower_ptr,
                                 ysB_input_upper_ptr, ysB_output_ptr);
      } else if (channels_ == 4) {
        this->ResizeRow_load1_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load1_4ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load2_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_4ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load4_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_4ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_load8_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_4ch_(ysB_lerp, ysB_input_lower_ptr,
                                   ysB_input_upper_ptr, ysB_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yB_lerp, ysB_input_lower_ptr,
                                 ysB_input_upper_ptr, ysB_output_ptr);
      } else {
        assert(false);
      }
    }
    for (; y <= y1_; ++y) {
      const int iyA = flip_y_ ? out_height_ - 1 - min_iy_ - y : y - min_iy_;
      const float yA_lerp = ys_[iyA].lerp;
      const __m128 ysA_lerp = _mm_set1_ps(yA_lerp);
      const T* ysA_input_lower_ptr =
          input_image + ys_[iyA].lower * in_width_ * channels_;
      const T* ysA_input_upper_ptr =
          input_image + ys_[iyA].upper * in_width_ * channels_;
      U* ysA_output_ptr = output_image + y * out_width_ * channels_;
      if (channels_ == 1) {
        this->ResizeRow_load1_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_1ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
      } else if (channels_ == 2) {
        this->ResizeRow_load1_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_2ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
      } else if (channels_ == 3) {
        this->ResizeRow_load1_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_3ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
      } else if (channels_ == 4) {
        this->ResizeRow_load1_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load2_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load4_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_load8_4ch_(ysA_lerp, ysA_input_lower_ptr,
                                   ysA_input_upper_ptr, ysA_output_ptr);
        this->ResizeRow_general_(yA_lerp, ysA_input_lower_ptr,
                                 ysA_input_upper_ptr, ysA_output_ptr);
      } else {
        assert(false);
      }
    }
  }
}

template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_general_(const float ys_lerp,
                                                   const T* ys_input_lower_ptr,
                                                   const T* ys_input_upper_ptr,
                                                   U* output_y_ptr) {
  for (int current = 0; current < num_general_; ++current) {
    int x = general_x_[current];
    const int ix = flip_x_ ? out_width_ - 1 - min_ix_ - x : x - min_ix_;
    const int xs_lower = xs_[ix].lower;
    const int xs_upper = xs_[ix].upper;
    const float xs_lerp = xs_[ix].lerp;
    for (int ichan = 0; ichan < channels_; ++ichan) {
      const float top_left0(ys_input_lower_ptr[xs_lower + ichan]);
      const float top_right0(ys_input_lower_ptr[xs_upper + ichan]);
      const float bottom_left0(ys_input_upper_ptr[xs_lower + ichan]);
      const float bottom_right0(ys_input_upper_ptr[xs_upper + ichan]);
      float result0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                   bottom_right0, xs_lerp, ys_lerp);
      output_y_ptr[x * channels_ + ichan] =
          cast_to<U>(result0, _f_min_val, _f_max_val, _u_min_val, _u_max_val);
    }
  }
}

#define CHANNELS 1
// Resize all points that fall in the 'load4from1' group for an entire row of a
// 1 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load1_1ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load1_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load1_shuffle_masks_ + current * CHANNELS * 3);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, right0;
    this->load1_1ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &left0, &right0);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
#else
    __m128 tl0, bl0, tr0, br0;
    this->load1_1ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &tl0, &bl0, &tr0,
                    &br0);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
#endif
#ifdef __AVX2__
    __m128 res[1];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    this->write_1ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#else
    __m128 res[1];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    this->write_1ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from2' group for an entire row of a
// 1 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load2_1ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load2_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load2_shuffle_masks_ + current * CHANNELS * 2);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, right0;
    this->load2_1ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &left0, &right0);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
#else
    __m128 tl0, bl0, tr0, br0;
    this->load2_1ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &tl0, &bl0, &tr0,
                    &br0);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
#endif
#ifdef __AVX2__
    __m128 res[1];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    this->write_1ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#else
    __m128 res[1];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    this->write_1ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from4' group for an entire row of a
// 1 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load4_1ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load4_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load4_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, right0;
    this->load4_1ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &left0, &right0);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
#else
    __m128 tl0, bl0, tr0, br0;
    this->load4_1ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &tl0, &bl0, &tr0, &br0);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
#endif
#ifdef __AVX2__
    __m128 res[1];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    this->write_1ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#else
    __m128 res[1];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    this->write_1ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from8' group for an entire row of a
// 1 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load8_1ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load8_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load8_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, right0;
    this->load8_1ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &left0, &right0);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
#else
    __m128 tl0, bl0, tr0, br0;
    this->load8_1ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &tl0, &bl0, &tr0, &br0);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
#endif
#ifdef __AVX2__
    __m128 res[1];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    this->write_1ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#else
    __m128 res[1];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    this->write_1ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#endif
  }
}
#undef CHANNELS

#define CHANNELS 2
// Resize all points that fall in the 'load4from1' group for an entire row of a
// 2 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load1_2ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load1_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load1_shuffle_masks_ + current * CHANNELS * 3);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, right0, right1;
    this->load1_2ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &left0, &left1,
                    &right0, &right1);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
#else
    __m128 tl0, tl1, bl0, bl1, tr0, tr1, br0, br1;
    this->load1_2ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &tl0, &tl1, &bl0,
                    &bl1, &tr0, &tr1, &br0, &br1);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
#endif
#ifdef __AVX2__
    __m128 res[2];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    this->write_2ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#else
    __m128 res[2];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    this->write_2ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from2' group for an entire row of a
// 2 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load2_2ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load2_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load2_shuffle_masks_ + current * CHANNELS * 2);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, right0, right1;
    this->load2_2ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &left0, &left1,
                    &right0, &right1);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
#else
    __m128 tl0, tl1, bl0, bl1, tr0, tr1, br0, br1;
    this->load2_2ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &tl0, &tl1, &bl0,
                    &bl1, &tr0, &tr1, &br0, &br1);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
#endif
#ifdef __AVX2__
    __m128 res[2];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    this->write_2ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#else
    __m128 res[2];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    this->write_2ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from4' group for an entire row of a
// 2 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load4_2ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load4_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load4_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, right0, right1;
    this->load4_2ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &left0, &left1, &right0, &right1);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
#else
    __m128 tl0, tl1, bl0, bl1, tr0, tr1, br0, br1;
    this->load4_2ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &tl0, &tl1, &bl0, &bl1, &tr0, &tr1,
        &br0, &br1);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
#endif
#ifdef __AVX2__
    __m128 res[2];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    this->write_2ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#else
    __m128 res[2];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    this->write_2ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from8' group for an entire row of a
// 2 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load8_2ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load8_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load8_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, right0, right1;
    this->load8_2ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &left0, &left1, &right0, &right1);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
#else
    __m128 tl0, tl1, bl0, bl1, tr0, tr1, br0, br1;
    this->load8_2ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &tl0, &tl1, &bl0, &bl1, &tr0, &tr1,
        &br0, &br1);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
#endif
#ifdef __AVX2__
    __m128 res[2];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    this->write_2ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#else
    __m128 res[2];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    this->write_2ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#endif
  }
}
#undef CHANNELS

#define CHANNELS 3
// Resize all points that fall in the 'load4from1' group for an entire row of a
// 3 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load1_3ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load1_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load1_shuffle_masks_ + current * CHANNELS * 3);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, left2, right0, right1, right2;
    this->load1_3ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &left0, &left1,
                    &left2, &right0, &right1, &right2);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
#else
    __m128 tl0, tl1, tl2, bl0, bl1, bl2, tr0, tr1, tr2, br0, br1, br2;
    this->load1_3ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &tl0, &tl1, &tl2,
                    &bl0, &bl1, &bl2, &tr0, &tr1, &tr2, &br0, &br1, &br2);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
#endif
#ifdef __AVX2__
    __m128 res[3];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    this->write_3ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#else
    __m128 res[3];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    this->write_3ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from2' group for an entire row of a
// 3 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load2_3ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load2_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load2_shuffle_masks_ + current * CHANNELS * 2);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, left2, right0, right1, right2;
    this->load2_3ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &left0, &left1,
                    &left2, &right0, &right1, &right2);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
#else
    __m128 tl0, tl1, tl2, bl0, bl1, bl2, tr0, tr1, tr2, br0, br1, br2;
    this->load2_3ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &tl0, &tl1, &tl2,
                    &bl0, &bl1, &bl2, &tr0, &tr1, &tr2, &br0, &br1, &br2);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
#endif
#ifdef __AVX2__
    __m128 res[3];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    this->write_3ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#else
    __m128 res[3];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    this->write_3ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from4' group for an entire row of a
// 3 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load4_3ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load4_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load4_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, left2, right0, right1, right2;
    this->load4_3ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &left0, &left1, &left2, &right0,
        &right1, &right2);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
#else
    __m128 tl0, tl1, tl2, bl0, bl1, bl2, tr0, tr1, tr2, br0, br1, br2;
    this->load4_3ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &tl0, &tl1, &tl2, &bl0, &bl1, &bl2,
        &tr0, &tr1, &tr2, &br0, &br1, &br2);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
#endif
#ifdef __AVX2__
    __m128 res[3];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    this->write_3ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#else
    __m128 res[3];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    this->write_3ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from8' group for an entire row of a
// 3 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load8_3ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load8_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load8_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, left2, right0, right1, right2;
    this->load8_3ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &left0, &left1, &left2, &right0,
        &right1, &right2);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
#else
    __m128 tl0, tl1, tl2, bl0, bl1, bl2, tr0, tr1, tr2, br0, br1, br2;
    this->load8_3ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &tl0, &tl1, &tl2, &bl0, &bl1, &bl2,
        &tr0, &tr1, &tr2, &br0, &br1, &br2);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
#endif
#ifdef __AVX2__
    __m128 res[3];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    this->write_3ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#else
    __m128 res[3];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    this->write_3ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#endif
  }
}
#undef CHANNELS

#define CHANNELS 4
// Resize all points that fall in the 'load4from1' group for an entire row of a
// 4 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load1_4ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load1_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load1_shuffle_masks_ + current * CHANNELS * 3);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, left2, left3, right0, right1, right2, right3;
    this->load1_4ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &left0, &left1,
                    &left2, &left3, &right0, &right1, &right2, &right3);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[3])));
    __m256 hori3 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right3, left3), left3);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
    __m128 top3 = _mm256_castps256_ps128(hori3);
    __m128 bot3 = _mm256_extractf128_ps(hori3, 1);
#else
    __m128 tl0, tl1, tl2, tl3, bl0, bl1, bl2, bl3, tr0, tr1, tr2, tr3, br0, br1,
        br2, br3;
    this->load1_4ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load1_offsets_[current], shuffle_masks, &tl0, &tl1, &tl2,
                    &tl3, &bl0, &bl1, &bl2, &bl3, &tr0, &tr1, &tr2, &tr3, &br0,
                    &br1, &br2, &br3);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
    x_lerp = mmxs_lerp[3];
    __m128 top3 = _mm_add_ps(tl3, _mm_mul_ps(x_lerp, _mm_sub_ps(tr3, tl3)));
    __m128 bot3 = _mm_add_ps(bl3, _mm_mul_ps(x_lerp, _mm_sub_ps(br3, bl3)));
#endif
#ifdef __AVX2__
    __m128 res[4];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    res[3] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot3, top3), top3);
    this->write_4ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#else
    __m128 res[4];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    res[3] = _mm_add_ps(top3, _mm_mul_ps(y_lerp, _mm_sub_ps(bot3, top3)));
    this->write_4ch(ysA_output_ptr + load1_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from2' group for an entire row of a
// 4 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load2_4ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load2_; ++current) {
    __m128* mmxs_lerp =
        (__m128*)(load2_shuffle_masks_ + current * CHANNELS * 2);
    __m128i* shuffle_masks = (__m128i*)mmxs_lerp + CHANNELS;
#ifdef __AVX2__
    __m256 left0, left1, left2, left3, right0, right1, right2, right3;
    this->load2_4ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &left0, &left1,
                    &left2, &left3, &right0, &right1, &right2, &right3);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[3])));
    __m256 hori3 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right3, left3), left3);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
    __m128 top3 = _mm256_castps256_ps128(hori3);
    __m128 bot3 = _mm256_extractf128_ps(hori3, 1);
#else
    __m128 tl0, tl1, tl2, tl3, bl0, bl1, bl2, bl3, tr0, tr1, tr2, tr3, br0, br1,
        br2, br3;
    this->load2_4ch(ysA_input_lower_ptr, ysA_input_upper_ptr,
                    load2_offsets_[current], shuffle_masks, &tl0, &tl1, &tl2,
                    &tl3, &bl0, &bl1, &bl2, &bl3, &tr0, &tr1, &tr2, &tr3, &br0,
                    &br1, &br2, &br3);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
    x_lerp = mmxs_lerp[3];
    __m128 top3 = _mm_add_ps(tl3, _mm_mul_ps(x_lerp, _mm_sub_ps(tr3, tl3)));
    __m128 bot3 = _mm_add_ps(bl3, _mm_mul_ps(x_lerp, _mm_sub_ps(br3, bl3)));
#endif
#ifdef __AVX2__
    __m128 res[4];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    res[3] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot3, top3), top3);
    this->write_4ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#else
    __m128 res[4];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    res[3] = _mm_add_ps(top3, _mm_mul_ps(y_lerp, _mm_sub_ps(bot3, top3)));
    this->write_4ch(ysA_output_ptr + load2_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from4' group for an entire row of a
// 4 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load4_4ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load4_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load4_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, left2, left3, right0, right1, right2, right3;
    this->load4_4ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &left0, &left1, &left2, &left3,
        &right0, &right1, &right2, &right3);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[3])));
    __m256 hori3 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right3, left3), left3);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
    __m128 top3 = _mm256_castps256_ps128(hori3);
    __m128 bot3 = _mm256_extractf128_ps(hori3, 1);
#else
    __m128 tl0, tl1, tl2, tl3, bl0, bl1, bl2, bl3, tr0, tr1, tr2, tr3, br0, br1,
        br2, br3;
    this->load4_4ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load4_offsets_[current * 4],
        load4_offsets_[current * 4 + 1], load4_offsets_[current * 4 + 2],
        load4_offsets_[current * 4 + 3], &tl0, &tl1, &tl2, &tl3, &bl0, &bl1,
        &bl2, &bl3, &tr0, &tr1, &tr2, &tr3, &br0, &br1, &br2, &br3);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
    x_lerp = mmxs_lerp[3];
    __m128 top3 = _mm_add_ps(tl3, _mm_mul_ps(x_lerp, _mm_sub_ps(tr3, tl3)));
    __m128 bot3 = _mm_add_ps(bl3, _mm_mul_ps(x_lerp, _mm_sub_ps(br3, bl3)));
#endif
#ifdef __AVX2__
    __m128 res[4];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    res[3] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot3, top3), top3);
    this->write_4ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#else
    __m128 res[4];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    res[3] = _mm_add_ps(top3, _mm_mul_ps(y_lerp, _mm_sub_ps(bot3, top3)));
    this->write_4ch(ysA_output_ptr + load4_x_[current] * CHANNELS, res);
#endif
  }
}
// Resize all points that fall in the 'load4from8' group for an entire row of a
// 4 channel image.
template <class T, class U>
void CropResizeCastImage<T, U>::ResizeRow_load8_4ch_(
    const __m128 y_lerp, const T* ysA_input_lower_ptr,
    const T* ysA_input_upper_ptr, U* ysA_output_ptr) {
  for (int current = 0; current < num_load8_; ++current) {
    __m128* mmxs_lerp = (__m128*)(load8_mmxs_lerp_ + current * CHANNELS);
#ifdef __AVX2__
    __m256 left0, left1, left2, left3, right0, right1, right2, right3;
    this->load8_4ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &left0, &left1, &left2, &left3,
        &right0, &right1, &right2, &right3);

    __m256 x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[0])));
    __m256 hori0 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right0, left0), left0);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[1])));
    __m256 hori1 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right1, left1), left1);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[2])));
    __m256 hori2 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right2, left2), left2);
    x_lerp = _mm256_castsi256_ps(
        _mm256_broadcastsi128_si256(_mm_castps_si128(mmxs_lerp[3])));
    __m256 hori3 = _mm256_fmadd_ps(x_lerp, _mm256_sub_ps(right3, left3), left3);

    __m128 top0 = _mm256_castps256_ps128(hori0);
    __m128 bot0 = _mm256_extractf128_ps(hori0, 1);
    __m128 top1 = _mm256_castps256_ps128(hori1);
    __m128 bot1 = _mm256_extractf128_ps(hori1, 1);
    __m128 top2 = _mm256_castps256_ps128(hori2);
    __m128 bot2 = _mm256_extractf128_ps(hori2, 1);
    __m128 top3 = _mm256_castps256_ps128(hori3);
    __m128 bot3 = _mm256_extractf128_ps(hori3, 1);
#else
    __m128 tl0, tl1, tl2, tl3, bl0, bl1, bl2, bl3, tr0, tr1, tr2, tr3, br0, br1,
        br2, br3;
    this->load8_4ch(
        ysA_input_lower_ptr, ysA_input_upper_ptr, load8_offsets_[current * 4],
        load8_offsets_[current * 4 + 1], load8_offsets_[current * 4 + 2],
        load8_offsets_[current * 4 + 3], &tl0, &tl1, &tl2, &tl3, &bl0, &bl1,
        &bl2, &bl3, &tr0, &tr1, &tr2, &tr3, &br0, &br1, &br2, &br3);

    __m128 x_lerp = mmxs_lerp[0];
    __m128 top0 = _mm_add_ps(tl0, _mm_mul_ps(x_lerp, _mm_sub_ps(tr0, tl0)));
    __m128 bot0 = _mm_add_ps(bl0, _mm_mul_ps(x_lerp, _mm_sub_ps(br0, bl0)));
    x_lerp = mmxs_lerp[1];
    __m128 top1 = _mm_add_ps(tl1, _mm_mul_ps(x_lerp, _mm_sub_ps(tr1, tl1)));
    __m128 bot1 = _mm_add_ps(bl1, _mm_mul_ps(x_lerp, _mm_sub_ps(br1, bl1)));
    x_lerp = mmxs_lerp[2];
    __m128 top2 = _mm_add_ps(tl2, _mm_mul_ps(x_lerp, _mm_sub_ps(tr2, tl2)));
    __m128 bot2 = _mm_add_ps(bl2, _mm_mul_ps(x_lerp, _mm_sub_ps(br2, bl2)));
    x_lerp = mmxs_lerp[3];
    __m128 top3 = _mm_add_ps(tl3, _mm_mul_ps(x_lerp, _mm_sub_ps(tr3, tl3)));
    __m128 bot3 = _mm_add_ps(bl3, _mm_mul_ps(x_lerp, _mm_sub_ps(br3, bl3)));
#endif
#ifdef __AVX2__
    __m128 res[4];
    res[0] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot0, top0), top0);
    res[1] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot1, top1), top1);
    res[2] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot2, top2), top2);
    res[3] = _mm_fmadd_ps(y_lerp, _mm_sub_ps(bot3, top3), top3);
    this->write_4ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#else
    __m128 res[4];
    res[0] = _mm_add_ps(top0, _mm_mul_ps(y_lerp, _mm_sub_ps(bot0, top0)));
    res[1] = _mm_add_ps(top1, _mm_mul_ps(y_lerp, _mm_sub_ps(bot1, top1)));
    res[2] = _mm_add_ps(top2, _mm_mul_ps(y_lerp, _mm_sub_ps(bot2, top2)));
    res[3] = _mm_add_ps(top3, _mm_mul_ps(y_lerp, _mm_sub_ps(bot3, top3)));
    this->write_4ch(ysA_output_ptr + load8_x_[current] * CHANNELS, res);
#endif
  }
}
#undef CHANNELS

template <class T, class U>
void CropResizeCastImage<T, U>::Configure_() {
  // num_cases[0] = general case
  // num_cases[1] = load4from1
  // num_cases[2] = load4from2
  // num_cases[3] = load4from4
  // num_cases[4] = load4from8
  int num_cases[5];
  for (int i = 0; i < 5; ++i) num_cases[i] = 0;
  for (int x = x0_; x <= x1_; ++x) {
    int load_group = this->DetermineLoadGroup_(x);
    assert(load_group >= 0 && load_group <= 4);
    ++num_cases[load_group];
    // load_group == 0 -> general case, pixel by pixel
    // every other value indidcates 1+3 = 4 pixels were processed this iteration
    if (load_group > 0) x += 3;
  }
  num_general_ = num_cases[0];
  num_load1_ = num_cases[1];
  num_load2_ = num_cases[2];
  num_load4_ = num_cases[3];
  num_load8_ = num_cases[4];
  if (num_general_ > 0) {
    general_x_ = new int[num_general_];
  } else {
    general_x_ = NULL;
  }
  if (num_load1_ > 0) {
    load1_offsets_ = new int[num_load1_];
    load1_shuffle_masks_ = new __m128i[num_load1_ * channels_ * 3];
    load1_mmxs_lerp_ = NULL;  // new __m128[num_load1_*channels_];
    load1_x_ = new int[num_load1_];
  } else {
    load1_offsets_ = NULL;
    load1_shuffle_masks_ = NULL;
    load1_mmxs_lerp_ = NULL;
    load1_x_ = NULL;
  }
  if (num_load2_ > 0) {
    load2_offsets_ = new int[num_load2_];
    load2_shuffle_masks_ = new __m128i[num_load2_ * channels_ * 2];
    load2_mmxs_lerp_ = NULL;  // new __m128[num_load2_*channels_];
    load2_x_ = new int[num_load2_];
  } else {
    load2_offsets_ = NULL;
    load2_shuffle_masks_ = NULL;
    load2_mmxs_lerp_ = NULL;
    load2_x_ = NULL;
  }
  if (num_load4_ > 0) {
    load4_offsets_ = new int[num_load4_ * 4];
    load4_mmxs_lerp_ = new __m128[num_load4_ * channels_];
    load4_x_ = new int[num_load4_];
  } else {
    load4_offsets_ = NULL;
    load4_mmxs_lerp_ = NULL;
    load4_x_ = NULL;
  }
  if (num_load8_ > 0) {
    load8_offsets_ = new int[num_load8_ * 4];
    load8_mmxs_lerp_ = new __m128[num_load8_ * channels_];
    load8_x_ = new int[num_load8_];
  } else {
    load8_offsets_ = NULL;
    load8_mmxs_lerp_ = NULL;
    load8_x_ = NULL;
  }
  for (int i = 0; i < 5; ++i) num_cases[i] = 0;
  if (verbose_) {
    printf("    load4from1  = %d\n", num_load1_);
    printf("    load4from2  = %d\n", num_load2_);
    printf("    load4from4  = %d\n", num_load4_);
    printf("    load4from8  = %d\n", num_load8_);
    printf("    general     = %d\n", num_general_);
  }
  for (int x = x0_; x <= x1_; ++x) {
    int load_group = DetermineLoadGroup_(x);
    assert(load_group >= 0 && load_group <= 4);
    int current = num_cases[load_group];
    assert(current >= 0);
    if (load_group == 0) {
      // general case
      assert(current < num_general_);
      general_x_[current] = x;
    } else if (load_group == 1) {
      // load4from1
      assert(current < num_load1_);
      load1_x_[current] = x;
      int min_xidx, max_xidx;
      ComputeXIndexRange_(x, &min_xidx, &max_xidx);
      load1_offsets_[current] = min_xidx * channels_;
      float* xs_lerp = (float*)(load1_shuffle_masks_ + current * channels_ * 3);
      char* shufmasks1 =
          (char*)(load1_shuffle_masks_ + current * channels_ * 3 + channels_);
      char* shufmasks2 = shufmasks1 + 16 * channels_;
      for (int j = 0; j < 32 * channels_; ++j) shufmasks1[j] = -128;
      for (int pix = 0; pix < 4; ++pix) {
        const int ix = flip_x_ ? out_width_ - 1 - min_ix_ - (x + pix)
                               : (x + pix) - min_ix_;
        float lerp = xs_[ix].lerp;
        int widx0 = xs_[ix].lower -
                    load1_offsets_[current];  // word index within SSE vector
        for (int ch = 0; ch < channels_; ++ch) {
          int idx = pix * channels_ + ch;
          xs_lerp[idx] = lerp;
          int shufvec = idx / 4;
          int shufidx = idx % 4;
          int widx = widx0 + ch;
          for (int b = 0; b < sizeof(T); ++b) {
            shufmasks1[shufvec * 16 + shufidx * sizeof(T) + b] =
                widx * sizeof(T) + b;
            shufmasks2[shufvec * 16 + shufidx * sizeof(T) + b] =
                (widx + channels_) * sizeof(T) + b;
          }
        }
      }
    } else if (load_group == 2) {
      // load4from2
      assert(current < num_load2_);
      load2_x_[current] = x;
      int min_xidx, max_xidx;
      ComputeXIndexRange_(x, &min_xidx, &max_xidx);
      load2_offsets_[current] = min_xidx * channels_;
      float* xs_lerp = (float*)(load2_shuffle_masks_ + current * channels_ * 2);
      char* shufmasks1 =
          (char*)(load2_shuffle_masks_ + current * channels_ * 2 + channels_);
      for (int j = 0; j < 16 * channels_; ++j) shufmasks1[j] = -128;
      for (int pix = 0; pix < 4; ++pix) {
        const int ix = flip_x_ ? out_width_ - 1 - min_ix_ - (x + pix)
                               : (x + pix) - min_ix_;
        float lerp = xs_[ix].lerp;
        int widx0 = xs_[ix].lower -
                    load2_offsets_[current];  // word index within SSE vector
        for (int ch = 0; ch < channels_; ++ch) {
          int idx = pix * channels_ + ch;
          xs_lerp[idx] = lerp;
          int shufvec = idx / 4;
          int shufidx = idx % 4;
          int widx = widx0 + ch;
          for (int b = 0; b < sizeof(T); ++b) {
            shufmasks1[shufvec * 16 + shufidx * sizeof(T) + b] =
                widx * sizeof(T) + b;
          }
        }
      }
    } else if (load_group == 3) {
      // load4from4
      assert(current < num_load4_);
      load4_x_[current] = x;
      int* index = load4_offsets_ + current * 4;
      float* xs_lerp = (float*)(load4_mmxs_lerp_ + current * channels_);
      for (int pix = 0; pix < 4; ++pix) {
        const int ix = flip_x_ ? out_width_ - 1 - min_ix_ - (x + pix)
                               : (x + pix) - min_ix_;
        float lerp = xs_[ix].lerp;
        index[pix] = xs_[ix].lower;
        for (int ch = 0; ch < channels_; ++ch) {
          int idx = pix * channels_ + ch;
          xs_lerp[idx] = lerp;
        }
      }
    } else if (load_group == 4) {
      // load4from8
      assert(current < num_load8_);
      load8_x_[current] = x;
      int* index = load8_offsets_ + current * 4;
      float* xs_lerp = (float*)(load8_mmxs_lerp_ + current * channels_);
      for (int pix = 0; pix < 4; ++pix) {
        const int ix = flip_x_ ? out_width_ - 1 - min_ix_ - (x + pix)
                               : (x + pix) - min_ix_;
        float lerp = xs_[ix].lerp;
        index[pix] = xs_[ix].lower;
        for (int ch = 0; ch < channels_; ++ch) {
          int idx = pix * channels_ + ch;
          xs_lerp[idx] = lerp;
        }
      }
    } else {
      assert(false);
    }
    ++num_cases[load_group];
    // load_group == 0 -> general case, pixel by pixel
    // every other value indidcates 1+3 = 4 pixels were processed this iteration
    if (load_group > 0) x += 3;
  }
}

template <class T, class U>
int CropResizeCastImage<T, U>::DetermineLoadGroup_(const int x) {
  int num_remaining = x1_ - x + 1;
  if (num_remaining >= 4) {
    // at least 4 values left, so theoretically possible to do SSE
    int min_xidx, max_xidx;
    // Using this-> is necessary in order to avoid compile error:
    // "there are no arguments to ‘xxx’ that depend on a template parameter, so
    // a declaration of ‘xxx’ must be available"
    // This is an issue for all member functions that have only builtin type
    // arguments and happens because
    // argument dependent lookup is not done for these arguments (so I've been
    // told).
    if (this->ComputeXIndexRange_(x, &min_xidx, &max_xidx)) {
      if ((allowed_load_groups_ & 1) && this->Load1_ok_(min_xidx, max_xidx)) {
        return 1;
      } else if ((allowed_load_groups_ & 2) &&
                 this->Load2_ok_(min_xidx, max_xidx)) {
        return 2;
      } else if ((allowed_load_groups_ & 4) &&
                 this->Load4_ok_(min_xidx, max_xidx)) {
        return 3;
      } else if ((allowed_load_groups_ & 8) &&
                 this->Load8_ok_(min_xidx, max_xidx)) {
        return 4;
      } else {
        return 0;
      }
    } else {
      // assumption xs[i].lower + channels == xs[i].upper NOT true for this
      // quintuple.
      return 0;
    }
  } else {
    // too few remaining values
    return 0;
  }
}

// Compute range of x indexes for xs[0] through xs[3].
// Returns true if valid (xs[i].lower + channels == xs[i].upper for all pixels).
template <class T, class U>
bool CropResizeCastImage<T, U>::ComputeXIndexRange_(const int x, int* min_xidx,
                                                    int* max_xidx) {
  bool upper_is_lower_plus_one = true;
  *min_xidx = 0;
  *max_xidx = -1;
  for (int pix = 0; pix < 4; ++pix) {
    const int ix =
        flip_x_ ? out_width_ - 1 - min_ix_ - (x + pix) : (x + pix) - min_ix_;
    int curr_xidx = xs_[ix].lower;
    if (curr_xidx + channels_ == xs_[ix].upper) {
      if (pix == 0) {
        *min_xidx = curr_xidx;
        *max_xidx = curr_xidx;
      } else {
        if (curr_xidx < *min_xidx) *min_xidx = curr_xidx;
        if (curr_xidx > *max_xidx) *max_xidx = curr_xidx;
      }
    } else {
      upper_is_lower_plus_one = false;
    }
  }
  *min_xidx /= channels_;
  *max_xidx /= channels_;
  return upper_is_lower_plus_one;
}

// This method returns true if it is possible to do load4from1
// for the load group pointed to by xs.
template <class T, class U>
bool CropResizeCastImage<T, U>::Load1_ok_(const int min_xidx,
                                          const int max_xidx) {
  // num_pixels_to_load_left_input = max_xs_low - min_xs_low + 1
  // num_pixels_to_load_left_and_right_input = num_pixels_to_load_left_input + 1
  int total_load_bytes = (max_xidx - min_xidx + 2) * channels_ * sizeof(T);
  if (total_load_bytes <= 16) {
    // a single (mis-aligned) SSE word gives us all the inputs
    // ensure that SSE word can be loaded without causing SEGV
    int load_offset = min_xidx * channels_;
    int load_offset_bytes = load_offset * sizeof(T);
    if (in_row_size_bytes_ - load_offset_bytes >= 16) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

// This method returns true if it is possible to do load4from2
// for the load group pointed to by xs.
template <class T, class U>
bool CropResizeCastImage<T, U>::Load2_ok_(const int min_xidx,
                                          const int max_xidx) {
  // num_pixels_to_load_left_input = max_xs_low - min_xs_low + 1
  int total_load_bytes = (max_xidx - min_xidx + 1) * channels_ * sizeof(T);
  if (total_load_bytes <= 16) {
    // a single (mis-aligned) SSE word gives us all the inputs
    // ensure that SSE word can be loaded without causing SEGV
    int load_offset = (min_xidx + 1) * channels_;
    int load_offset_bytes = load_offset * sizeof(T);
    if (in_row_size_bytes_ - load_offset_bytes >= 16) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

// This method returns true if it is possible to do load4from4
// for the load group pointed to by xs.
template <class T, class U>
bool CropResizeCastImage<T, U>::Load4_ok_(const int min_xidx,
                                          const int max_xidx) {
  int total_load_bytes = 2 * channels_ * sizeof(T);
  if (total_load_bytes <= 16) {
    // ensure that SSE word can be loaded without causing SEGV
    int load_offset = max_xidx * channels_;
    int load_offset_bytes = load_offset * sizeof(T);
    if (in_row_size_bytes_ - load_offset_bytes >= 16) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

// This method returns true if it is possible to do load4from8
// for the load group pointed to by xs.
template <class T, class U>
bool CropResizeCastImage<T, U>::Load8_ok_(const int min_xidx,
                                          const int max_xidx) {
  int total_load_bytes = channels_ * sizeof(T);
  if (total_load_bytes <= 16) {
    // ensure that SSE word can be loaded without causing SEGV
    int load_offset = (max_xidx + 1) * channels_;
    int load_offset_bytes = load_offset * sizeof(T);
    if (in_row_size_bytes_ - load_offset_bytes >= 16) {
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

//
// full implementations of templated static member function clip_necessary()
//

template <>
bool CropResizeCastImage<uint8, uint8>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint8, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<uint8, uint16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint8, int16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint8, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint8, Eigen::half>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint8, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<int8, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int8, int8>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int8, uint16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int8, int16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int8, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int8, Eigen::half>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int8, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<uint16, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<uint16, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<uint16, uint16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint16, int16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<uint16, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<uint16, Eigen::half>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<uint16, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<int16, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int16, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int16, uint16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int16, int16>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int16, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int16, Eigen::half>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int16, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<int32, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int32, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int32, uint16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int32, int16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int32, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<int32, Eigen::half>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<int32, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<Eigen::half, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<Eigen::half, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<Eigen::half, uint16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<Eigen::half, int16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<Eigen::half, int32>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<Eigen::half, Eigen::half>::clip_necessary() {
  return false;
}
template <>
bool CropResizeCastImage<Eigen::half, float>::clip_necessary() {
  return false;
}

template <>
bool CropResizeCastImage<float, uint8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, int8>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, uint16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, int16>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, int32>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, Eigen::half>::clip_necessary() {
  return true;
}
template <>
bool CropResizeCastImage<float, float>::clip_necessary() {
  return false;
}

// full specializations of crop_resize_single_image_common for data types that
// have vectorized implementations.
// at the moment, this is uint8, int8, uint16, int16, int32, Eigen::half,
// bfloat16 and float.

#define CROP_RESIZE_SINGLE_IMAGE_VECT(T_type, U_type)                          \
  template <>                                                                  \
  void crop_resize_single_image_common<T_type, U_type>(                        \
      const T_type* image, const int64 in_height, const int64 in_width,        \
      const int64 out_height, const int64 out_width, const int channels,       \
      const int min_ix, const int max_ix, const CachedInterpolation* xs,       \
      const int min_iy, const int max_iy, const CachedInterpolation* ys,       \
      const float extrapolated_value, const bool flip_x, const bool flip_y,    \
      U_type* output) {                                                        \
    if (channels <= 4) {                                                       \
      CropResizeCastImage<T_type, U_type>* resizer =                           \
          new CropResizeCastImage<T_type, U_type>(                             \
              in_height, in_width, out_height, out_width, channels, min_ix,    \
              max_ix, xs, min_iy, max_iy, ys, extrapolated_value, flip_x,      \
              flip_y, false, 15);                                              \
      resizer->Resize(image, output);                                          \
      delete resizer;                                                          \
    } else {                                                                   \
      crop_resize_single_image(image, in_height, in_width, out_height,         \
                               out_width, channels, min_ix, max_ix, xs,        \
                               min_iy, max_iy, ys, extrapolated_value, flip_x, \
                               flip_y, output);                                \
    }                                                                          \
  }

CROP_RESIZE_SINGLE_IMAGE_VECT(uint8, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(int8, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(uint16, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(int16, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(int32, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(Eigen::half, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(bfloat16, float)
CROP_RESIZE_SINGLE_IMAGE_VECT(float, float)

// full specializations of crop_resize_single_image_common for data types that
// don't have vectorized implementations.
// image resizing for these data types default to the original code.
// at the moment, this is int64 and double.

#define CROP_RESIZE_SINGLE_IMAGE_REGULAR(T_type, U_type)                      \
  template <>                                                                 \
  void crop_resize_single_image_common<T_type, U_type>(                       \
      const T_type* image, const int64 in_height, const int64 in_width,       \
      const int64 out_height, const int64 out_width, const int channels,      \
      const int min_ix, const int max_ix, const CachedInterpolation* xs,      \
      const int min_iy, const int max_iy, const CachedInterpolation* ys,      \
      const float extrapolated_value, const bool flip_x, const bool flip_y,   \
      U_type* output) {                                                       \
    crop_resize_single_image(image, in_height, in_width, out_height,          \
                             out_width, channels, min_ix, max_ix, xs, min_iy, \
                             max_iy, ys, extrapolated_value, flip_x, flip_y,  \
                             output);                                         \
  }

CROP_RESIZE_SINGLE_IMAGE_REGULAR(int64, float)
CROP_RESIZE_SINGLE_IMAGE_REGULAR(double, float)

#else

// compile fall-back code if either
// a) target is not a linux machine
// b) target architecture does not support at least SSE4.1

template <class T, class U>
void crop_resize_single_image_common(
    const T* image, const int64 in_height, const int64 in_width,
    const int64 out_height, const int64 out_width, const int channels,
    const int min_ix, const int max_ix, const CachedInterpolation* xs,
    const int min_iy, const int max_iy, const CachedInterpolation* ys,
    const float extrapolated_value, const bool flip_x, const bool flip_y,
    U* output) {
  crop_resize_single_image(image, in_height, in_width, out_height, out_width,
                           channels, min_ix, max_ix, xs, min_iy, max_iy, ys,
                           extrapolated_value, flip_x, flip_y, output);
}

#endif

}  // namespace
}  // namespace tensorflow
#endif  // define TENSORFLOW_CORE_KERNELS_CROP_RESIZE_BILINEAR_CORE_H_
