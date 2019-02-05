/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
#ifndef TENSORFLOW_CORE_KERNELS_ADJUST_HSV_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_ADJUST_HSV_GPU_CU_H_

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace internal {

typedef struct RgbTuple {
  float r;
  float g;
  float b;
} RgbTuple;

typedef struct HsvTuple {
  float h;
  float s;
  float v;
} HsvTuple;

inline __device__ HsvTuple rgb2hsv_cuda(const float r, const float g,
                                        const float b) {
  HsvTuple tuple;
  const float M = fmaxf(r, fmaxf(g, b));
  const float m = fminf(r, fminf(g, b));
  const float chroma = M - m;
  float h = 0.0f, s = 0.0f;
  // hue
  if (chroma > 0.0f) {
    if (M == r) {
      const float num = (g - b) / chroma;
      const float sign = copysignf(1.0f, num);
      h = ((sign < 0.0f) * 6.0f + sign * fmodf(sign * num, 6.0f)) / 6.0f;
    } else if (M == g) {
      h = ((b - r) / chroma + 2.0f) / 6.0f;
    } else {
      h = ((r - g) / chroma + 4.0f) / 6.0f;
    }
  } else {
    h = 0.0f;
  }
  // saturation
  if (M > 0.0) {
    s = chroma / M;
  } else {
    s = 0.0f;
  }
  tuple.h = h;
  tuple.s = s;
  tuple.v = M;
  return tuple;
}

inline __device__ RgbTuple hsv2rgb_cuda(const float h, const float s,
                                        const float v) {
  RgbTuple tuple;
  const float new_h = h * 6.0f;
  const float chroma = v * s;
  const float x = chroma * (1.0f - fabsf(fmodf(new_h, 2.0f) - 1.0f));
  const float new_m = v - chroma;
  const bool between_0_and_1 = new_h >= 0.0f && new_h < 1.0f;
  const bool between_1_and_2 = new_h >= 1.0f && new_h < 2.0f;
  const bool between_2_and_3 = new_h >= 2.0f && new_h < 3.0f;
  const bool between_3_and_4 = new_h >= 3.0f && new_h < 4.0f;
  const bool between_4_and_5 = new_h >= 4.0f && new_h < 5.0f;
  const bool between_5_and_6 = new_h >= 5.0f && new_h < 6.0f;
  tuple.r = chroma * (between_0_and_1 || between_5_and_6) +
            x * (between_1_and_2 || between_4_and_5) + new_m;
  tuple.g = chroma * (between_1_and_2 || between_2_and_3) +
            x * (between_0_and_1 || between_3_and_4) + new_m;
  tuple.b = chroma * (between_3_and_4 || between_4_and_5) +
            x * (between_2_and_3 || between_5_and_6) + new_m;
  return tuple;
}

template <bool AdjustHue, bool AdjustSaturation, bool AdjustV, typename T>
__global__ void adjust_hsv_nhwc(const int64 number_elements,
                                const T* const __restrict__ input,
                                T* const output, const float* const hue_delta,
                                const float* const saturation_scale,
                                const float* const value_scale) {
  // multiply by 3 since we're dealing with contiguous RGB bytes for each pixel
  // (NHWC)
  const int64 idx = (blockDim.x * blockIdx.x + threadIdx.x) * 3;
  // bounds check
  if (idx > number_elements - 1) {
    return;
  }
  if (!AdjustHue && !AdjustSaturation && !AdjustV) {
    output[idx] = input[idx];
    output[idx + 1] = input[idx + 1];
    output[idx + 2] = input[idx + 2];
    return;
  }
  const HsvTuple hsv = rgb2hsv_cuda(static_cast<float>(input[idx]),
                                    static_cast<float>(input[idx + 1]),
                                    static_cast<float>(input[idx + 2]));
  float new_h = hsv.h;
  float new_s = hsv.s;
  float new_v = hsv.v;
  // hue adjustment
  if (AdjustHue) {
    const float delta = *hue_delta;
    new_h = fmodf(hsv.h + delta, 1.0f);
    if (new_h < 0.0f) {
      new_h = fmodf(1.0f + new_h, 1.0f);
    }
  }
  // saturation adjustment
  if (AdjustSaturation && saturation_scale != nullptr) {
    const float scale = *saturation_scale;
    new_s = fminf(1.0f, fmaxf(0.0f, hsv.s * scale));
  }
  // value adjustment
  if (AdjustV && value_scale != nullptr) {
    const float scale = *value_scale;
    new_v = hsv.v * scale;
  }
  const RgbTuple rgb = hsv2rgb_cuda(new_h, new_s, new_v);
  output[idx] = static_cast<T>(rgb.r);
  output[idx + 1] = static_cast<T>(rgb.g);
  output[idx + 2] = static_cast<T>(rgb.b);
}

}  // namespace internal
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CORE_KERNELS_ADJUST_HSV_GPU_CU_H_
