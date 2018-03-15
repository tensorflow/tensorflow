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

// Implements a quantized version of the resize bilinear op.

#define EIGEN_USE_THREADS

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#define QUANTIZED_RESIZE_BILINEAR_USE_NEON
#include <arm_neon.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

static constexpr bool USE_REFERENCE = false;

namespace {
// Compute the interpolation indices only once.
template <typename T_SCALE>
struct InterpolationCache {
  std::vector<int64> lower;  // Lower source index used in the interpolation
  std::vector<int64> upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  std::vector<float> lerp;
  std::vector<T_SCALE> ilerp;
};

template <typename T_SCALE>
inline void ComputeInterpolationWeights(
    const int64 out_size, const int64 in_size, const float scale,
    const int resolution, InterpolationCache<T_SCALE>* interpolation) {
  interpolation->lower.resize(out_size + 1);
  interpolation->upper.resize(out_size + 1);
  interpolation->lerp.resize(out_size + 1);
  interpolation->ilerp.resize(out_size + 1);

  interpolation->lower[out_size] = 0;
  interpolation->upper[out_size] = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation->lower[i] = static_cast<int64>(in);
    interpolation->upper[i] =
        std::min(interpolation->lower[i] + 1, in_size - 1);
    interpolation->lerp[i] = in - interpolation->lower[i];
    interpolation->ilerp[i] = static_cast<T_SCALE>(
        (in - interpolation->lower[i]) * (1 << resolution));
  }
}

template <typename T_SCALE>
inline InterpolationCache<T_SCALE> BuildLerpCache(const int64 out_size,
                                                  const int64 in_size,
                                                  const float scale,
                                                  const int index_step,
                                                  const int resolution) {
  InterpolationCache<T_SCALE> cache;
  // Compute the cached interpolation weights on the x and y dimensions.
  ComputeInterpolationWeights<T_SCALE>(out_size, in_size, scale, resolution,
                                       &cache);
  CHECK(index_step > 0);
  if (index_step > 1) {
    for (int i = 0; i < cache.lower.size(); ++i) {
      cache.lower[i] *= index_step;
      cache.upper[i] *= index_step;
    }
  }
  return cache;
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
template <typename T>
inline T ComputeLerpReference(const T in_top_left, const T in_top_right,
                              const T in_bottom_left, const T in_bottom_right,
                              const float x_lerp, const float y_lerp,
                              const float min, const float max) {
  const float top_left = QuantizedToFloat<T>(in_top_left, min, max);
  const float top_right = QuantizedToFloat<T>(in_top_right, min, max);
  const float bottom_left = QuantizedToFloat<T>(in_bottom_left, min, max);
  const float bottom_right = QuantizedToFloat<T>(in_bottom_right, min, max);
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  const float out = top + (bottom - top) * y_lerp;
  return FloatToQuantized<T>(out, min, max);
}

template <typename T, typename T_SCALE, typename T_CALC>
inline T_CALC MulOffset(T a, T b, T_SCALE c) {
  return (static_cast<T_CALC>(a) - static_cast<T_CALC>(b)) *
         static_cast<T_CALC>(c);
}

template <int RESOLUTION, typename T, typename T_SCALE, typename T_CALC>
inline T ComputeLerp(const T top_left, const T top_right, const T bottom_left,
                     const T bottom_right, const T_SCALE x_lerp,
                     const T_SCALE y_lerp) {
  constexpr T_CALC RESOLUTION_MULT = (1 << RESOLUTION);
  const T_CALC top = static_cast<T_CALC>(top_left) * RESOLUTION_MULT +
                     MulOffset<T, T_SCALE, T_CALC>(top_right, top_left, x_lerp);
  const T_CALC bottom =
      static_cast<T_CALC>(bottom_left) * RESOLUTION_MULT +
      MulOffset<T, T_SCALE, T_CALC>(bottom_right, bottom_left, x_lerp);
  const T_CALC out = top + (bottom - top) / RESOLUTION_MULT * y_lerp;
  return static_cast<T>(
      static_cast<int32>((out + RESOLUTION_MULT / 2) / RESOLUTION_MULT));
}

#ifdef QUANTIZED_RESIZE_BILINEAR_USE_NEON
inline uint8x8_t ToUint8x8(const quint8* v0, const quint8* v1, const quint8* v2,
                           const quint8* v3, const quint8* v4, const quint8* v5,
                           const quint8* v6, const quint8* v7) {
  static const uint8x8_t ZERO_8x8 = vmov_n_u8(0);
  uint8x8_t ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v0), ZERO_8x8, 0);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v1), ret, 1);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v2), ret, 2);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v3), ret, 3);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v4), ret, 4);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v5), ret, 5);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v6), ret, 6);
  ret = vld1_lane_u8(reinterpret_cast<const uint8*>(v7), ret, 7);
  return ret;
}

inline int16x8_t ToInt16x8(const int16* v0, const int16* v1, const int16* v2,
                           const int16* v3, const int16* v4, const int16* v5,
                           const int16* v6, const int16* v7) {
  static const int16x8_t ZERO_16x8 = vmovq_n_s16(0);
  int16x8_t ret = vld1q_lane_s16(v0, ZERO_16x8, 0);
  ret = vld1q_lane_s16(v1, ret, 1);
  ret = vld1q_lane_s16(v2, ret, 2);
  ret = vld1q_lane_s16(v3, ret, 3);
  ret = vld1q_lane_s16(v4, ret, 4);
  ret = vld1q_lane_s16(v5, ret, 5);
  ret = vld1q_lane_s16(v6, ret, 6);
  ret = vld1q_lane_s16(v7, ret, 7);
  return ret;
}

inline int32x2_t ToInt32x2(const qint32* v0, const qint32* v1) {
  static const int32x2_t ZERO_32x2 = vmov_n_s32(0);
  const int32x2_t ret0 =
      vld1_lane_s32(reinterpret_cast<const int32*>(v0), ZERO_32x2, 0);
  const int32x2_t ret1 =
      vld1_lane_s32(reinterpret_cast<const int32*>(v1), ret0, 1);
  return ret1;
}

template <int RESOLUTION, bool X_LERP_SAME>
inline int32x2_t ComputeLerpx2(
    const qint32* top_left0, const qint32* top_right0,
    const qint32* bottom_left0, const qint32* bottom_right0,
    const qint32* top_left1, const qint32* top_right1,
    const qint32* bottom_left1, const qint32* bottom_right1,
    const int32* x_lerp, const int32x2_t y_lerpsx) {
  const int32x2_t x_lerpsx =
      X_LERP_SAME ? vld1_dup_s32(reinterpret_cast<const int32*>(x_lerp))
                  : vld1_s32(reinterpret_cast<const int32*>(x_lerp));

  const int32x2_t top_leftsx = ToInt32x2(top_left0, top_left1);
  const int32x2_t top_rightsx = ToInt32x2(top_right0, top_right1);
  const int32x2_t bottom_leftsx = ToInt32x2(bottom_left0, bottom_left1);
  const int32x2_t bottom_rightsx = ToInt32x2(bottom_right0, bottom_right1);

  const int32x2_t retval =
      ComputeLerp32x2<RESOLUTION>(top_leftsx, top_rightsx, bottom_leftsx,
                                  bottom_rightsx, x_lerpsx, y_lerpsx);
  return retval;
}

template <int RESOLUTION>
inline uint8x8_t ComputeLerpx8(
    const quint8* tl0, const quint8* tr0, const quint8* bl0, const quint8* br0,
    const int16* xlp0, const quint8* tl1, const quint8* tr1, const quint8* bl1,
    const quint8* br1, const int16* xlp1, const quint8* tl2, const quint8* tr2,
    const quint8* bl2, const quint8* br2, const int16* xlp2, const quint8* tl3,
    const quint8* tr3, const quint8* bl3, const quint8* br3, const int16* xlp3,
    const quint8* tl4, const quint8* tr4, const quint8* bl4, const quint8* br4,
    const int16* xlp4, const quint8* tl5, const quint8* tr5, const quint8* bl5,
    const quint8* br5, const int16* xlp5, const quint8* tl6, const quint8* tr6,
    const quint8* bl6, const quint8* br6, const int16* xlp6, const quint8* tl7,
    const quint8* tr7, const quint8* bl7, const quint8* br7, const int16* xlp7,
    const int16x8_t ys_lerpsx) {
  const uint8x8_t tl8x8 = ToUint8x8(tl0, tl1, tl2, tl3, tl4, tl5, tl6, tl7);
  const uint8x8_t tr8x8 = ToUint8x8(tr0, tr1, tr2, tr3, tr4, tr5, tr6, tr7);
  const uint8x8_t bl8x8 = ToUint8x8(bl0, bl1, bl2, bl3, bl4, bl5, bl6, bl7);
  const uint8x8_t br8x8 = ToUint8x8(br0, br1, br2, br3, br4, br5, br6, br7);
  const int16x8_t xs_lerpsx =
      ToInt16x8(xlp0, xlp1, xlp2, xlp3, xlp4, xlp5, xlp6, xlp7);
  return ComputeLerp8x8<RESOLUTION>(tl8x8, tr8x8, bl8x8, br8x8, xs_lerpsx,
                                    ys_lerpsx);
}

// Expand address at compile time to improve performance
template <int RESOLUTION, int ID0, int CH0, int ID1, int CH1, int ID2, int CH2,
          int ID3, int CH3, int ID4, int CH4, int ID5, int CH5, int ID6,
          int CH6, int ID7, int CH7>
inline uint8x8_t ComputeLerpx8Tmpl(const quint8* const yl, const quint8* yu,
                                   const int64* xl, const int64* xu,
                                   const int16* xlp,
                                   const int16x8_t ys_lerpsx) {
  return ComputeLerpx8<RESOLUTION>(
      yl + xl[ID0] + CH0, yl + xu[ID0] + CH0, yu + xl[ID0] + CH0,
      yu + xu[ID0] + CH0, xlp + ID0, yl + xl[ID1] + CH1, yl + xu[ID1] + CH1,
      yu + xl[ID1] + CH1, yu + xu[ID1] + CH1, xlp + ID1, yl + xl[ID2] + CH2,
      yl + xu[ID2] + CH2, yu + xl[ID2] + CH2, yu + xu[ID2] + CH2, xlp + ID2,
      yl + xl[ID3] + CH3, yl + xu[ID3] + CH3, yu + xl[ID3] + CH3,
      yu + xu[ID3] + CH3, xlp + ID3, yl + xl[ID4] + CH4, yl + xu[ID4] + CH4,
      yu + xl[ID4] + CH4, yu + xu[ID4] + CH4, xlp + ID4, yl + xl[ID5] + CH5,
      yl + xu[ID5] + CH5, yu + xl[ID5] + CH5, yu + xu[ID5] + CH5, xlp + ID5,
      yl + xl[ID6] + CH6, yl + xu[ID6] + CH6, yu + xl[ID6] + CH6,
      yu + xu[ID6] + CH6, xlp + ID6, yl + xl[ID7] + CH7, yl + xu[ID7] + CH7,
      yu + xl[ID7] + CH7, yu + xu[ID7] + CH7, xlp + ID7, ys_lerpsx);
}

#endif

template <int RESOLUTION, typename T, typename T_SCALE, typename T_CALC>
inline void OutputLerpForChannels(const InterpolationCache<T_SCALE>& xs,
                                  const int64 x, const T_SCALE ys_ilerp,
                                  const int channels, const float min,
                                  const float max, const T* ys_input_lower_ptr,
                                  const T* ys_input_upper_ptr,
                                  T* output_y_ptr) {
  const int64 xs_lower = xs.lower[x];
  const int64 xs_upper = xs.upper[x];
  const T_SCALE xs_ilerp = xs.ilerp[x];
  for (int c = 0; c < channels; ++c) {
    const T top_left = ys_input_lower_ptr[xs_lower + c];
    const T top_right = ys_input_lower_ptr[xs_upper + c];
    const T bottom_left = ys_input_upper_ptr[xs_lower + c];
    const T bottom_right = ys_input_upper_ptr[xs_upper + c];
    const T val = ComputeLerp<RESOLUTION, T, T_SCALE, T_CALC>(
        top_left, top_right, bottom_left, bottom_right, xs_ilerp, ys_ilerp);
    output_y_ptr[x * channels + c] = val;
  }
}

template <int RES>
inline void OutputLerp8x8x1(const InterpolationCache<int16>& xs,
                            const int64 x_start, const int16 ys_ilerp,
                            const float min, const float max,
                            const quint8* const ys_input_lower_ptr,
                            const quint8* const ys_input_upper_ptr,
                            quint8* output_y_ptr) {
#ifdef QUANTIZED_RESIZE_BILINEAR_USE_NEON
  const int16x8_t y_lerpsx = vmovq_n_s16(ys_ilerp);

  const uint8x8_t x0x7 =
      ComputeLerpx8Tmpl<RES, 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0>(
          ys_input_lower_ptr, ys_input_upper_ptr, &xs.lower[x_start],
          &xs.upper[x_start], &xs.ilerp[x_start], y_lerpsx);

  vst1_u8(reinterpret_cast<uint8_t*>(output_y_ptr + x_start), x0x7);

#else
  for (int x = x_start; x < x_start + 8; ++x) {
    OutputLerpForChannels<RES, quint8, int16, int16>(
        xs, x, ys_ilerp, 1, min, max, ys_input_lower_ptr, ys_input_upper_ptr,
        output_y_ptr);
  }
#endif
}

template <int RES>
inline void OutputLerp8x8x3(const InterpolationCache<int16>& xs,
                            const int64 x_start, const int16 ys_ilerp,
                            const float min, const float max,
                            const quint8* const ys_input_lower_ptr,
                            const quint8* const ys_input_upper_ptr,
                            quint8* output_y_ptr) {
#ifdef QUANTIZED_RESIZE_BILINEAR_USE_NEON
  const int16x8_t y_lerpsx = vmovq_n_s16(ys_ilerp);

  const uint8x8_t x0c0x2c1 =
      ComputeLerpx8Tmpl<RES, 0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 2, 2, 0, 2, 1>(
          ys_input_lower_ptr, ys_input_upper_ptr, &xs.lower[x_start],
          &xs.upper[x_start], &xs.ilerp[x_start], y_lerpsx);

  vst1_u8(reinterpret_cast<uint8_t*>(output_y_ptr + x_start * 3), x0c0x2c1);

  const uint8x8_t x2c2x5c0 =
      ComputeLerpx8Tmpl<RES, 2, 2, 3, 0, 3, 1, 3, 2, 4, 0, 4, 1, 4, 2, 5, 0>(
          ys_input_lower_ptr, ys_input_upper_ptr, &xs.lower[x_start],
          &xs.upper[x_start], &xs.ilerp[x_start], y_lerpsx);

  vst1_u8(reinterpret_cast<uint8_t*>(output_y_ptr + x_start * 3 + 8), x2c2x5c0);

  const uint8x8_t x5c1x7c2 =
      ComputeLerpx8Tmpl<RES, 5, 1, 5, 2, 6, 0, 6, 1, 6, 2, 7, 0, 7, 1, 7, 2>(
          ys_input_lower_ptr, ys_input_upper_ptr, &xs.lower[x_start],
          &xs.upper[x_start], &xs.ilerp[x_start], y_lerpsx);

  vst1_u8(reinterpret_cast<uint8_t*>(output_y_ptr + x_start * 3 + 16),
          x5c1x7c2);

#else
  for (int x = x_start; x < x_start + 8; ++x) {
    OutputLerpForChannels<RES, quint8, int16, int16>(
        xs, x, ys_ilerp, 3, min, max, ys_input_lower_ptr, ys_input_upper_ptr,
        output_y_ptr);
  }
#endif
}

template <int RESOLUTION>
inline void OutputLerp32x4x1(const InterpolationCache<int32>& xs,
                             const int64 x_start, const int32 ys_ilerp,
                             const float min, const float max,
                             const qint32* const ys_input_lower_ptr,
                             const qint32* const ys_input_upper_ptr,
                             qint32* output_y_ptr) {
#ifdef QUANTIZED_RESIZE_BILINEAR_USE_NEON
  const int64 xs_lower0 = xs.lower[x_start];
  const int64 xs_upper0 = xs.upper[x_start];
  const int32* const xs_ilerp0 = &xs.ilerp[x_start];
  const int64 xs_lower1 = xs.lower[x_start + 1];
  const int64 xs_upper1 = xs.upper[x_start + 1];
  const int64 xs_lower2 = xs.lower[x_start + 2];
  const int64 xs_upper2 = xs.upper[x_start + 2];
  const int32* const xs_ilerp2 = &xs.ilerp[x_start + 2];
  const int64 xs_lower3 = xs.lower[x_start + 3];
  const int64 xs_upper3 = xs.upper[x_start + 3];

  const int32x2_t y_lerpsx = vmov_n_s32(ys_ilerp);

  const int32x2_t x0x1 = ComputeLerpx2<RESOLUTION, false>(
      ys_input_lower_ptr + xs_lower0, ys_input_lower_ptr + xs_upper0,
      ys_input_upper_ptr + xs_lower0, ys_input_upper_ptr + xs_upper0,
      ys_input_lower_ptr + xs_lower1, ys_input_lower_ptr + xs_upper1,
      ys_input_upper_ptr + xs_lower1, ys_input_upper_ptr + xs_upper1, xs_ilerp0,
      y_lerpsx);

  const int32x2_t x1x2 = ComputeLerpx2<RESOLUTION, false>(
      ys_input_lower_ptr + xs_lower2, ys_input_lower_ptr + xs_upper2,
      ys_input_upper_ptr + xs_lower2, ys_input_upper_ptr + xs_upper2,
      ys_input_lower_ptr + xs_lower3, ys_input_lower_ptr + xs_upper3,
      ys_input_upper_ptr + xs_lower3, ys_input_upper_ptr + xs_upper3, xs_ilerp2,
      y_lerpsx);

  const int32x4_t x0x1x2x3 = vcombine_s32(x0x1, x1x2);

  vst1q_s32(reinterpret_cast<int32*>(output_y_ptr + x_start), x0x1x2x3);

#else
  for (int x = x_start; x < x_start + 4; ++x) {
    OutputLerpForChannels<RESOLUTION, qint32, int32, int64>(
        xs, x, ys_ilerp, 1, min, max, ys_input_lower_ptr, ys_input_upper_ptr,
        output_y_ptr);
  }
#endif
}

template <int RESOLUTION>
inline void OutputLerp32x4x3(const InterpolationCache<int32>& xs,
                             const int64 x_start, const int32 ys_ilerp,
                             const float min, const float max,
                             const qint32* const ys_input_lower_ptr,
                             const qint32* const ys_input_upper_ptr,
                             qint32* output_y_ptr) {
#ifdef QUANTIZED_RESIZE_BILINEAR_USE_NEON
  const int64 xs_lower0 = xs.lower[x_start];
  const int64 xs_upper0 = xs.upper[x_start];
  const int32* const xs_ilerp0 = &xs.ilerp[x_start];
  const int64 xs_lower1 = xs.lower[x_start + 1];
  const int64 xs_upper1 = xs.upper[x_start + 1];
  const int32* const xs_ilerp1 = &xs.ilerp[x_start + 1];
  const int64 xs_lower2 = xs.lower[x_start + 2];
  const int64 xs_upper2 = xs.upper[x_start + 2];
  const int32* const xs_ilerp2 = &xs.ilerp[x_start + 2];
  const int64 xs_lower3 = xs.lower[x_start + 3];
  const int64 xs_upper3 = xs.upper[x_start + 3];
  const int32* const xs_ilerp3 = &xs.ilerp[x_start + 3];

  const int32x2_t y_lerpsx = vmov_n_s32(ys_ilerp);

  const int32x2_t x0c0x0c1 = ComputeLerpx2<RESOLUTION, true>(
      ys_input_lower_ptr + xs_lower0, ys_input_lower_ptr + xs_upper0,
      ys_input_upper_ptr + xs_lower0, ys_input_upper_ptr + xs_upper0,
      ys_input_lower_ptr + xs_lower0 + 1, ys_input_lower_ptr + xs_upper0 + 1,
      ys_input_upper_ptr + xs_lower0 + 1, ys_input_upper_ptr + xs_upper0 + 1,
      xs_ilerp0, y_lerpsx);

  const int32x2_t x0c2x1c0 = ComputeLerpx2<RESOLUTION, false>(
      ys_input_lower_ptr + xs_lower0 + 2, ys_input_lower_ptr + xs_upper0 + 2,
      ys_input_upper_ptr + xs_lower0 + 2, ys_input_upper_ptr + xs_upper0 + 2,
      ys_input_lower_ptr + xs_lower1, ys_input_lower_ptr + xs_upper1,
      ys_input_upper_ptr + xs_lower1, ys_input_upper_ptr + xs_upper1, xs_ilerp0,
      y_lerpsx);

  const int32x2_t x1c1x1c2 = ComputeLerpx2<RESOLUTION, true>(
      ys_input_lower_ptr + xs_lower1 + 1, ys_input_lower_ptr + xs_upper1 + 1,
      ys_input_upper_ptr + xs_lower1 + 1, ys_input_upper_ptr + xs_upper1 + 1,
      ys_input_lower_ptr + xs_lower1 + 2, ys_input_lower_ptr + xs_upper1 + 2,
      ys_input_upper_ptr + xs_lower1 + 2, ys_input_upper_ptr + xs_upper1 + 2,
      xs_ilerp1, y_lerpsx);

  const int32x2_t x2c0x2c1 = ComputeLerpx2<RESOLUTION, true>(
      ys_input_lower_ptr + xs_lower2, ys_input_lower_ptr + xs_upper2,
      ys_input_upper_ptr + xs_lower2, ys_input_upper_ptr + xs_upper2,
      ys_input_lower_ptr + xs_lower2 + 1, ys_input_lower_ptr + xs_upper2 + 1,
      ys_input_upper_ptr + xs_lower2 + 1, ys_input_upper_ptr + xs_upper2 + 1,
      xs_ilerp2, y_lerpsx);

  const int32x2_t x2c2x3c0 = ComputeLerpx2<RESOLUTION, false>(
      ys_input_lower_ptr + xs_lower2 + 2, ys_input_lower_ptr + xs_upper2 + 2,
      ys_input_upper_ptr + xs_lower2 + 2, ys_input_upper_ptr + xs_upper2 + 2,
      ys_input_lower_ptr + xs_lower3, ys_input_lower_ptr + xs_upper3,
      ys_input_upper_ptr + xs_lower3, ys_input_upper_ptr + xs_upper3, xs_ilerp2,
      y_lerpsx);

  const int32x2_t x3c1x3c2 = ComputeLerpx2<RESOLUTION, true>(
      ys_input_lower_ptr + xs_lower3 + 1, ys_input_lower_ptr + xs_upper3 + 1,
      ys_input_upper_ptr + xs_lower3 + 1, ys_input_upper_ptr + xs_upper3 + 1,
      ys_input_lower_ptr + xs_lower3 + 2, ys_input_lower_ptr + xs_upper3 + 2,
      ys_input_upper_ptr + xs_lower3 + 2, ys_input_upper_ptr + xs_upper3 + 2,
      xs_ilerp3, y_lerpsx);

  const int32x4_t x0c0x0c1x0c2x1c0 = vcombine_s32(x0c0x0c1, x0c2x1c0);
  const int32x4_t x1c1x1c2x2c0x2c1 = vcombine_s32(x1c1x1c2, x2c0x2c1);
  const int32x4_t x2c2x3c0x3c1x3c2 = vcombine_s32(x2c2x3c0, x3c1x3c2);

  vst1q_s32(reinterpret_cast<int32*>(output_y_ptr + x_start * 3),
            x0c0x0c1x0c2x1c0);
  vst1q_s32(reinterpret_cast<int32*>(output_y_ptr + x_start * 3 + 4),
            x1c1x1c2x2c0x2c1);
  vst1q_s32(reinterpret_cast<int32*>(output_y_ptr + x_start * 3 + 8),
            x2c2x3c0x3c1x3c2);

#else
  for (int x = x_start; x < x_start + 4; ++x) {
    OutputLerpForChannels<RESOLUTION, qint32, int32, int64>(
        xs, x, ys_ilerp, 3, min, max, ys_input_lower_ptr, ys_input_upper_ptr,
        output_y_ptr);
  }
#endif
}

template <typename T>
void ResizeImageReference(typename TTypes<T, 4>::ConstTensor images,
                          const int batch_size, const int64 in_height,
                          const int64 in_width, const int64 out_height,
                          const int64 out_width, const int channels,
                          const float height_scale, const float width_scale,
                          const float in_min, const float in_max,
                          typename TTypes<T, 4>::Tensor* output) {
  CHECK_NOTNULL(output);

  const InterpolationCache<float> xs =
      BuildLerpCache<float>(out_width, in_width, width_scale, channels, 0);
  const InterpolationCache<float> ys =
      BuildLerpCache<float>(out_height, in_height, height_scale, 1, 0);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const T* input_b_ptr = images.data();

  T* output_y_ptr = output->data();
  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const T* ys_input_lower_ptr = input_b_ptr + ys.lower[y] * in_row_size;
      const T* ys_input_upper_ptr = input_b_ptr + ys.upper[y] * in_row_size;
      const float ys_lerp = ys.lerp[y];
      for (int64 x = 0; x < out_width; ++x) {
        const int64 xs_lower = xs.lower[x];
        const int64 xs_upper = xs.upper[x];
        const float xs_lerp = xs.lerp[x];
        for (int c = 0; c < channels; ++c) {
          const T top_left = ys_input_lower_ptr[xs_lower + c];
          const T top_right = ys_input_lower_ptr[xs_upper + c];
          const T bottom_left = ys_input_upper_ptr[xs_lower + c];
          const T bottom_right = ys_input_upper_ptr[xs_upper + c];
          const T val = ComputeLerpReference<T>(
              top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp,
              in_min, in_max);
          output_y_ptr[x * channels + c] = val;
        }
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <typename T>
void ResizeImage(typename TTypes<T, 4>::ConstTensor images,
                 const int batch_size, const int64 in_height,
                 const int64 in_width, const int64 out_height,
                 const int64 out_width, const int channels,
                 const float height_scale, const float width_scale,
                 const float in_min, const float in_max,
                 typename TTypes<T, 4>::Tensor* output) {
  ResizeImageReference<T>(images, batch_size, in_height, in_width, out_height,
                          out_width, channels, height_scale, width_scale,
                          in_min, in_max, output);
}

template <>
void ResizeImage<qint32>(typename TTypes<qint32, 4>::ConstTensor images,
                         const int batch_size, const int64 in_height,
                         const int64 in_width, const int64 out_height,
                         const int64 out_width, const int channels,
                         const float height_scale, const float width_scale,
                         const float in_min, const float in_max,
                         typename TTypes<qint32, 4>::Tensor* output) {
  // 30 is maximum resolution for signed int.
  constexpr int RESOLUTION = 30;
  constexpr int SIMD_STEP = 4;

  CHECK_NOTNULL(output);

  const InterpolationCache<int32> xs = BuildLerpCache<int32>(
      out_width, in_width, width_scale, channels, RESOLUTION);
  const InterpolationCache<int32> ys =
      BuildLerpCache<int32>(out_height, in_height, height_scale, 1, RESOLUTION);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const qint32* input_b_ptr = images.data();

  qint32* output_y_ptr = output->data();

  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const qint32* ys_input_lower_ptr =
          input_b_ptr + ys.lower[y] * in_row_size;
      const qint32* ys_input_upper_ptr =
          input_b_ptr + ys.upper[y] * in_row_size;
      const int32 ys_ilerp = ys.ilerp[y];
      // Optimized for channels == 1 or channels == 3 as this
      // is typical channels.
      int64 x = 0;
      if (channels == 1) {
        for (; x < out_width - SIMD_STEP + 1; x += SIMD_STEP) {
          OutputLerp32x4x1<RESOLUTION>(xs, x, ys_ilerp, in_min, in_max,
                                       ys_input_lower_ptr, ys_input_upper_ptr,
                                       output_y_ptr);
        }
      } else if (channels == 3) {
        for (; x < out_width - SIMD_STEP + 1; x += SIMD_STEP) {
          OutputLerp32x4x3<RESOLUTION>(xs, x, ys_ilerp, in_min, in_max,
                                       ys_input_lower_ptr, ys_input_upper_ptr,
                                       output_y_ptr);
        }
      }
      for (; x < out_width; ++x) {
        OutputLerpForChannels<RESOLUTION, qint32, int32, int64>(
            xs, x, ys_ilerp, channels, in_min, in_max, ys_input_lower_ptr,
            ys_input_upper_ptr, output_y_ptr);
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <>
void ResizeImage<quint8>(typename TTypes<quint8, 4>::ConstTensor images,
                         const int batch_size, const int64 in_height,
                         const int64 in_width, const int64 out_height,
                         const int64 out_width, const int channels,
                         const float height_scale, const float width_scale,
                         const float in_min, const float in_max,
                         typename TTypes<quint8, 4>::Tensor* output) {
  // 7 is maximum resolution for unsigned byte.
  constexpr int RESOLUTION = 7;
  constexpr int SIMD_STEP = 8;

  CHECK_NOTNULL(output);

  const InterpolationCache<int16> xs = BuildLerpCache<int16>(
      out_width, in_width, width_scale, channels, RESOLUTION);
  const InterpolationCache<int16> ys =
      BuildLerpCache<int16>(out_height, in_height, height_scale, 1, RESOLUTION);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const quint8* input_b_ptr = images.data();

  quint8* output_y_ptr = output->data();

  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const quint8* ys_input_lower_ptr =
          input_b_ptr + ys.lower[y] * in_row_size;
      const quint8* ys_input_upper_ptr =
          input_b_ptr + ys.upper[y] * in_row_size;
      const int32 ys_ilerp = ys.ilerp[y];
      // Optimized for channels == 1 or channels == 3 as this
      // is typical channels.
      // TODO(satok): Support more generic NEON optimized implementation
      // for different channels.
      int64 x = 0;
      if (channels == 1) {
        for (; x < out_width - SIMD_STEP + 1; x += SIMD_STEP) {
          OutputLerp8x8x1<RESOLUTION>(xs, x, ys_ilerp, in_min, in_max,
                                      ys_input_lower_ptr, ys_input_upper_ptr,
                                      output_y_ptr);
        }
      } else if (channels == 3) {
        for (; x < out_width - SIMD_STEP + 1; x += SIMD_STEP) {
          OutputLerp8x8x3<RESOLUTION>(xs, x, ys_ilerp, in_min, in_max,
                                      ys_input_lower_ptr, ys_input_upper_ptr,
                                      output_y_ptr);
        }
      }
      for (; x < out_width; ++x) {
        OutputLerpForChannels<RESOLUTION, quint8, int16, int16>(
            xs, x, ys_ilerp, channels, in_min, in_max, ys_input_lower_ptr,
            ys_input_upper_ptr, output_y_ptr);
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <typename T>
void ResizeBilinear(const typename TTypes<T, 4>::ConstTensor& images,
                    const float height_scale, const float width_scale,
                    const float in_min, const float in_max,
                    typename TTypes<T, 4>::Tensor* output) {
  CHECK_NOTNULL(output);

  const int batch_size = images.dimension(0);
  const int64 in_height = images.dimension(1);
  const int64 in_width = images.dimension(2);
  const int channels = images.dimension(3);

  const int64 out_height = output->dimension(1);
  const int64 out_width = output->dimension(2);

  // Handle no-op resizes efficiently.
  if (out_height == in_height && out_width == in_width) {
    *output = images.template cast<T>();
    return;
  }

  if (USE_REFERENCE) {
    ResizeImageReference<T>(images, batch_size, in_height, in_width, out_height,
                            out_width, channels, height_scale, width_scale,
                            in_min, in_max, output);
  } else {
    ResizeImage<T>(images, batch_size, in_height, in_width, out_height,
                   out_width, channels, height_scale, width_scale, in_min,
                   in_max, output);
  }
}

}  // namespace

template <class T>
class QuantizedResizeBilinearOp : public OpKernel {
 public:
  explicit QuantizedResizeBilinearOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const float in_min = context->input(2).flat<float>()(0);
    const float in_max = context->input(3).flat<float>()(0);

    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data(input.tensor<T, 4>());
    typename TTypes<T, 4>::Tensor output_data(st.output->tensor<T, 4>());

    ResizeBilinear<T>(image_data, st.height_scale, st.width_scale, in_min,
                      in_max, &output_data);
    Tensor* out_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &out_min));
    out_min->flat<float>()(0) = in_min;

    Tensor* out_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &out_max));
    out_max->flat<float>()(0) = in_max;
  }

 private:
  bool align_corners_;

  TF_DISALLOW_COPY_AND_ASSIGN(QuantizedResizeBilinearOp<T>);
};

#define REGISTER_CPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("QuantizedResizeBilinear") \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          QuantizedResizeBilinearOp<type>)

REGISTER_CPU_KERNEL(::tensorflow::quint8);
REGISTER_CPU_KERNEL(::tensorflow::qint32);
REGISTER_CPU_KERNEL(float);

}  // namespace tensorflow
