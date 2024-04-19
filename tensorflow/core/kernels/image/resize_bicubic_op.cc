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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <math.h>

#include <algorithm>
#include <array>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/image_resizer_state.h"

namespace tensorflow {
namespace {

static const int64_t kTableSize = (1 << 10);

const float* InitCoeffsTable(const double a) {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  float* coeffs_table = new float[(kTableSize + 1) * 2];
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * 2] = ((a + 2) * x - (a + 3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * 2 + 1] = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
  }

  return coeffs_table;
}

const float* GetCoeffsTable(const bool use_keys_cubic) {
  // Static so that we initialize it on first use
  if (use_keys_cubic) {
    // http://ieeexplore.ieee.org/document/1163711/
    // R. G. Keys. Cubic convolution interpolation for digital image
    // processing. IEEE Transactions on Acoustics, Speech, and Signal
    // Processing, 29(6):1153–1160, 1981.
    static const float* coeffs_table = InitCoeffsTable(-0.5f);
    return coeffs_table;
  } else {
    static const float* coeffs_table = InitCoeffsTable(-0.75f);
    return coeffs_table;
  }
}

inline int64_t Bound(int64_t val, int64_t limit) {
  return std::min(limit - 1, std::max(int64_t{0}, val));
}

struct WeightsAndIndices {
  float weight_0;
  float weight_1;
  float weight_2;
  float weight_3;
  int64_t index_0;
  int64_t index_1;
  int64_t index_2;
  int64_t index_3;

  int advance;  // advance value.
};

template <typename Scaler, bool use_keys_cubic>
inline void GetWeightsAndIndices(const float scale, const int64_t out_loc,
                                 const int64_t limit, WeightsAndIndices* out) {
  const Scaler scaler;
  const float in_loc_f = scaler(out_loc, scale);
  const int64_t in_loc = std::floor(in_loc_f);
  const float delta = in_loc_f - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float* coeffs_table = GetCoeffsTable(use_keys_cubic);
  if (use_keys_cubic) {
    // The legacy code placed more weight on the edge pixels, since bounding
    // the set of inputs to sample could cause an edge pixel to be repeated.
    // Here we change the behavior at borders to match that used by the
    // scale_and_translate_op, where sampling locations outside the image have
    // their weight set to 0, and the weights are renormalized so that their sum
    // is 1.0.
    out->index_0 = Bound(in_loc - 1, limit);
    out->weight_0 =
        (out->index_0 == in_loc - 1 ? coeffs_table[offset * 2 + 1] : 0.0f);
    out->index_1 = Bound(in_loc, limit);
    out->weight_1 = (out->index_1 == in_loc ? coeffs_table[offset * 2] : 0.0f);
    out->index_2 = Bound(in_loc + 1, limit);
    out->weight_2 =
        (out->index_2 == in_loc + 1 ? coeffs_table[(kTableSize - offset) * 2]
                                    : 0.0f);
    out->index_3 = Bound(in_loc + 2, limit);
    out->weight_3 = (out->index_3 == in_loc + 2
                         ? coeffs_table[(kTableSize - offset) * 2 + 1]
                         : 0.0f);

    const float weight_sum =
        out->weight_0 + out->weight_1 + out->weight_2 + out->weight_3;
    if (std::abs(weight_sum) >= 1000.0f * std::numeric_limits<float>::min()) {
      const float one_over_weight_sum = 1.0f / weight_sum;
      out->weight_0 *= one_over_weight_sum;
      out->weight_1 *= one_over_weight_sum;
      out->weight_2 *= one_over_weight_sum;
      out->weight_3 *= one_over_weight_sum;
    }
  } else {
    out->weight_0 = coeffs_table[offset * 2 + 1];
    out->weight_1 = coeffs_table[offset * 2];
    out->weight_2 = coeffs_table[(kTableSize - offset) * 2];
    out->weight_3 = coeffs_table[(kTableSize - offset) * 2 + 1];
    out->index_0 = Bound(in_loc - 1, limit);
    out->index_1 = Bound(in_loc, limit);
    out->index_2 = Bound(in_loc + 1, limit);
    out->index_3 = Bound(in_loc + 2, limit);
  }
}

template <typename T>
inline float Interpolate1D(const float weight_0, const float weight_1,
                           const float weight_2, const float weight_3,
                           const T value_0, const T value_1, const T value_2,
                           const T value_3) {
  return static_cast<float>(value_0) * weight_0 +
         static_cast<float>(value_1) * weight_1 +
         static_cast<float>(value_2) * weight_2 +
         static_cast<float>(value_3) * weight_3;
}

// Compute the 1D interpolation for a given X index using the y_weights
static float Compute(float values_[4], const float xw_0, const float xw_1,
                     const float xw_2, const float xw_3) {
  return Interpolate1D(xw_0, xw_1, xw_2, xw_3, values_[0], values_[1],
                       values_[2], values_[3]);
}

// In order to compute a single output value, we look at a 4x4 patch in the
// source image. As we iterate increasing X across the image, the new 4x4 patch
// often overlaps with the previous 4x4 patch we just looked at.
//
// This class helps compute the number of values to copy from the previous
// point's values.
class CachedInterpolationCalculator {
 public:
  CachedInterpolationCalculator() : indexes_{-1, -1, -1, -1} {}

  // Advances iteration. Returns the number of values that should be copied from
  // the current point to the next point. The copying should always be done by
  // copying the last <retval> values from the old point to the first <retval>
  // values of the new point.
  inline int Advance(const int64_t x_0, const int64_t x_1, const int64_t x_2,
                     const int64_t x_3) {
    // We use 2 hands and walk through, copying from one to another where
    // we already have values.
    // Invariant, new_indices_hand <= cached_values_hand
    const std::array<int64_t, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    int cached_values_hand = 0;
    int new_indices_hand = 0;
    while (cached_values_hand < 4) {
      if (indexes_[cached_values_hand] == new_x_indices[new_indices_hand]) {
        if (new_indices_hand < cached_values_hand) {
          indexes_[new_indices_hand] = indexes_[cached_values_hand];
        }
        cached_values_hand++;
        new_indices_hand++;
      } else {
        cached_values_hand++;
      }
    }
    switch (new_indices_hand) {
      case 0:
        indexes_[0] = x_0;
        TF_FALLTHROUGH_INTENDED;
      case 1:
        indexes_[1] = x_1;
        TF_FALLTHROUGH_INTENDED;
      case 2:
        indexes_[2] = x_2;
        TF_FALLTHROUGH_INTENDED;
      case 3:
        indexes_[3] = x_3;
        break;
    }
    return new_indices_hand;
  }

 private:
  int64_t indexes_[4];
};

static void ComputeXWeightsAndIndices(const ImageResizerState& resizer_state,
                                      const bool half_pixel_centers,
                                      std::vector<WeightsAndIndices>* x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers) {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<HalfPixelScaler, true>(
          resizer_state.width_scale, x, resizer_state.in_width, &(*x_wais)[x]);
      auto& x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2,
                                   x_wai.index_3);
    }
  } else {
    for (int64_t x = 0; x < resizer_state.out_width; ++x) {
      GetWeightsAndIndices<LegacyScaler, false>(
          resizer_state.width_scale, x, resizer_state.in_width, &(*x_wais)[x]);
      auto& x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2,
                                   x_wai.index_3);
    }
  }
  // Scale the values so they can be used as offsets into buffers.
  for (int x = 0; x < resizer_state.out_width; ++x) {
    (*x_wais)[x].index_0 *= resizer_state.channels;
    (*x_wais)[x].index_1 *= resizer_state.channels;
    (*x_wais)[x].index_2 *= resizer_state.channels;
    (*x_wais)[x].index_3 *= resizer_state.channels;
  }
}

static void ComputeGradientXWeightsAndIndices(
    const ImageResizerGradientState& resizer_state,
    const bool half_pixel_centers, std::vector<WeightsAndIndices>* x_wais) {
  CachedInterpolationCalculator calc;
  if (half_pixel_centers) {
    for (int64_t x = 0; x < resizer_state.resized_width; ++x) {
      GetWeightsAndIndices<HalfPixelScaler, true>(resizer_state.width_scale, x,
                                                  resizer_state.original_width,
                                                  &(*x_wais)[x]);
      auto& x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2,
                                   x_wai.index_3);
    }

  } else {
    for (int64_t x = 0; x < resizer_state.resized_width; ++x) {
      GetWeightsAndIndices<LegacyScaler, false>(resizer_state.width_scale, x,
                                                resizer_state.original_width,
                                                &(*x_wais)[x]);
      auto& x_wai = (*x_wais)[x];
      x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2,
                                   x_wai.index_3);
    }
  }
  // Do not scale, as we will be using these directly as tensor indices on the
  // gradient pass.
}

template <typename T>
static EIGEN_ALWAYS_INLINE float ComputeYInterpolation(
    int which, int channel_num, const WeightsAndIndices& y_wai,
    const T* y_ptr_0, const T* y_ptr_1, const T* y_ptr_2, const T* y_ptr_3,
    const WeightsAndIndices& x_wai) {
  int x_index;
  switch (which) {
    case 0:
      x_index = x_wai.index_0;
      break;
    case 1:
      x_index = x_wai.index_1;
      break;
    case 2:
      x_index = x_wai.index_2;
      break;
    default:
      x_index = x_wai.index_3;
      break;
  }
  const int64_t pt_index = x_index + channel_num;
  return Interpolate1D<T>(y_wai.weight_0, y_wai.weight_1, y_wai.weight_2,
                          y_wai.weight_3, y_ptr_0[pt_index], y_ptr_1[pt_index],
                          y_ptr_2[pt_index], y_ptr_3[pt_index]);
}

template <typename T>
inline void interpolate_with_caching(
    const typename TTypes<T, 4>::ConstTensor& input_data,
    const ImageResizerState& resizer_state, const bool half_pixel_centers,
    typename TTypes<float, 4>::Tensor output_data) {
  std::vector<WeightsAndIndices> x_wais(resizer_state.out_width);
  ComputeXWeightsAndIndices(resizer_state, half_pixel_centers, &x_wais);

  const auto num_channels = resizer_state.channels;
  const int64_t in_row_width = resizer_state.in_width * num_channels;
  const int64_t in_batch_width = resizer_state.in_height * in_row_width;

  const T* input_b_ptr = input_data.data();
  float* output_y_ptr = output_data.data();
  std::vector<float> cached_value(num_channels == 3 ? 0 : 4 * num_channels, 0);

  for (int64_t b = 0; b < resizer_state.batch_size;
       ++b, input_b_ptr += in_batch_width) {
    for (int64_t y = 0; y < resizer_state.out_height;
         ++y, output_y_ptr += resizer_state.out_width * num_channels) {
      WeightsAndIndices y_wai;
      if (half_pixel_centers) {
        GetWeightsAndIndices<HalfPixelScaler, true>(
            resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
      } else {
        GetWeightsAndIndices<LegacyScaler, false>(
            resizer_state.height_scale, y, resizer_state.in_height, &y_wai);
      }
      // Make pointers represent offsets of data in input_b_ptr.
      const T* y_ptr_0 = input_b_ptr + y_wai.index_0 * in_row_width;
      const T* y_ptr_1 = input_b_ptr + y_wai.index_1 * in_row_width;
      const T* y_ptr_2 = input_b_ptr + y_wai.index_2 * in_row_width;
      const T* y_ptr_3 = input_b_ptr + y_wai.index_3 * in_row_width;

      if (num_channels == 3) {
        // Manually unroll case of 3 channels.
        float cached_value_0[4] = {0};
        float cached_value_1[4] = {0};
        float cached_value_2[4] = {0};
        for (int64_t x = 0; x < resizer_state.out_width; ++x) {
          const WeightsAndIndices& x_wai = x_wais[x];
          // Shift values in cached_value_* to fill first 'advance' values.
          switch (x_wai.advance) {
            case 3:
              cached_value_0[0] = cached_value_0[1];
              cached_value_0[1] = cached_value_0[2];
              cached_value_0[2] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[1];
              cached_value_1[1] = cached_value_1[2];
              cached_value_1[2] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[1];
              cached_value_2[1] = cached_value_2[2];
              cached_value_2[2] = cached_value_2[3];
              break;
            case 2:
              cached_value_0[0] = cached_value_0[2];
              cached_value_0[1] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[2];
              cached_value_1[1] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[2];
              cached_value_2[1] = cached_value_2[3];
              break;
            case 1: {
              cached_value_0[0] = cached_value_0[3];
              cached_value_1[0] = cached_value_1[3];
              cached_value_2[0] = cached_value_2[3];
              break;
            }
          }

          // Set the remaining '4-advance' values by computing.
          switch (x_wai.advance) {
            case 0:
              cached_value_0[0] = ComputeYInterpolation(
                  0, 0, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_1[0] = ComputeYInterpolation(
                  0, 1, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_2[0] = ComputeYInterpolation(
                  0, 2, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              TF_FALLTHROUGH_INTENDED;
            case 1:
              cached_value_0[1] = ComputeYInterpolation(
                  1, 0, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_1[1] = ComputeYInterpolation(
                  1, 1, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_2[1] = ComputeYInterpolation(
                  1, 2, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              TF_FALLTHROUGH_INTENDED;
            case 2:
              cached_value_0[2] = ComputeYInterpolation(
                  2, 0, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_1[2] = ComputeYInterpolation(
                  2, 1, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_2[2] = ComputeYInterpolation(
                  2, 2, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              TF_FALLTHROUGH_INTENDED;
            case 3:
              cached_value_0[3] = ComputeYInterpolation(
                  3, 0, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_1[3] = ComputeYInterpolation(
                  3, 1, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              cached_value_2[3] = ComputeYInterpolation(
                  3, 2, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              break;
          }
          output_y_ptr[x * num_channels + 0] =
              Compute(cached_value_0, x_wai.weight_0, x_wai.weight_1,
                      x_wai.weight_2, x_wai.weight_3);
          output_y_ptr[x * num_channels + 1] =
              Compute(cached_value_1, x_wai.weight_0, x_wai.weight_1,
                      x_wai.weight_2, x_wai.weight_3);
          output_y_ptr[x * num_channels + 2] =
              Compute(cached_value_2, x_wai.weight_0, x_wai.weight_1,
                      x_wai.weight_2, x_wai.weight_3);
        }
      } else {
        for (int64_t x = 0; x < resizer_state.out_width; ++x) {
          const WeightsAndIndices& x_wai = x_wais[x];
          // Shift values in cached_value to fill first 'advance' values.
          switch (x_wai.advance) {
            case 3:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 0] = cached_value[4 * c + 1];
                cached_value[4 * c + 1] = cached_value[4 * c + 2];
                cached_value[4 * c + 2] = cached_value[4 * c + 3];
              }
              break;
            case 2:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 0] = cached_value[4 * c + 2];
                cached_value[4 * c + 1] = cached_value[4 * c + 3];
              }
              break;
            case 1: {
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 0] = cached_value[4 * c + 3];
              }
              break;
            }
          }

          // Set the remaining '4-advance' values by computing.
          switch (x_wai.advance) {
            case 0:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 0] = ComputeYInterpolation(
                    0, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              }
              TF_FALLTHROUGH_INTENDED;
            case 1:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 1] = ComputeYInterpolation(
                    1, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              }
              TF_FALLTHROUGH_INTENDED;
            case 2:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 2] = ComputeYInterpolation(
                    2, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              }
              TF_FALLTHROUGH_INTENDED;
            case 3:
              for (int64_t c = 0; c < num_channels; ++c) {
                cached_value[4 * c + 3] = ComputeYInterpolation(
                    3, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
              }
              break;
          }
          for (int64_t c = 0; c < num_channels; ++c) {
            output_y_ptr[x * num_channels + c] =
                Compute(&cached_value[4 * c], x_wai.weight_0, x_wai.weight_1,
                        x_wai.weight_2, x_wai.weight_3);
          }
        }
      }
    }
  }
}

template <typename T>
inline void ResizeBicubicGrad(typename TTypes<float, 4>::ConstTensor input_grad,
                              const ImageResizerGradientState& resizer_state,
                              const bool half_pixel_centers,
                              typename TTypes<T, 4>::Tensor output_grad) {
  // This function computes gradients for the ResizeBicubic op by iterating over
  // the input_grad Tensor and using WeightsAndIndices to appropriately update
  // the output gradient.
  const float height_scale = resizer_state.height_scale;
  const int64_t original_height = resizer_state.original_height;
  const int channels = resizer_state.channels;
  const int64_t resized_width = resizer_state.resized_width;
  const int64_t resized_height = resizer_state.resized_height;

  output_grad.setZero();

  std::vector<WeightsAndIndices> x_wais(resizer_state.resized_width);
  ComputeGradientXWeightsAndIndices(resizer_state, half_pixel_centers, &x_wais);
  for (int64_t b = 0; b < resizer_state.batch_size; ++b) {
    for (int64_t y = 0; y < resized_height; ++y) {
      WeightsAndIndices y_wai;
      if (half_pixel_centers) {
        GetWeightsAndIndices<HalfPixelScaler, true>(height_scale, y,
                                                    original_height, &y_wai);
      } else {
        GetWeightsAndIndices<LegacyScaler, false>(height_scale, y,
                                                  original_height, &y_wai);
      }
      for (int64_t x = 0; x < resized_width; ++x) {
        const WeightsAndIndices& x_wai = x_wais[x];
        for (int64_t c = 0; c < channels; ++c) {
          T curr_input_grad = input_grad(b, y, x, c);
          // row 0 of 0, 1, 2, 3
          output_grad(b, y_wai.index_0, x_wai.index_0, c) +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_0);
          output_grad(b, y_wai.index_0, x_wai.index_1, c) +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_1);
          output_grad(b, y_wai.index_0, x_wai.index_2, c) +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_2);
          output_grad(b, y_wai.index_0, x_wai.index_3, c) +=
              T(curr_input_grad * y_wai.weight_0 * x_wai.weight_3);
          // row 1 of 0, 1, 2, 3
          output_grad(b, y_wai.index_1, x_wai.index_0, c) +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_0);
          output_grad(b, y_wai.index_1, x_wai.index_1, c) +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_1);
          output_grad(b, y_wai.index_1, x_wai.index_2, c) +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_2);
          output_grad(b, y_wai.index_1, x_wai.index_3, c) +=
              T(curr_input_grad * y_wai.weight_1 * x_wai.weight_3);
          // row 2 of 0, 1, 2, 3
          output_grad(b, y_wai.index_2, x_wai.index_0, c) +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_0);
          output_grad(b, y_wai.index_2, x_wai.index_1, c) +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_1);
          output_grad(b, y_wai.index_2, x_wai.index_2, c) +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_2);
          output_grad(b, y_wai.index_2, x_wai.index_3, c) +=
              T(curr_input_grad * y_wai.weight_2 * x_wai.weight_3);
          // row 3 of 0, 1, 2, 3
          output_grad(b, y_wai.index_3, x_wai.index_0, c) +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_0);
          output_grad(b, y_wai.index_3, x_wai.index_1, c) +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_1);
          output_grad(b, y_wai.index_3, x_wai.index_2, c) +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_2);
          output_grad(b, y_wai.index_3, x_wai.index_3, c) +=
              T(curr_input_grad * y_wai.weight_3 * x_wai.weight_3);
        }
      }
    }
  }
}

}  // namespace

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeBicubicOp : public OpKernel {
 public:
  explicit ResizeBicubicOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    ImageResizerState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data(
        context->input(0).tensor<T, 4>());
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();

    interpolate_with_caching<T>(input_data, st, half_pixel_centers_,
                                output_data);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

template <typename Device, typename T>
class ResizeBicubicOpGrad : public OpKernel {
 public:
  explicit ResizeBicubicOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
    OP_REQUIRES_OK(
        context, context->GetAttr("half_pixel_centers", &half_pixel_centers_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    ImageResizerGradientState st(align_corners_, half_pixel_centers_);
    st.ValidateAndCreateOutput(context);

    if (!context->status().ok()) return;

    // First argument is gradient with respect to resized image.
    TTypes<float, 4>::ConstTensor input_grad =
        context->input(0).tensor<float, 4>();

    typename TTypes<T, 4>::Tensor output_grad(st.output->tensor<T, 4>());

    ResizeBicubicGrad<T>(input_grad, st, half_pixel_centers_, output_grad);
  }

 private:
  bool align_corners_;
  bool half_pixel_centers_;
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBicubic")       \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBicubicOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                            \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("ResizeBicubicGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ResizeBicubicOpGrad<CPUDevice, T>);

TF_CALL_float(REGISTER_GRAD_KERNEL);
TF_CALL_double(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

}  // namespace tensorflow
