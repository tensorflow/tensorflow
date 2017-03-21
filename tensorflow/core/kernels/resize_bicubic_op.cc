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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

static const int64 kTableSize = (1 << 10);

const float* InitCoeffsTable() {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  float* coeffs_table = new float[(kTableSize + 1) * 2];
  static const double A = -0.75;
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_table[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs_table[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  }
  return coeffs_table;
}

const float* GetCoeffsTable() {
  // Static so that we initialize it on first use
  static const float* coeffs_table = InitCoeffsTable();
  return coeffs_table;
}

inline int64 Bound(int64 val, int64 limit) {
  return std::min(limit - 1ll, std::max(0ll, val));
}

struct WeightsAndIndices {
  float weight_0;
  float weight_1;
  float weight_2;
  float weight_3;
  int64 index_0;
  int64 index_1;
  int64 index_2;
  int64 index_3;

  int advance;  // advance value.
};

inline void GetWeightsAndIndices(const float scale, const int64 out_loc,
                                 const int64 limit, WeightsAndIndices* out) {
  const int64 in_loc = scale * out_loc;
  const float delta = scale * out_loc - in_loc;
  const int64 offset = lrintf(delta * kTableSize);
  const float* coeffs_table = GetCoeffsTable();
  out->weight_0 = coeffs_table[offset * 2 + 1];
  out->weight_1 = coeffs_table[offset * 2];
  out->weight_2 = coeffs_table[(kTableSize - offset) * 2];
  out->weight_3 = coeffs_table[(kTableSize - offset) * 2 + 1];
  out->index_0 = Bound(in_loc - 1, limit);
  out->index_1 = Bound(in_loc, limit);
  out->index_2 = Bound(in_loc + 1, limit);
  out->index_3 = Bound(in_loc + 2, limit);
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
  inline int Advance(const int64 x_0, const int64 x_1, const int64 x_2,
                     const int64 x_3) {
    // We use 2 hands and walk through, copying from one to another where
    // we already have values.
    // Invariant, new_indicies_hand <= cached_values_hand
    const std::array<int64, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    int cached_values_hand = 0;
    int new_indicies_hand = 0;
    while (cached_values_hand < 4) {
      if (indexes_[cached_values_hand] == new_x_indices[new_indicies_hand]) {
        if (new_indicies_hand < cached_values_hand) {
          indexes_[new_indicies_hand] = indexes_[cached_values_hand];
        }
        cached_values_hand++;
        new_indicies_hand++;
      } else {
        cached_values_hand++;
      }
    }
    switch (new_indicies_hand) {
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
    return new_indicies_hand;
  }

 private:
  int64 indexes_[4];
};

static void ComputeXWeightsAndIndices(const ImageResizerState& resizer_state,
                                      std::vector<WeightsAndIndices>* x_wais) {
  CachedInterpolationCalculator calc;
  for (int64 x = 0; x < resizer_state.out_width; ++x) {
    GetWeightsAndIndices(resizer_state.width_scale, x, resizer_state.in_width,
                         &(*x_wais)[x]);
    auto& x_wai = (*x_wais)[x];
    x_wai.advance = calc.Advance(x_wai.index_0, x_wai.index_1, x_wai.index_2,
                                 x_wai.index_3);
  }
  // Scale the values so they can be used as offsets into buffers.
  for (int x = 0; x < resizer_state.out_width; ++x) {
    (*x_wais)[x].index_0 *= resizer_state.channels;
    (*x_wais)[x].index_1 *= resizer_state.channels;
    (*x_wais)[x].index_2 *= resizer_state.channels;
    (*x_wais)[x].index_3 *= resizer_state.channels;
  }
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
  const int64 pt_index = x_index + channel_num;
  return Interpolate1D<T>(y_wai.weight_0, y_wai.weight_1, y_wai.weight_2,
                          y_wai.weight_3, y_ptr_0[pt_index], y_ptr_1[pt_index],
                          y_ptr_2[pt_index], y_ptr_3[pt_index]);
}

template <typename T>
inline void interpolate_with_caching(
    const typename TTypes<T, 4>::ConstTensor& input_data,
    const ImageResizerState& resizer_state,
    typename TTypes<float, 4>::Tensor output_data) {
  std::vector<WeightsAndIndices> x_wais(resizer_state.out_width);
  ComputeXWeightsAndIndices(resizer_state, &x_wais);

  const auto num_channels = resizer_state.channels;
  const int64 in_row_width = resizer_state.in_width * num_channels;
  const int64 in_batch_width = resizer_state.in_height * in_row_width;

  const T* input_b_ptr = input_data.data();
  float* output_y_ptr = output_data.data();

  for (int64 b = 0; b < resizer_state.batch_size;
       ++b, input_b_ptr += in_batch_width) {
    for (int64 y = 0; y < resizer_state.out_height;
         ++y, output_y_ptr += resizer_state.out_width * num_channels) {
      WeightsAndIndices y_wai;
      GetWeightsAndIndices(resizer_state.height_scale, y,
                           resizer_state.in_height, &y_wai);
      // Make pointers represent offsets of data in input_b_ptr.
      const T* y_ptr_0 = input_b_ptr + y_wai.index_0 * in_row_width;
      const T* y_ptr_1 = input_b_ptr + y_wai.index_1 * in_row_width;
      const T* y_ptr_2 = input_b_ptr + y_wai.index_2 * in_row_width;
      const T* y_ptr_3 = input_b_ptr + y_wai.index_3 * in_row_width;
      if (num_channels == 3) {
        // Manually unroll case of 3 channels.
        float cached_value_0[4];
        float cached_value_1[4];
        float cached_value_2[4];
        for (int64 x = 0; x < resizer_state.out_width; ++x) {
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
        for (int64 c = 0; c < num_channels; ++c) {
          float cached_value[4];
          for (int64 x = 0; x < resizer_state.out_width; ++x) {
            const WeightsAndIndices& x_wai = x_wais[x];
            // Shift values in cached_value to fill first 'advance' values.
            switch (x_wai.advance) {
              case 3:
                cached_value[0] = cached_value[1];
                cached_value[1] = cached_value[2];
                cached_value[2] = cached_value[3];
                break;
              case 2:
                cached_value[0] = cached_value[2];
                cached_value[1] = cached_value[3];
                break;
              case 1: {
                cached_value[0] = cached_value[3];
                break;
              }
            }

            // Set the remaining '4-advance' values by computing.
            switch (x_wai.advance) {
              case 0:
                cached_value[0] = ComputeYInterpolation(
                    0, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
                TF_FALLTHROUGH_INTENDED;
              case 1:
                cached_value[1] = ComputeYInterpolation(
                    1, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
                TF_FALLTHROUGH_INTENDED;
              case 2:
                cached_value[2] = ComputeYInterpolation(
                    2, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
                TF_FALLTHROUGH_INTENDED;
              case 3:
                cached_value[3] = ComputeYInterpolation(
                    3, c, y_wai, y_ptr_0, y_ptr_1, y_ptr_2, y_ptr_3, x_wai);
                break;
            }
            output_y_ptr[x * num_channels + c] =
                Compute(cached_value, x_wai.weight_0, x_wai.weight_1,
                        x_wai.weight_2, x_wai.weight_3);
          }
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
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data =
        st.output->tensor<float, 4>();

    interpolate_with_caching<T>(input_data, st, output_data);
  }

 private:
  bool align_corners_;
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBicubic")       \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBicubicOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // namespace tensorflow
