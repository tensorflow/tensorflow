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

inline void GetWeightsAndIndices(const float scale, const int64 out_loc,
                                 const int64 limit, float* weight_0,
                                 float* weight_1, float* weight_2,
                                 float* weight_3, int64* index_0,
                                 int64* index_1, int64* index_2,
                                 int64* index_3) {
  const int64 in_loc = scale * out_loc;
  const float delta = scale * out_loc - in_loc;
  const int64 offset = lrintf(delta * kTableSize);
  const float* coeffs_table = GetCoeffsTable();
  *weight_0 = coeffs_table[offset * 2 + 1];
  *weight_1 = coeffs_table[offset * 2];
  *weight_2 = coeffs_table[(kTableSize - offset) * 2];
  *weight_3 = coeffs_table[(kTableSize - offset) * 2 + 1];
  *index_0 = Bound(in_loc - 1, limit);
  *index_1 = Bound(in_loc, limit);
  *index_2 = Bound(in_loc + 1, limit);
  *index_3 = Bound(in_loc + 2, limit);
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

// In order to compute a single output value, we look at a 4x4 patch in the
// source image. As we iterate increasing X across the image, the new 4x4 patch
// often overlaps with the previous 4x4 patch we just looked at.
//
// This class helps retain that intermediate computation work.
class CachedInterpolation {
 public:
  CachedInterpolation()
      : values_({{std::make_pair(-1, -1), std::make_pair(-1, -1),
                  std::make_pair(-1, -1), std::make_pair(-1, -1)}}) {}

  // Advances the buffer. Returns the number of valid values.
  inline int Advance(const int64 x_0, const int64 x_1, const int64 x_2,
                     const int64 x_3) {
    // Either we have started a new line, or we don't have any values yet.
    if (x_0 < values_[0].first || values_[0].first == -1) {
      // Zero cached values were valid, we must recompute everything.
      return 0;
    }
    if (values_[0].first == x_0 && values_[3].first == x_3) {
      // Everything's the same. Yay!
      return 4;
    }
    if (values_[1].first != 0 && values_[2].first != values_[3].first) {
      // Fast (normal) path
      if (values_[1].first == x_0) {
        CopyPoint(1, 0);
        CopyPoint(2, 1);
        CopyPoint(3, 2);
        return 3;
      }
      if (values_[2].first == x_0) {
        CopyPoint(2, 0);
        CopyPoint(3, 1);
        return 2;
      }
    }
    // We use 2 hands and walk through, copying from one to another where
    // we already have values.
    // Invarient, new_indicies_hand <= cached_values_hand
    const std::array<int64, 4> new_x_indices{{x_0, x_1, x_2, x_3}};
    int cached_values_hand = 0;
    int new_indicies_hand = 0;
    while (cached_values_hand < 4) {
      if (values_[cached_values_hand].first ==
          new_x_indices[new_indicies_hand]) {
        if (new_indicies_hand < cached_values_hand) {
          CopyPoint(cached_values_hand, new_indicies_hand);
        }
        cached_values_hand++;
        new_indicies_hand++;
      } else {
        cached_values_hand++;
      }
    }
    return new_indicies_hand;
  }

  inline void SetPoint(const int index, const int64 x_index,
                       const float value) {
    values_[index] = std::make_pair(x_index, value);
  }

  // Compute the 1D interpolation for a given X index using the y_weights
  inline float Compute(const float xw_0, const float xw_1, const float xw_2,
                       const float xw_3) const {
    return Interpolate1D(xw_0, xw_1, xw_2, xw_3, values_[0].second,
                         values_[1].second, values_[2].second,
                         values_[3].second);
  }

 private:
  inline void CopyPoint(const int source, const int dest) {
    values_[dest] = values_[source];
  }

  std::array<std::pair<int64, float>, 4> values_;
};

template <typename T>
inline void interpolate_with_caching(
    const typename TTypes<T, 4>::ConstTensor& input_data,
    const ImageResizerState& resizer_state,
    typename TTypes<float, 4>::Tensor output_data) {
  std::vector<CachedInterpolation> cached_values(resizer_state.channels);
  for (int64 b = 0; b < resizer_state.batch_size; ++b) {
    for (int64 y = 0; y < resizer_state.out_height; ++y) {
      float y_weight_0;
      float y_weight_1;
      float y_weight_2;
      float y_weight_3;
      int64 y_index_0;
      int64 y_index_1;
      int64 y_index_2;
      int64 y_index_3;
      GetWeightsAndIndices(resizer_state.height_scale, y,
                           resizer_state.in_height, &y_weight_0, &y_weight_1,
                           &y_weight_2, &y_weight_3, &y_index_0, &y_index_1,
                           &y_index_2, &y_index_3);
      for (int64 x = 0; x < resizer_state.out_width; ++x) {
        float xw_0;
        float xw_1;
        float xw_2;
        float xw_3;
        int64 x_index_0;
        int64 x_index_1;
        int64 x_index_2;
        int64 x_index_3;
        GetWeightsAndIndices(resizer_state.width_scale, x,
                             resizer_state.in_width, &xw_0, &xw_1, &xw_2, &xw_3,
                             &x_index_0, &x_index_1, &x_index_2, &x_index_3);
        for (int64 c = 0; c < resizer_state.channels; ++c) {
          const int advance = cached_values[c].Advance(x_index_0, x_index_1,
                                                       x_index_2, x_index_3);
          switch (advance) {
            case 0:
              cached_values[c].SetPoint(
                  0, x_index_0,
                  Interpolate1D<T>(y_weight_0, y_weight_1, y_weight_2,
                                   y_weight_3,
                                   input_data(b, y_index_0, x_index_0, c),
                                   input_data(b, y_index_1, x_index_0, c),
                                   input_data(b, y_index_2, x_index_0, c),
                                   input_data(b, y_index_3, x_index_0, c)));
              TF_FALLTHROUGH_INTENDED;
            case 1:
              cached_values[c].SetPoint(
                  1, x_index_1,
                  Interpolate1D<T>(y_weight_0, y_weight_1, y_weight_2,
                                   y_weight_3,
                                   input_data(b, y_index_0, x_index_1, c),
                                   input_data(b, y_index_1, x_index_1, c),
                                   input_data(b, y_index_2, x_index_1, c),
                                   input_data(b, y_index_3, x_index_1, c)));
              TF_FALLTHROUGH_INTENDED;
            case 2:
              cached_values[c].SetPoint(
                  2, x_index_2,
                  Interpolate1D<T>(y_weight_0, y_weight_1, y_weight_2,
                                   y_weight_3,
                                   input_data(b, y_index_0, x_index_2, c),
                                   input_data(b, y_index_1, x_index_2, c),
                                   input_data(b, y_index_2, x_index_2, c),
                                   input_data(b, y_index_3, x_index_2, c)));
              TF_FALLTHROUGH_INTENDED;
            case 3:
              cached_values[c].SetPoint(
                  3, x_index_3,
                  Interpolate1D<T>(y_weight_0, y_weight_1, y_weight_2,
                                   y_weight_3,
                                   input_data(b, y_index_0, x_index_3, c),
                                   input_data(b, y_index_1, x_index_3, c),
                                   input_data(b, y_index_2, x_index_3, c),
                                   input_data(b, y_index_3, x_index_3, c)));
              TF_FALLTHROUGH_INTENDED;
            default:
              output_data(b, y, x, c) =
                  cached_values[c].Compute(xw_0, xw_1, xw_2, xw_3);
              break;
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
