/* Copyright 2015 Google Inc. All Rights Reserved.

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
  float* coeffs_tab = new float[(kTableSize + 1) * 2];
  static const double A = -0.75;
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  }
  return coeffs_tab;
}

const float* GetCoeffsTable() {
  // Static so that we initialize it on first use
  static const float* coeffs_tab = InitCoeffsTable();
  return coeffs_tab;
}

inline int64 Bound(int64 val, int64 limit) {
  return std::min(limit - 1ll, std::max(0ll, val));
}

inline void GetWeightsAndIndices(float scale, int64 out_loc, int64 limit,
                                 std::array<float, 4>* weights,
                                 std::array<int64, 4>* indices) {
  const int64 in_loc = scale * out_loc;
  const float delta = scale * out_loc - in_loc;
  const int64 offset = lrintf(delta * kTableSize);
  const float* coeffs_tab = GetCoeffsTable();
  *weights = {{coeffs_tab[offset * 2 + 1], coeffs_tab[offset * 2],
               coeffs_tab[(kTableSize - offset) * 2],
               coeffs_tab[(kTableSize - offset) * 2 + 1]}};
  *indices = {{Bound(in_loc - 1, limit), Bound(in_loc, limit),
               Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)}};
}

inline float Interpolate1D(const std::array<float, 4>& weights,
                           const std::array<float, 4>& values) {
  return values[0] * weights[0] + values[1] * weights[1] +
         values[2] * weights[2] + values[3] * weights[3];
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

    std::array<float, 4> coeff = {{0.0, 0.0, 0.0, 0.0}};
    for (int64 b = 0; b < st.batch_size; ++b) {
      for (int64 y = 0; y < st.out_height; ++y) {
        std::array<float, 4> y_weights;
        std::array<int64, 4> y_indices;
        GetWeightsAndIndices(st.height_scale, y, st.in_height, &y_weights,
                             &y_indices);
        for (int64 x = 0; x < st.out_width; ++x) {
          std::array<float, 4> x_weights;
          std::array<int64, 4> x_indices;
          GetWeightsAndIndices(st.width_scale, x, st.in_width, &x_weights,
                               &x_indices);
          for (int64 c = 0; c < st.channels; ++c) {
            // Use a 4x4 patch to compute the interpolated output value at
            // (b, y, x, c).
            for (int64 i = 0; i < 4; ++i) {
              const std::array<float, 4> values = {
                  {static_cast<float>(
                       input_data(b, y_indices[i], x_indices[0], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[1], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[2], c)),
                   static_cast<float>(
                       input_data(b, y_indices[i], x_indices[3], c))}};
              coeff[i] = Interpolate1D(x_weights, values);
            }
            output_data(b, y, x, c) = Interpolate1D(y_weights, coeff);
          }
        }
      }
    }
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
