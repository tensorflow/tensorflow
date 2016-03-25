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

#include "tensorflow/core/kernels/eigen_activations.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/test.h"

namespace Eigen {

namespace {
void EigenApprox(float a, float b) {
  ASSERT_TRUE(std::abs(a - b) <= std::min(std::abs(a), std::abs(b)) * 1e-3);
}
}

TEST(EigenBackwardSpatialConvolutionsTest, SigmoidFastDerivative) {
  const ptrdiff_t depth = 3;
  const ptrdiff_t batch = 10;
  const ptrdiff_t rows = 32;
  const ptrdiff_t cols = 48;

  Tensor<float, 4> input(depth, rows, cols, batch);
  input.setRandom();

  Tensor<float, 4> result(depth, rows, cols, batch);
  result = input.unaryExpr(scalar_sigmoid_fast_derivative_op<float>());

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < cols; ++c) {
      for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < depth; ++d) {
          float val = input(d, r, c, b);
          EigenApprox(result(d, r, c, b), (1 - val) * val);
        }
      }
    }
  }
}

TEST(EigenBackwardSpatialConvolutionsTest, TanhFastDerivative) {
  const ptrdiff_t depth = 3;
  const ptrdiff_t batch = 10;
  const ptrdiff_t rows = 32;
  const ptrdiff_t cols = 48;

  Tensor<float, 4> input(depth, rows, cols, batch);
  input.setRandom();

  Tensor<float, 4> result(depth, rows, cols, batch);
  result = input.unaryExpr(scalar_tanh_fast_derivative_op<float>());

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < cols; ++c) {
      for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < depth; ++d) {
          float val = input(d, r, c, b);
          EigenApprox(result(d, r, c, b), 1 - (val * val));
        }
      }
    }
  }
}

TEST(EigenBackwardSpatialConvolutionsTest, Clip) {
  const ptrdiff_t depth = 3;
  const ptrdiff_t batch = 10;
  const ptrdiff_t rows = 32;
  const ptrdiff_t cols = 48;

  Tensor<float, 4> input(depth, rows, cols, batch);
  input.setRandom();

  Tensor<float, 4> result(depth, rows, cols, batch);
  result = input.binaryExpr(input.constant(0.01), scalar_clip_op<float>());

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < cols; ++c) {
      for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < depth; ++d) {
          float val = input(d, r, c, b);
          EigenApprox(result(d, r, c, b),
                      (std::min)((std::max)(val, -0.01f), 0.01f));
        }
      }
    }
  }
}

}  // namespace Eigen
