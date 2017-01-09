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

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"

#include <memory>
#include <string>

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/service/cpu/runtime_matmul.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/common_runtime/eigen_thread_pool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class CpuRuntimeTest : public ::testing::Test {};

template <typename T>
std::unique_ptr<Array2D<float>> MaybeTransposeArray2D(const Array2D<T>& array,
                                                      bool transpose) {
  int64 output_height = array.height();
  int64 output_width = array.width();
  if (transpose) {
    std::swap(output_width, output_height);
  }
  auto output = MakeUnique<Array2D<float>>(output_height, output_width);
  for (int y = 0; y < array.height(); y++) {
    for (int x = 0; x < array.width(); x++) {
      if (transpose) {
        (*output)(x, y) = array(y, x);
      } else {
        (*output)(y, x) = array(y, x);
      }
    }
  }
  return output;
}

// Verifies that matrix 'c' equals the result of matrix 'a' times matrix 'b'.
// Each element is compared to within a small error bound.
void CheckMatrixMultiply(const Array2D<float>& a, const Array2D<float>& b,
                         const Array2D<float>& c) {
  for (int i = 0; i < a.height(); ++i) {
    for (int j = 0; j < b.width(); ++j) {
      float sum = 0.0;
      for (int k = 0; k < a.width(); ++k) {
        sum += a(i, k) * b(k, j);
      }
      EXPECT_NEAR(sum, c(i, j), 0.01);
    }
  }
}

std::unique_ptr<Array2D<float>> EigenMatrixMultiply(const Array2D<float>& a,
                                                    const Array2D<float>& b,
                                                    bool transpose_lhs,
                                                    bool transpose_rhs) {
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "XLAEigen",
                                      2);
  tensorflow::EigenThreadPoolWrapper tp(&pool);
  Eigen::ThreadPoolDevice device(&tp, tp.NumThreads());
  ExecutableRunOptions run_options;
  run_options.set_intra_op_thread_pool(&device);

  CHECK_EQ(a.width(), b.height());
  int64 m = a.height();
  int64 n = b.width();
  int64 k = a.width();

  // The Eigen matmul runtime function expects the matrix to be in column major
  // order and array2d is in row-major order. Create transposes of a and b. The
  // 'data' buffer in the transposed array is the original array in column major
  // order.
  auto a_transpose = MaybeTransposeArray2D(a, !transpose_lhs);
  auto b_transpose = MaybeTransposeArray2D(b, !transpose_rhs);

  // Since we're going to transpose c before returning it. Swap the order of the
  // dimension sizes to ensure the returned array is properly dimensioned.
  auto c_transpose = MakeUnique<Array2D<float>>(n, m);
  __xla_cpu_runtime_EigenMatMulF32(&run_options, c_transpose->data(),
                                   a_transpose->data(), b_transpose->data(), m,
                                   n, k, transpose_lhs, transpose_rhs);
  return MaybeTransposeArray2D(*c_transpose, true);
}

TEST_F(CpuRuntimeTest, SmallEigenMatmul) {
  Array2D<float> a({{1.0f, 2.0f}, {3.0f, 4.0f}});
  Array2D<float> b({{5.0f, -1.0f, 3.0f}, {2.0f, 6.0f, 4.0f}});

  for (bool transpose_lhs : {false, true}) {
    for (bool transpose_rhs : {false, true}) {
      auto c = EigenMatrixMultiply(a, b, transpose_lhs, transpose_rhs);

      LOG(INFO) << "a = " << a.ToString();
      LOG(INFO) << "b = " << b.ToString();
      LOG(INFO) << "c = " << c->ToString();

      CheckMatrixMultiply(a, b, *c);
    }
  }
}

TEST_F(CpuRuntimeTest, LargeEigenMatmul) {
  auto a = MakeLinspaceArray2D(0.0, 1.0, 256, 512);
  auto b = MakeLinspaceArray2D(-2.0, 2.0, 512, 1024);

  for (bool transpose_lhs : {false, true}) {
    for (bool transpose_rhs : {false, true}) {
      auto c = EigenMatrixMultiply(*a, *b, transpose_lhs, transpose_rhs);

      CheckMatrixMultiply(*a, *b, *c);
    }
  }
}

}  // namespace
}  // namespace xla
