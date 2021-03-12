/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/qr.h"

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"

namespace {

using QrTest = xla::ClientLibraryTestBase;

XLA_TEST_F(QrTest, Simple) {
  // Test fails with TensorFloat-32 enabled
  tensorflow::enable_tensor_float_32_execution(false);

  xla::Array2D<float> data({
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  });

  for (bool full_matrices : {false, true}) {
    for (xla::int64 m : {3, 4}) {
      for (xla::int64 n : {3, 4}) {
        xla::XlaBuilder builder(TestName());
        xla::XlaOp a, q, r;
        xla::Array<float> a_vals = data.Slice({0, 0}, {m, n});
        auto a_data = CreateParameter<float>(a_vals, 0, "a", &builder, &a);
        xla::QrExplicit(a, full_matrices, q, r);

        // Verifies that the decomposition composes back to the original matrix.
        //
        // This isn't a terribly demanding test, (e.g., we should verify that Q
        // is orthonormal and R is upper-triangular) but it's awkward to write
        // such tests without more linear algebra libraries. It's easier to test
        // the numerics from Python, anyway, where we have access to numpy and
        // scipy.
        xla::BatchDot(q, r, xla::PrecisionConfig::HIGHEST);
        TF_ASSERT_OK_AND_ASSIGN(xla::Shape q_shape, builder.GetShape(q));
        TF_ASSERT_OK_AND_ASSIGN(xla::Shape r_shape, builder.GetShape(r));
        EXPECT_EQ(q_shape,
                  xla::ShapeUtil::MakeShape(
                      xla::F32, {m, full_matrices ? m : std::min(m, n)}));
        EXPECT_EQ(r_shape,
                  xla::ShapeUtil::MakeShape(
                      xla::F32, {full_matrices ? m : std::min(m, n), n}));
        ComputeAndCompare<float>(&builder, a_vals, {a_data.get()},
                                 xla::ErrorSpec(1e-4, 1e-4));
      }
    }
  }
}

XLA_TEST_F(QrTest, ZeroDiagonal) {
  // Test fails with TensorFloat-32 enabled
  tensorflow::enable_tensor_float_32_execution(false);
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 0},
  });

  xla::XlaOp a, q, r;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::QrExplicit(a, /*full_matrices=*/true, q, r);

  // Verifies that the decomposition composes back to the original matrix.
  //
  // This isn't a terribly demanding test, (e.g., we should verify that Q is
  // orthonormal and R is upper-triangular) but it's awkward to write such tests
  // without more linear algebra libraries. It's easier to test the numerics
  // from Python, anyway, where we have access to numpy and scipy.
  xla::BatchDot(q, r, xla::PrecisionConfig::HIGHEST);

  ComputeAndCompareR2<float>(&builder, a_vals, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(QrTest, SimpleBatched) {
  // Test fails with TensorFloat-32 enabled
  tensorflow::enable_tensor_float_32_execution(false);
  xla::XlaBuilder builder(TestName());

  xla::Array3D<float> a_vals({
      {
          {4, 6, 8, 10},
          {6, 45, 54, 63},
          {8, 54, 146, 166},
          {10, 63, 166, 310},
      },
      {
          {16, 24, 8, 12},
          {24, 61, 82, 48},
          {8, 82, 456, 106},
          {12, 48, 106, 62},
      },
  });

  xla::XlaOp a, q, r;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  xla::QrExplicit(a, /*full_matrices=*/true, q, r);

  xla::BatchDot(q, r, xla::PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, a_vals, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(QrTest, SubnormalComplex) {
  tensorflow::enable_tensor_float_32_execution(false);

  // Verifies that we don't get NaNs in the case that the norm of a complex
  // number would be denormal but its imaginary value is not exactly 0.
  xla::Array2D<xla::complex64> a_vals({
      {xla::complex64(4e-20, 5e-23), 6, 80},
      {0, 45, 54},
      {0, 54, 146},
  });

  xla::XlaBuilder builder(TestName());
  xla::XlaOp a, q, r;
  auto a_data = CreateParameter<xla::complex64>(a_vals, 0, "a", &builder, &a);
  xla::QrExplicit(a, /*full_matrices=*/true, q, r);
  xla::BatchDot(q, r, xla::PrecisionConfig::HIGHEST);
  ComputeAndCompare<xla::complex64>(&builder, a_vals, {a_data.get()},
                                    xla::ErrorSpec(1e-4, 1e-4));
}

}  // namespace
