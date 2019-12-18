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
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace {

using QrTest = xla::ClientLibraryTestBase;

XLA_TEST_F(QrTest, Simple) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {4, 6, 8, 10},
      {6, 45, 54, 63},
      {8, 54, 146, 166},
      {10, 63, 166, 310},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      xla::QRDecomposition(a, /*full_matrices=*/true, /*block_size=*/2));

  // Verifies that the decomposition composes back to the original matrix.
  //
  // This isn't a terribly demanding test, (e.g., we should verify that Q is
  // orthonormal and R is upper-triangular) but it's awkward to write such tests
  // without more linear algebra libraries. It's easier to test the numerics
  // from Python, anyway, where we have access to numpy and scipy.
  xla::BatchDot(result.q, result.r, xla::PrecisionConfig::HIGHEST);

  ComputeAndCompareR2<float>(&builder, a_vals, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(QrTest, ZeroDiagonal) {
  xla::XlaBuilder builder(TestName());

  xla::Array2D<float> a_vals({
      {0, 1, 1},
      {1, 0, 1},
      {1, 1, 0},
  });

  xla::XlaOp a;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      xla::QRDecomposition(a, /*full_matrices=*/true, /*block_size=*/8));

  // Verifies that the decomposition composes back to the original matrix.
  //
  // This isn't a terribly demanding test, (e.g., we should verify that Q is
  // orthonormal and R is upper-triangular) but it's awkward to write such tests
  // without more linear algebra libraries. It's easier to test the numerics
  // from Python, anyway, where we have access to numpy and scipy.
  xla::BatchDot(result.q, result.r, xla::PrecisionConfig::HIGHEST);

  ComputeAndCompareR2<float>(&builder, a_vals, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

XLA_TEST_F(QrTest, SimpleBatched) {
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

  xla::XlaOp a;
  auto a_data = CreateR3Parameter<float>(a_vals, 0, "a", &builder, &a);
  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      xla::QRDecomposition(a, /*full_matrices=*/true, /*block_size=*/2));

  xla::BatchDot(result.q, result.r, xla::PrecisionConfig::HIGHEST);

  ComputeAndCompareR3<float>(&builder, a_vals, {a_data.get()},
                             xla::ErrorSpec(1e-4, 1e-4));
}

}  // namespace
