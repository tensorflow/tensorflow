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

#include "tensorflow/compiler/tf2xla/lib/triangular_solve.h"

#include <memory>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

using TriangularSolveTest = xla::ClientLibraryTestBase;

XLA_TEST_F(TriangularSolveTest, Simple) {
  xla::ComputationBuilder builder(client_, TestName());

  xla::Array2D<float> a_vals({
      {2, 0, 0, 0},
      {3, 6, 0, 0},
      {4, 7, 9, 0},
      {5, 8, 10, 11},
  });
  xla::Array2D<float> b_vals({
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  });

  xla::ComputationDataHandle a, b;
  auto a_data = CreateR2Parameter<float>(a_vals, 0, "a", &builder, &a);
  auto b_data = CreateR2Parameter<float>(b_vals, 1, "b", &builder, &b);
  auto result = TriangularSolve(&builder, a, b, /*block_size=*/2);
  TF_ASSERT_OK(result.status());

  xla::Array2D<float> expected({
      {0.5, 0.08333334, 0.04629629, 0.03367003},
      {2.5, -0.25, -0.1388889, -0.1010101},
      {4.5, -0.58333331, -0.32407406, -0.23569024},
  });

  ComputeAndCompareR2<float>(&builder, expected, {a_data.get(), b_data.get()},
                             xla::ErrorSpec(2e-3, 2e-3));
}

}  // namespace
}  // namespace tensorflow
