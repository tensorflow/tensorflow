/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/tools/aot_compiled_vars_and_arithmetic.h"
#include "tensorflow/python/tools/aot_compiled_vars_and_arithmetic_frozen.h"
#include "tensorflow/python/tools/aot_compiled_x_matmul_y_large.h"
#include "tensorflow/python/tools/aot_compiled_x_matmul_y_small.h"
#include "tensorflow/python/tools/aot_compiled_x_plus_y.h"

namespace tensorflow {
namespace {
TEST(AOTCompiledSavedModelTest, XPlusY) {
  XPlusY model;
  // Calculation is: output_0 = x + y.
  *model.arg_feed_x_data() = 3.0f;
  *model.arg_feed_y_data() = 4.0f;
  CHECK(model.Run());
  ASSERT_NEAR(model.result_fetch_output_0(), 7.0f, /*abs_error=*/1e-6f);
}

TEST(AOTCompiledSavedModelTest, XMatmulYLarge) {
  XMatmulYLarge model;
  // Calculation is: output_0 = x @ y.
  EXPECT_EQ(model.arg0_size(), sizeof(float) * 3000 * 5000);
  EXPECT_EQ(model.arg1_size(), sizeof(float) * 5000 * 4000);
  EXPECT_EQ(model.result0_size(), sizeof(float) * 3000 * 4000);

  Eigen::Tensor<float, 2, Eigen::RowMajor> arg0(3000, 5000);
  Eigen::Tensor<float, 2, Eigen::RowMajor> arg1(5000, 4000);
  arg0.setRandom();
  arg1.setRandom();

  // Set up dimensions for standard matmul.
  const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(1, 0)};
  // Ground truth matmul.
  const Eigen::Tensor<float, 2, Eigen::RowMajor> expected_output0 =
      arg0.contract(arg1, product_dims);

  model.set_arg_feed_x_data(arg0.data());
  model.set_arg_feed_y_data(arg1.data());
  CHECK(model.Run());
  EXPECT_NEAR(model.result_fetch_output_0(0, 0), expected_output0(0, 0),
              /*abs_error=*/1e-6f);
  EXPECT_NEAR(model.result_fetch_output_0(2999, 3999),
              expected_output0(2999, 3999),
              /*abs_error=*/1e-6f);
}

TEST(AOTCompiledSavedModelTest, XMatmulYSmall) {
  XMatmulYSmall model;
  // Calculation is: output_0 = x @ y.
  EXPECT_EQ(model.arg0_size(), sizeof(float) * 3 * 5);
  EXPECT_EQ(model.arg1_size(), sizeof(float) * 5 * 4);
  EXPECT_EQ(model.result0_size(), sizeof(float) * 3 * 4);

  Eigen::Tensor<float, 2, Eigen::RowMajor> arg0(3, 5);
  Eigen::Tensor<float, 2, Eigen::RowMajor> arg1(5, 4);
  arg0.setRandom();
  arg1.setRandom();

  // Set up dimensions for standard matmul.
  const Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {
      Eigen::IndexPair<int>(1, 0)};
  // Ground truth matmul.
  const Eigen::Tensor<float, 2, Eigen::RowMajor> expected_output0 =
      arg0.contract(arg1, product_dims);

  model.set_arg_feed_x_data(arg0.data());
  model.set_arg_feed_y_data(arg1.data());
  CHECK(model.Run());
  EXPECT_NEAR(model.result_fetch_output_0(0, 0), expected_output0(0, 0),
              /*abs_error=*/1e-6f);
  EXPECT_NEAR(model.result_fetch_output_0(2, 3), expected_output0(2, 3),
              /*abs_error=*/1e-6f);
}

TEST(AOTCompiledSavedModelTest, VarsAndArithmetic) {
  VarsAndArithmeticFrozen frozen_model;
  // Calculation is:
  //   output_0 = [(a + variable_x) * (b + variable_y) / child_variable] + 5.0
  // where {variable_x, variable_y, child_variable} = {1.0, 2.0, 3.0} when
  // initialized (frozen).
  *frozen_model.arg_feed_a_data() = 1.0f;
  *frozen_model.arg_feed_b_data() = 2.0f;
  CHECK(frozen_model.Run());
  ASSERT_NEAR(frozen_model.result_fetch_output_0(),
              (1.0f + 1.0f) * (2.0f + 2.0f) / 3.0f + 5.0f, /*abs_error=*/1e-6f);

  VarsAndArithmetic nonfrozen_model;
  *nonfrozen_model.arg_feed_a_data() = 1.0f;
  *nonfrozen_model.arg_feed_b_data() = 2.0f;
  // variable_x is no longer frozen.  set it to 4.0;
  float new_variable_x = 4.0f;
  nonfrozen_model.set_var_param_variable_x_data(&new_variable_x);
  CHECK(nonfrozen_model.Run());
  ASSERT_NEAR(nonfrozen_model.result_fetch_output_0(),
              (1.0f + 4.0f) * (2.0f + 2.0f) / 3.0f + 5.0f, /*abs_error=*/1e-6f);
}
}  // namespace
}  // namespace tensorflow
