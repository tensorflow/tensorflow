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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/tools/aot_compiled_vars_and_arithmetic.h"
#include "tensorflow/python/tools/aot_compiled_vars_and_arithmetic_frozen.h"
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
