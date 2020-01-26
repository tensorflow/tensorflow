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
#include "tensorflow/python/tools/compiled_model.h"

namespace tensorflow {
namespace {
TEST(AOTCompiledSavedModelTest, Run) {
  CompiledModel model;
  *model.arg_feed_x_data() = 3.0f;
  *model.arg_feed_y_data() = 4.0f;
  CHECK(model.Run());
  ASSERT_NEAR(model.result_fetch_output_0(), 7.0f, /*abs_error=*/1e-6f);
}
}  // namespace
}  // namespace tensorflow
