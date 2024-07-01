/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"

#include <gtest/gtest.h>
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"

namespace tensorflow {
namespace {

TEST(GetTfrtPipelineOptions, BatchPaddingPolicy) {
  tensorflow::TfrtCompileOptions options;
  options.batch_padding_policy = "PAD_TEST_OPTION";
  auto pipeline_options = GetTfrtPipelineOptions(options);
  EXPECT_EQ(pipeline_options->batch_padding_policy, "PAD_TEST_OPTION");
}

}  // namespace
}  // namespace tensorflow
