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

#include "tensorflow/compiler/mlir/lite/utils/tftext_utils.h"

#include <memory>

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace TFL {

using tensorflow::OpRegistrationData;
using tensorflow::OpRegistry;
using tensorflow::Status;

namespace {

void Register(const std::string& op_name, OpRegistry* registry) {
  registry->Register([op_name](OpRegistrationData* op_reg_data) -> Status {
    op_reg_data->op_def.set_name(op_name);
    return Status::OK();
  });
}

}  // namespace

TEST(TfTextUtilsTest, TestTfTextRegistered) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("WhitespaceTokenizeWithOffsets", registry.get());
  EXPECT_TRUE(IsTFTextRegistered(registry.get()));
}

TEST(TfTextUtilsTest, TestTfTextNotRegistered) {
  std::unique_ptr<OpRegistry> registry(new OpRegistry);
  Register("Test", registry.get());
  EXPECT_FALSE(IsTFTextRegistered(registry.get()));
}
}  // namespace TFL
}  // namespace mlir
