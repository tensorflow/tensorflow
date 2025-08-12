/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/dtensor/cc/dtensor_device_util.h"
#include "tensorflow/dtensor/cc/dtensor_operation.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"

namespace tensorflow {
namespace dtensor {
namespace {

using ::testing::HasSubstr;
using ::tsl::error::UNAVAILABLE;

class ExecutableManagerTest : public ::testing::Test {
 protected:
  DTensorOperation CreateTestDTensorOperation() {
    return DTensorOperation{"test_fn", nullptr, empty_mesh_, {}};
  }

  Mesh empty_mesh_ = Mesh::Empty();

  core::RefCountPtr<ExecutableManager<ExecutionFunctions>> function_manager_{
      new ExecutableManager<ExecutionFunctions>()};
};

TEST_F(ExecutableManagerTest, ShouldFoldInputUnavailable) {
  auto result =
      function_manager_->ShouldFoldInput(CreateTestDTensorOperation(), {}, 0);
  EXPECT_THAT(result,
              absl_testing::StatusIs(
                  UNAVAILABLE, HasSubstr("ExecutionFunctions manager can not "
                                         "check if the input is foldable")));
}

TEST_F(ExecutableManagerTest, GetCachedExecutableUnavailable) {
  DTensorOperation doperation = CreateTestDTensorOperation();
  NameAttrList func_attr;
  func_attr.set_name(doperation.name);
  auto result = function_manager_->GetCachedExecutable(
      doperation, func_attr,
      {nullptr},  // Dummy input to trigger ShouldFoldInput check.
      {});
  EXPECT_THAT(result, absl_testing::StatusIs(UNAVAILABLE));
}

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
