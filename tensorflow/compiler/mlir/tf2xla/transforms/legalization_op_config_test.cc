/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status.h"

namespace mlir {
namespace mhlo {

using mlir::ModuleOp;
using tsl::Status;

static constexpr char kMlirModuleStr[] = R"(
module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1442 : i32}} {
  func.func @main(%arg0: tensor<3xi64> {tf._user_specified_name = "resource", tf.aliasing_output = 3 : i64}) -> () attributes {tf.entry_function = {control_outputs = "stateful_normal/RngReadAndSkip,stateful_uniform/RngReadAndSkip,stateful_uniform_full_int/RngReadAndSkip", inputs = "stateful_normal_rngreadandskip_resource", outputs = "identity_RetVal,identity_1_RetVal,identity_2_RetVal"}} {
    %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    return
  }
})";

class LegalizationOpConfigTest : public ::testing::Test {
 public:
  Status CreateMlirModule(std::string module_string = kMlirModuleStr) {
    TF_ASSIGN_OR_RETURN(
        module_, test::GetMlirModuleFromString(module_string, &context_));

    context_.loadAllAvailableDialects();
    return tsl::OkStatus();
  }

 protected:
  MLIRContext context_;
  OwningOpRef<ModuleOp> module_;
};

TEST_F(LegalizationOpConfigTest, ExpectsLegalizationWithMlir) {
  TF_EXPECT_OK(CreateMlirModule());
  EXPECT_FALSE(IsLegalizedWithMlir(*module_->getOperation()));
}

}  // namespace mhlo
}  // namespace mlir
