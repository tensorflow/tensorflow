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

#include "tensorflow/core/common_runtime/function_optimization_registry.h"

#include <memory>
#include <string>

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class PassingFunctionPass : public FunctionOptimizationPass {
 public:
  static bool ran_;

  absl::Status Run(const std::string& function_name,
                   const DeviceSet& device_set, const ConfigProto& config_proto,
                   const FunctionOptions& function_options,
                   std::unique_ptr<Graph>* graph,
                   FunctionLibraryDefinition* flib_def,
                   std::vector<std::string>* control_ret_node_names,
                   bool* control_rets_updated) override {
    ran_ = true;
    return absl::OkStatus();
  }
};

bool PassingFunctionPass::ran_ = false;

TEST(FunctionOptimizationPassRegistry, PassNoError) {
  EXPECT_FALSE(PassingFunctionPass::ran_);

  FunctionOptimizationPassRegistry::Global().Init(
      std::make_unique<PassingFunctionPass>());
  DeviceSet device_set;
  ConfigProto config_proto;
  FunctionOptimizationPass::FunctionOptions function_options;
  absl::Status status = FunctionOptimizationPassRegistry::Global().Run(
      "test_func", device_set, config_proto, function_options,
      /*graph=*/nullptr,
      /*flib_def=*/nullptr,
      /*control_ret_node_names=*/nullptr, /*control_rets_updated=*/nullptr);

  EXPECT_EQ(status, absl::OkStatus());
  EXPECT_TRUE(PassingFunctionPass::ran_);
}

}  // namespace tensorflow
