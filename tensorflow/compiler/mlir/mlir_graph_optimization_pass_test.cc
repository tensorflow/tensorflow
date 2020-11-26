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

#include "tensorflow/compiler/mlir/mlir_graph_optimization_pass.h"

#include <memory>

#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Test;

class MockMlirOptimizationPass : public MlirOptimizationPass {
 public:
  // MOCK_METHOD does not work on Windows build, using MOCK_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD2(IsEnabled,
                     bool(const ConfigProto& config_proto, const Graph& graph));
  MOCK_METHOD3(Run, Status(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph));
};

class MlirGraphOptimizationPassTest : public Test {
 public:
  void Init(MlirBridgeRolloutPolicy rollout_policy, Status pass_run_result) {
    graph_ = std::make_unique<Graph>(OpRegistry::Global());

    function_optimization_pass_ = MlirFunctionOptimizationPass(
        &MlirOptimizationPassRegistry::Global(),
        [rollout_policy](const Graph& graph, absl::optional<ConfigProto>) {
          return rollout_policy;
        });

    auto optimization_pass =
        std::make_unique<NiceMock<MockMlirOptimizationPass>>();

    EXPECT_CALL(*optimization_pass, IsEnabled(_, _))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(*optimization_pass, Run(_, _, _))
        .WillOnce(Return(pass_run_result));
    MlirOptimizationPassRegistry::Global().Add(0, std::move(optimization_pass));

    flib_.reset(new FunctionLibraryDefinition(graph_->flib_def()));
  }

  void TearDown() override {
    MlirOptimizationPassRegistry::Global().ClearPasses();
  }

  ConfigProto config_proto_;
  MlirFunctionOptimizationPass function_optimization_pass_;
  DeviceSet device_set_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> flib_;
  std::vector<std::string> control_ret_node_names_;
  bool control_rets_updated_{false};
};

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsNoShadow) {
  Init(MlirBridgeRolloutPolicy::kEnabledByUser,
       Status(error::Code::ABORTED, "aborted"));

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status(error::Code::ABORTED, "aborted"));

// Proto matchers might be unavailable.
#if defined(PLATFORM_GOOGLE)
  GraphDef resulted_graph_def;
  graph_->ToGraphDef(&resulted_graph_def);
  EXPECT_THAT(resulted_graph_def,
              ::testing::proto::IgnoringRepeatedFieldOrdering(
                  ::testing::EquivToProto(original_graph_def)));
#endif
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsShadow) {
  Init(MlirBridgeRolloutPolicy::kEnabledAfterGraphAnalysis,
       Status(error::Code::ABORTED, "aborted"));

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status::OK());

// Proto matchers might be unavailable.
#if defined(PLATFORM_GOOGLE)
  GraphDef resulted_graph_def;
  graph_->ToGraphDef(&resulted_graph_def);
  EXPECT_THAT(resulted_graph_def,
              ::testing::proto::IgnoringRepeatedFieldOrdering(
                  ::testing::EquivToProto(original_graph_def)));
#endif
}

}  // namespace tensorflow
