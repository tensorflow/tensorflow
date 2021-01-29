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
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD3(GetPassState,
                     MlirOptimizationPassState(const DeviceSet* device_set,
                                               const ConfigProto& config_proto,
                                               const Graph& graph));
  MOCK_METHOD3(Run, Status(const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph));
};

class MlirGraphOptimizationPassTest : public Test {
 public:
  void Init(Status pass_run_result,
            const std::vector<MlirOptimizationPassState>& pass_states) {
    graph_ = std::make_unique<Graph>(OpRegistry::Global());

    int pass_priority = 0;
    for (const MlirOptimizationPassState& pass_state : pass_states) {
      auto optimization_pass =
          std::make_unique<NiceMock<MockMlirOptimizationPass>>();

      ON_CALL(*optimization_pass, GetPassState(_, _, _))
          .WillByDefault(Return(pass_state));
      ON_CALL(*optimization_pass, Run(_, _, _))
          .WillByDefault(Return(pass_run_result));
      MlirOptimizationPassRegistry::Global().Add(pass_priority++,
                                                 std::move(optimization_pass));
    }

    flib_.reset(new FunctionLibraryDefinition(graph_->flib_def()));
  }

  void TearDown() override {
    MlirOptimizationPassRegistry::Global().ClearPasses();
  }

  void verifyGraphUnchanged(const GraphDef& original_graph_def) {
// Proto matchers might be unavailable in the OSS.
#if defined(PLATFORM_GOOGLE)
    GraphDef resulted_graph_def;
    graph_->ToGraphDef(&resulted_graph_def);
    EXPECT_THAT(resulted_graph_def,
                ::testing::proto::IgnoringRepeatedFieldOrdering(
                    ::testing::EquivToProto(original_graph_def)));
#endif
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
  Init(Status(error::Code::ABORTED, "aborted"),
       {MlirOptimizationPassState::Enabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status(error::Code::ABORTED, "aborted"));
  verifyGraphUnchanged(original_graph_def);
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsShadow) {
  Init(Status(error::Code::ABORTED, "aborted"),
       {MlirOptimizationPassState::ShadowEnabled,
        MlirOptimizationPassState::ShadowEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status::OK());
  verifyGraphUnchanged(original_graph_def);
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassDoesNotFailShadow) {
  Init(Status::OK(), {MlirOptimizationPassState::Disabled,
                      MlirOptimizationPassState::ShadowEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status::OK());
  verifyGraphUnchanged(original_graph_def);
}

TEST_F(MlirGraphOptimizationPassTest,
       OptimizationPassFailsMixShadowAndEnabled) {
  Init(Status(error::Code::ABORTED, "aborted"),
       {MlirOptimizationPassState::Disabled, MlirOptimizationPassState::Enabled,
        MlirOptimizationPassState::ShadowEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                device_set_, config_proto_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status(error::Code::ABORTED, "aborted"));
}

TEST(MlirOptimizationPassRegistry, RegisterPassesWithTheSamePriorityFails) {
  MlirOptimizationPassRegistry::Global().Add(
      0, std::make_unique<NiceMock<MockMlirOptimizationPass>>());
  EXPECT_DEATH(MlirOptimizationPassRegistry::Global().Add(
                   0, std::make_unique<NiceMock<MockMlirOptimizationPass>>()),
               "Pass priority must be unique.");
}

}  // namespace tensorflow
