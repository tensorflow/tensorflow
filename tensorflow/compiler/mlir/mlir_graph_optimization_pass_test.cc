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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/function_optimization_registry.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::Test;

constexpr char kOk[] = "OK";
constexpr char kInvalidArgument[] = "INVALID_ARGUMENT";
constexpr char kSuccess[] = "kSuccess";
constexpr char kFailure[] = "kFailure";

class MockMlirOptimizationPass : public MlirOptimizationPass {
 public:
  MOCK_METHOD(llvm::StringRef, name, (), (const, override));
  MOCK_METHOD(MlirOptimizationPassState, GetPassState,
              (const DeviceSet* device_set, const ConfigProto& config_proto,
               const Graph& graph,
               const FunctionLibraryDefinition& function_library),
              (const, override));
  MOCK_METHOD(absl::Status, Run,
              (const std::string& function_name,
               const ConfigProto& config_proto, mlir::ModuleOp module,
               const Graph& graph,
               const FunctionLibraryDefinition& function_library),
              (override));
};

class MockMlirV1CompatOptimizationPass : public MlirV1CompatOptimizationPass {
 public:
  MOCK_METHOD(llvm::StringRef, name, (), (const, override));
  MOCK_METHOD(MlirOptimizationPassState, GetPassState,
              (const DeviceSet* device_set, const ConfigProto& config_proto,
               const Graph& graph,
               const FunctionLibraryDefinition& function_library),
              (const, override));
  MOCK_METHOD(absl::Status, Run,
              (const GraphOptimizationPassOptions& options,
               mlir::ModuleOp module),
              (override));
};

class ModifyMlirModulePass : public MlirOptimizationPass {
 public:
  explicit ModifyMlirModulePass(absl::Status run_status)
      : run_status_(run_status) {}
  MOCK_METHOD(llvm::StringRef, name, (), (const, override));
  MOCK_METHOD(MlirOptimizationPassState, GetPassState,
              (const DeviceSet* device_set, const ConfigProto& config_proto,
               const Graph& graph,
               const FunctionLibraryDefinition& function_library),
              (const, override));

  // Just modify MLIR module so that we can check whether original TF graph
  // has changed or not.
  absl::Status Run(const std::string& function_name,
                   const ConfigProto& config_proto, mlir::ModuleOp module,
                   const Graph& graph,
                   const FunctionLibraryDefinition& function_library) override {
    mlir::Builder b(module.getContext());
    auto producer = b.getNamedAttr("producer", b.getI32IntegerAttr(0));
    auto min_consumer = b.getNamedAttr("min_consumer", b.getI32IntegerAttr(0));
    auto bad_consumers =
        b.getNamedAttr("bad_consumers", b.getI32ArrayAttr({1, 2, 3, 4}));

    module->setAttr("tf.versions",
                    b.getDictionaryAttr(llvm::ArrayRef<mlir::NamedAttribute>(
                        {producer, min_consumer, bad_consumers})));

    return run_status_;
  }

  absl::Status run_status_;
};

FunctionDef XTimesTwo() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });
}

class MlirGraphOptimizationPassTest : public Test {
 public:
  void Init(absl::Status pass_run_result,
            const std::vector<MlirOptimizationPassState>& pass_states) {
    graph_ = std::make_unique<Graph>(OpRegistry::Global());

    int pass_priority = 0;
    for (const MlirOptimizationPassState& pass_state : pass_states) {
      auto optimization_pass =
          std::make_unique<NiceMock<MockMlirOptimizationPass>>();

      ON_CALL(*optimization_pass, GetPassState(_, _, _, _))
          .WillByDefault(Return(pass_state));
      ON_CALL(*optimization_pass, Run(_, _, _, _, _))
          .WillByDefault(Return(pass_run_result));
      MlirOptimizationPassRegistry::Global().Add(pass_priority++,
                                                 std::move(optimization_pass));
      pass_result_expected_[pass_state][pass_run_result.ok()]++;
    }

    flib_ = std::make_unique<FunctionLibraryDefinition>(graph_->flib_def());
  }

  void AddModuleModificationPass(MlirOptimizationPassState pass_state,
                                 absl::Status run_status) {
    // Add FallbackEnabled pass that modifies the graph.
    auto optimization_pass =
        std::make_unique<NiceMock<ModifyMlirModulePass>>(run_status);
    ON_CALL(*optimization_pass, GetPassState(_, _, _, _))
        .WillByDefault(Return(pass_state));
    MlirOptimizationPassRegistry::Global().Add(10,
                                               std::move(optimization_pass));
    pass_result_expected_[pass_state][run_status.ok()]++;
  }

  void TearDown() override {
    MlirOptimizationPassRegistry::Global().ClearPasses();
  }

  void verifyGraph(const GraphDef& original_graph_def, bool changed = false) {
// Proto matchers might be unavailable in the OSS.
#if defined(PLATFORM_GOOGLE)
    GraphDef resulted_graph_def;
    graph_->ToGraphDef(&resulted_graph_def);

    if (changed)
      EXPECT_THAT(resulted_graph_def,
                  Not(::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def))));
    else
      EXPECT_THAT(resulted_graph_def,
                  ::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def)));
#endif
  }

  void verifyCounters() {
    EXPECT_EQ(mlir_function_pass_fallback_count_.Read(kSuccess),
              pass_result_expected_[MlirOptimizationPassState::FallbackEnabled]
                                   [true]);
    EXPECT_EQ(mlir_function_pass_fallback_count_.Read(kFailure),
              pass_result_expected_[MlirOptimizationPassState::FallbackEnabled]
                                   [false]);
    EXPECT_EQ(mlir_function_pass_graph_conversion_count_.Read(kOk), 1);
  }

  ConfigProto config_proto_;
  FunctionOptimizationPass::FunctionOptions function_options_;
  MlirFunctionOptimizationPass function_optimization_pass_;
  DeviceSet device_set_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> flib_;
  std::vector<std::string> control_ret_node_names_;
  bool control_rets_updated_{false};
  monitoring::testing::CellReader<int64_t> mlir_function_pass_fallback_count_ =
      monitoring::testing::CellReader<int64_t>(
          /* metric name */
          "/tensorflow/core/mlir_function_pass_fallback_count");
  monitoring::testing::CellReader<int64_t>
      mlir_graph_optimization_pass_fallback_count_ =
          monitoring::testing::CellReader<int64_t>(
              /* metric name */
              "/tensorflow/core/mlir_graph_optimization_pass_fallback_count");
  monitoring::testing::CellReader<int64_t>
      mlir_function_pass_graph_conversion_count_ =
          monitoring::testing::CellReader<int64_t>(
              /* metric name */
              "/tensorflow/core/mlir_function_pass_graph_conversion_count");
  std::map<MlirOptimizationPassState, std::map<bool, int64_t>>
      pass_result_expected_;
};

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsNoFallback) {
  Init(absl::Status(absl::StatusCode::kAborted, "aborted"),
       {MlirOptimizationPassState::Enabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(
      function_optimization_pass_.Run(
          "test_func", device_set_, config_proto_, function_options_, &graph_,
          flib_.get(), &control_ret_node_names_, &control_rets_updated_),
      absl::Status(absl::StatusCode::kAborted, "aborted"));
  verifyGraph(original_graph_def);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsDisabledFallback) {
  Init(absl::Status(absl::StatusCode::kAborted, "aborted"),
       {MlirOptimizationPassState::Disabled,
        MlirOptimizationPassState::FallbackEnabled});

  // We expect the result graph to be exactly the same as the original graph
  // so we define the `graph_` by the following `flib` in this test point
  // instead of the way we do in the Init method.
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwo();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  graph_ = std::make_unique<Graph>(flib_def);

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);
  AddModuleModificationPass(
      MlirOptimizationPassState::FallbackEnabled,
      absl::Status(absl::StatusCode::kAborted, "aborted"));

  EXPECT_EQ(
      function_optimization_pass_.Run(
          "test_func", device_set_, config_proto_, function_options_, &graph_,
          flib_.get(), &control_ret_node_names_, &control_rets_updated_),
      absl::OkStatus());
  verifyGraph(original_graph_def);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassDoesNotFailFallback) {
  Init(absl::OkStatus(), {MlirOptimizationPassState::FallbackEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            absl::OkStatus());
  EXPECT_EQ(
      function_optimization_pass_.Run(
          "test_func", device_set_, config_proto_, function_options_, &graph_,
          flib_.get(), &control_ret_node_names_, &control_rets_updated_),
      absl::OkStatus());

  verifyGraph(original_graph_def, true);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, GraphDoesntConvertUpdatesCounter) {
  Init(absl::OkStatus(), {MlirOptimizationPassState::FallbackEnabled});

  graph_ = std::make_unique<Graph>(OpRegistry::Global());
  control_ret_node_names_.push_back("foo");

  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            absl::OkStatus());
  EXPECT_EQ(
      function_optimization_pass_.Run(
          "test_func", device_set_, config_proto_, function_options_, &graph_,
          flib_.get(), &control_ret_node_names_, &control_rets_updated_),
      absl::OkStatus());

  EXPECT_EQ(mlir_function_pass_graph_conversion_count_.Read(kOk), 0);
  EXPECT_EQ(mlir_function_pass_graph_conversion_count_.Read(kInvalidArgument),
            1);
}

TEST(MlirOptimizationPassRegistry, RegisterPassesWithTheSamePriorityFails) {
  MlirOptimizationPassRegistry::Global().Add(
      0, std::make_unique<NiceMock<MockMlirOptimizationPass>>());
  EXPECT_DEATH(MlirOptimizationPassRegistry::Global().Add(
                   0, std::make_unique<NiceMock<MockMlirOptimizationPass>>()),
               "Pass priority must be unique.");
}

TEST(MlirV1CompatOptimizationPassRegistry, RegisterMultiplePassesFails) {
  MlirV1CompatOptimizationPassRegistry::Global().Add(
      std::make_unique<NiceMock<MockMlirV1CompatOptimizationPass>>());
  EXPECT_DEATH(
      MlirV1CompatOptimizationPassRegistry::Global().Add(
          std::make_unique<NiceMock<MockMlirV1CompatOptimizationPass>>()),
      "Only a single pass can be registered");
}

class MlirGraphOptimizationV1PassTest : public Test {
 public:
  void Init(absl::Status pass_run_result,
            const std::vector<MlirOptimizationPassState>& pass_states) {
    graph_ = std::make_unique<Graph>(OpRegistry::Global());
    MlirV1CompatOptimizationPassRegistry::Global().ClearPass();
    for (const MlirOptimizationPassState& pass_state : pass_states) {
      auto optimization_pass =
          std::make_unique<NiceMock<MockMlirV1CompatOptimizationPass>>();
      ON_CALL(*optimization_pass, GetPassState(_, _, _, _))
          .WillByDefault(Return(pass_state));
      ON_CALL(*optimization_pass, Run(_, _))
          .WillByDefault(Return(pass_run_result));
      MlirV1CompatOptimizationPassRegistry::Global().Add(
          std::move(optimization_pass));
      pass_result_expected_[pass_state][pass_run_result.ok()]++;
    }
    flib_ = std::make_unique<FunctionLibraryDefinition>(graph_->flib_def());
    InitGraphOptions();
  }

  void verifyGraph(const GraphDef& original_graph_def, bool changed = false) {
// Proto matchers might be unavailable in the OSS.
#if defined(PLATFORM_GOOGLE)
    GraphDef resulted_graph_def;
    graph_->ToGraphDef(&resulted_graph_def);

    if (changed)
      EXPECT_THAT(resulted_graph_def,
                  Not(::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def))));
    else
      EXPECT_THAT(resulted_graph_def,
                  ::testing::proto::IgnoringRepeatedFieldOrdering(
                      ::testing::EquivToProto(original_graph_def)));
#endif
  }

  void InitGraphOptions() {
    session_options_.config = config_proto_;
    graph_optimization_pass_options_.device_set = &device_set_;
    graph_optimization_pass_options_.session_options = &session_options_;
    graph_optimization_pass_options_.graph = &graph_;
    graph_optimization_pass_options_.flib_def = flib_.get();
  }

  void verifyCounters() {
    EXPECT_EQ(mlir_function_pass_fallback_count_.Read(kSuccess),
              pass_result_expected_[MlirOptimizationPassState::FallbackEnabled]
                                   [false]);
    EXPECT_EQ(mlir_function_pass_fallback_count_.Read(kFailure),
              pass_result_expected_[MlirOptimizationPassState::FallbackEnabled]
                                   [false]);
    EXPECT_EQ(mlir_function_pass_graph_conversion_count_.Read(kOk), 0);
    EXPECT_EQ(mlir_v1_compat_graph_conversion_count_.Read(kOk), 1);
  }

  void TearDown() override {
    MlirV1CompatOptimizationPassRegistry::Global().ClearPass();
  }

  ConfigProto config_proto_;
  FunctionOptimizationPass::FunctionOptions function_options_;
  MlirV1CompatGraphOptimizationPass function_optimization_pass_;
  DeviceSet device_set_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> flib_;
  std::vector<std::string> control_ret_node_names_;
  bool control_rets_updated_{false};
  SessionOptions session_options_;
  tensorflow::GraphOptimizationPassOptions graph_optimization_pass_options_;
  std::map<MlirOptimizationPassState, std::map<bool, int64_t>>
      pass_result_expected_;
  monitoring::testing::CellReader<int64_t> mlir_function_pass_fallback_count_ =
      monitoring::testing::CellReader<int64_t>(
          /* metric name */
          "/tensorflow/core/mlir_function_pass_fallback_count");
  monitoring::testing::CellReader<int64_t>
      mlir_graph_optimization_pass_fallback_count_ =
          monitoring::testing::CellReader<int64_t>(
              /* metric name */
              "/tensorflow/core/mlir_graph_optimization_pass_fallback_count");
  monitoring::testing::CellReader<int64_t>
      mlir_function_pass_graph_conversion_count_ =
          monitoring::testing::CellReader<int64_t>(
              /* metric name */
              "/tensorflow/core/mlir_function_pass_graph_conversion_count");
  monitoring::testing::CellReader<int64_t>
      mlir_v1_compat_graph_conversion_count_ =
          monitoring::testing::CellReader<int64_t>(
              /* metric name */
              "/tensorflow/core/mlir_v1_compat_graph_conversion_count");
};

TEST_F(MlirGraphOptimizationV1PassTest, OptimizationPassDoesNotFailFallback) {
  Init(absl::OkStatus(), {MlirOptimizationPassState::FallbackEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(graph_optimization_pass_options_),
            absl::OkStatus());

  verifyGraph(original_graph_def, /*changed=*/false);
  verifyCounters();
}

}  // namespace tensorflow
