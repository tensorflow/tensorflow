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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/test.h"

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
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));
  MOCK_METHOD5(Run, Status(const std::string& function_name,
                           const ConfigProto& config_proto,
                           mlir::ModuleOp module, const Graph& graph,
                           const FunctionLibraryDefinition& function_library));
};

class MockMlirV1CompatOptimizationPass : public MlirV1CompatOptimizationPass {
 public:
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));
  MOCK_METHOD2(Run, Status(const GraphOptimizationPassOptions& options,
                           mlir::ModuleOp module));
};

class ModifyMlirModulePass : public MlirOptimizationPass {
 public:
  explicit ModifyMlirModulePass(Status run_status) : run_status_(run_status) {}
  // MOCK_METHOD does not work on Windows build, using MOCK_CONST_METHODX
  // instead.
  MOCK_CONST_METHOD0(name, llvm::StringRef());
  MOCK_CONST_METHOD4(GetPassState,
                     MlirOptimizationPassState(
                         const DeviceSet* device_set,
                         const ConfigProto& config_proto, const Graph& graph,
                         const FunctionLibraryDefinition& function_library));

  // Just modify MLIR module so that we can check whether original TF graph
  // has changed or not.
  Status Run(const std::string& function_name, const ConfigProto& config_proto,
             mlir::ModuleOp module, const Graph& graph,
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

  Status run_status_;
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
  void Init(Status pass_run_result,
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
                                 Status run_status) {
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
  MlirFunctionOptimizationPass function_optimization_pass_;
  DeviceSet device_set_;
  std::unique_ptr<Graph> graph_;
  std::unique_ptr<FunctionLibraryDefinition> flib_;
  std::vector<std::string> control_ret_node_names_;
  std::string xla_compile_device_type_;
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
  Init(Status(absl::StatusCode::kAborted, "aborted"),
       {MlirOptimizationPassState::Enabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  EXPECT_EQ(function_optimization_pass_.Run(
                "test_func", device_set_, config_proto_,
                xla_compile_device_type_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            Status(absl::StatusCode::kAborted, "aborted"));
  verifyGraph(original_graph_def);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassFailsDisabledFallback) {
  Init(Status(absl::StatusCode::kAborted, "aborted"),
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
  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            Status(absl::StatusCode::kAborted, "aborted"));

  EXPECT_EQ(function_optimization_pass_.Run(
                "test_func", device_set_, config_proto_,
                xla_compile_device_type_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            OkStatus());
  verifyGraph(original_graph_def);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, OptimizationPassDoesNotFailFallback) {
  Init(OkStatus(), {MlirOptimizationPassState::FallbackEnabled});

  GraphDef original_graph_def;
  graph_->ToGraphDef(&original_graph_def);

  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            OkStatus());
  EXPECT_EQ(function_optimization_pass_.Run(
                "test_func", device_set_, config_proto_,
                xla_compile_device_type_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            OkStatus());

  verifyGraph(original_graph_def, true);
  verifyCounters();
}

TEST_F(MlirGraphOptimizationPassTest, GraphDoesntConvertUpdatesCounter) {
  Init(OkStatus(), {MlirOptimizationPassState::FallbackEnabled});

  graph_ = std::make_unique<Graph>(OpRegistry::Global());
  control_ret_node_names_.push_back("foo");

  AddModuleModificationPass(MlirOptimizationPassState::FallbackEnabled,
                            OkStatus());
  EXPECT_EQ(function_optimization_pass_.Run(
                "test_func", device_set_, config_proto_,
                xla_compile_device_type_, &graph_, flib_.get(),
                &control_ret_node_names_, &control_rets_updated_),
            OkStatus());

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

}  // namespace tensorflow
