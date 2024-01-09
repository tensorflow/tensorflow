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

#include <memory>

#include <gtest/gtest.h>
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/test_utils.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

constexpr char kJitCompileSingleCoreTpuCount[] =
    "/tensorflow/core/jit_compile_single_core_tpu_count";
constexpr char kUseMlirBridge[] = "kUseMlirBridge";
using mlir::mhlo::test::GetMlirModuleFromString;

class TPUClusterFormationPassTest : public testing::Test {
 protected:
  void CreateModule(const char* module_string) {
    TF_ASSERT_OK_AND_ASSIGN(module_,
                            GetMlirModuleFromString(module_string, &context_));
    bool strict_clusters = true;
    pm_ = std::make_unique<mlir::PassManager>(&context_);
    pm_->addPass(tensorflow::tf2xla::internal::CreateTPUClusterFormationPass(
        strict_clusters));
  }

  mlir::LogicalResult Run() { return pm_->run(module_.get()); }

 private:
  mlir::MLIRContext context_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  std::unique_ptr<mlir::PassManager> pm_;
};

TEST_F(TPUClusterFormationPassTest, NonReplicatedTPU) {
  monitoring::testing::CellReader<int64_t> feature_metric_reader(
      kJitCompileSingleCoreTpuCount);
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @valid_compilation_cluster_no_replication() {
      "tf.opA"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
      "tf.opB"() { _xla_compile_device_type = "TPU", is_stateless = true} : () -> ()
      func.return
    }
  })";
  CreateModule(kMlirModuleStr);
  auto result = Run();
  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(feature_metric_reader.Delta(kUseMlirBridge), 1);
}

TEST_F(TPUClusterFormationPassTest, ReplicatedTPU) {
  monitoring::testing::CellReader<int64_t> feature_metric_reader(
      kJitCompileSingleCoreTpuCount);
  static constexpr char kMlirModuleStr[] = R"(
  module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
    func.func @interleaved_clusters(%arg0 : tensor<i1>) -> (tensor<i1>, tensor<i1>) {
      "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", device = "device_1", num_replicas = 1, topology = "topology_1"} : () -> ()
      %0 = "tf.opA"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", is_stateless = true} : (tensor<i1>) -> tensor<i1>
      %1 = "tf.opB"(%arg0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", is_stateless = true} : (tensor<i1>) -> tensor<i1>
      %2 = "tf.opC"(%0) {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", is_stateless = true} : (tensor<i1>) -> tensor<i1>
      %3 = "tf.opD"(%1) {_xla_compile_device_type = "TPU", _replication_info = "replicate_1", is_stateless = true} : (tensor<i1>) -> tensor<i1>
      "tf.TPUReplicateMetadata"() {_xla_compile_device_type = "TPU", _replication_info = "replicate_0", device = "device_0", num_replicas = 1, topology = "topology_0"} : () -> ()
      func.return %2, %3 : tensor<i1>, tensor<i1>
    }
  })";
  CreateModule(kMlirModuleStr);
  auto result = Run();
  EXPECT_TRUE(result.succeeded());
  EXPECT_EQ(feature_metric_reader.Delta(kUseMlirBridge), 0);
}

}  // namespace
}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
