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

#include "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/lower_cluster_to_runtime_ops.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/tsl/framework/device_type.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/util/debug_data_dumper.h"

namespace tensorflow {
namespace tfrt_compiler {
namespace {

using mlir::DialectRegistry;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OwningOpRef;
using mlir::func::FuncOp;
using ::tensorflow::monitoring::testing::CellReader;
using tsl::DeviceType;

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tensorflow/transforms/host_runtime/testdata/");
}

static constexpr char kCompilationStreamz[] =
    "/tensorflow/core/tf_mlir_bridge_first_phase_v2_count";

class LowerClusterToRuntimeOpsTest : public ::testing::Test {
 public:
  LowerClusterToRuntimeOpsTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();

    env_ = Env::Default();
    test_group_name_ = "TestGroup";
    test_dir_ = testing::TmpDir();
    setenv(/*name=*/"TF_DUMP_GRAPH_PREFIX", /*value=*/test_dir_.c_str(),
           /*overwrite=*/1);
  }

  absl::Status CreateMlirModule(std::string mlir_module_filename) {
    std::string mlir_module_path = TestDataPath() + mlir_module_filename;
    mlir_module_ =
        mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
    if (!mlir_module_) {
      return absl::Status(
          absl::StatusCode::kNotFound,
          absl::StrCat("Could not find MLIR module at ", mlir_module_path));
    }
    return absl::OkStatus();
  }

  DialectRegistry registry_;
  MLIRContext context_;
  OwningOpRef<mlir::ModuleOp> mlir_module_;

  Env* env_;
  std::string test_dir_;
  std::string test_group_name_;
};

TEST_F(LowerClusterToRuntimeOpsTest, SanityCheck) {
  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(RunLowerClusterToRuntimeOpsPassPipeline(
      *mlir_module_, DeviceType(DEVICE_TPU_XLA_JIT)));
}

TEST_F(LowerClusterToRuntimeOpsTest, LowersClusterOpsTPU) {
  TF_ASSERT_OK(CreateMlirModule("basic_cluster.mlir"));

  TF_EXPECT_OK(RunLowerClusterToRuntimeOpsPassPipeline(
      *mlir_module_, DeviceType(DEVICE_TPU_XLA_JIT)));

  FuncOp main = mlir_module_->lookupSymbol<FuncOp>("main");
  ASSERT_TRUE(main);

  bool has_cluster_op = false;
  main.walk([&](mlir::tf_device::ClusterOp) {
    has_cluster_op = true;
    return mlir::WalkResult::interrupt();
  });

  EXPECT_FALSE(has_cluster_op);
}

TEST_F(LowerClusterToRuntimeOpsTest, LowersClusterOpsCPU) {
  TF_ASSERT_OK(CreateMlirModule("basic_cluster.mlir"));

  TF_EXPECT_OK(RunLowerClusterToRuntimeOpsPassPipeline(
      *mlir_module_, DeviceType(DEVICE_CPU_XLA_JIT)));

  FuncOp main = mlir_module_->lookupSymbol<FuncOp>("main");
  ASSERT_TRUE(main);

  bool has_cluster_op = false;
  main.walk([&](mlir::tf_device::ClusterOp) {
    has_cluster_op = true;
    return mlir::WalkResult::interrupt();
  });

  EXPECT_FALSE(has_cluster_op);
}

TEST_F(LowerClusterToRuntimeOpsTest, LowersClusterOpsGPU) {
  TF_ASSERT_OK(CreateMlirModule("basic_cluster.mlir"));

  TF_EXPECT_OK(RunLowerClusterToRuntimeOpsPassPipeline(
      *mlir_module_, DeviceType(DEVICE_GPU_XLA_JIT)));

  FuncOp main = mlir_module_->lookupSymbol<FuncOp>("main");
  ASSERT_TRUE(main);

  bool has_cluster_op = false;
  main.walk([&](mlir::tf_device::ClusterOp) {
    has_cluster_op = true;
    return mlir::WalkResult::interrupt();
  });

  EXPECT_FALSE(has_cluster_op);
}

TEST_F(LowerClusterToRuntimeOpsTest, ErrorsWithBadCluster) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("malformed_cluster.mlir"));

  EXPECT_FALSE(RunLowerClusterToRuntimeOpsPassPipeline(
                   *mlir_module_, DeviceType(DEVICE_TPU_XLA_JIT))
                   .ok());

  EXPECT_EQ(
      compilation_status.Delta(mlir::TF::kMlirPh1BridgeCounterReplicated,
                               mlir::TF::kMlirPh1BridgeCounterV2, "XLA_TPU_JIT",
                               "fallback_disabled", "failure"),
      1);
}

TEST_F(LowerClusterToRuntimeOpsTest, DumpsPipelinePasses) {
  std::vector<std::string> files;
  TF_ASSERT_OK(env_->GetChildren(test_dir_, &files));
  EXPECT_THAT(files, ::testing::IsEmpty());
  setenv(/*name=*/"TF_DUMP_GRAPH_NAME_FILTER", /*value=*/"*", /*overwrite=*/1);
  setenv(/*name=*/"TF_DUMP_GRAPH_GROUPS", /*value=*/"main,runtime_lowering",
         /*overwrite=*/1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  TF_ASSERT_OK(CreateMlirModule("basic_cluster.mlir"));

  TF_EXPECT_OK(RunLowerClusterToRuntimeOpsPassPipeline(
      *mlir_module_, DeviceType(DEVICE_TPU_XLA_JIT)));

  TF_ASSERT_OK(env_->GetChildren(test_dir_, &files));
  EXPECT_THAT(files, ::testing::SizeIs(15));
}

}  // namespace
}  // namespace tfrt_compiler
}  // namespace tensorflow
