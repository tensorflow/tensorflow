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

#include "tensorflow/compiler/mlir/tf2xla/api/v2/cluster_tf.h"

#include <cstdint>
#include <string>

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
#include "tensorflow/compiler/mlir/register_common_dialects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/resource_loader.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace tf2xla {
namespace v2 {
namespace {

using ::mlir::DialectRegistry;
using ::mlir::MLIRContext;
using ::mlir::ModuleOp;
using ::mlir::OwningOpRef;
using ::mlir::WalkResult;
using ::mlir::func::FuncOp;
using ::tensorflow::monitoring::testing::CellReader;

std::string TestDataPath() {
  return tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/tf2xla/api/v2/testdata/");
}

static constexpr char kCompilationStreamz[] =
    "/tensorflow/core/tf_mlir_bridge_first_phase_count";

class FunctionClusterTensorflowDialectTest : public ::testing::Test {
 public:
  FunctionClusterTensorflowDialectTest() {
    mlir::RegisterCommonToolingDialects(registry_);
    context_.appendDialectRegistry(registry_);
    context_.loadAllAvailableDialects();
  }

  tsl::Status CreateMlirModule(std::string mlir_module_filename) {
    std::string mlir_module_path = TestDataPath() + mlir_module_filename;
    mlir_module_ =
        mlir::parseSourceFile<mlir::ModuleOp>(mlir_module_path, &context_);
    if (!mlir_module_) {
      return tsl::Status(
          absl::StatusCode::kNotFound,
          absl::StrCat("Could not find MLIR module at ", mlir_module_path));
    }
    return tsl::OkStatus();
  }

  DialectRegistry registry_;
  MLIRContext context_;
  OwningOpRef<mlir::ModuleOp> mlir_module_;
};

TEST_F(FunctionClusterTensorflowDialectTest, ClustersTfTPU) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(
      RunFunctionTf2xlaClusteringBridge(*mlir_module_, DeviceType::XLA_TPU_JIT,
                                        /*is_in_fallback_enabled_mode=*/false));

  FuncOp main = mlir_module_->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);

  EXPECT_EQ(
      compilation_status.Delta("tpu", "v2", "fallback_disabled", "success"), 1);
}

TEST_F(FunctionClusterTensorflowDialectTest, RunsOutsideCompilationTPU) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("outside_compilation.mlir"));

  TF_EXPECT_OK(
      RunFunctionTf2xlaClusteringBridge(*mlir_module_, DeviceType::XLA_TPU_JIT,
                                        /*is_in_fallback_enabled_mode=*/false));

  FuncOp main = mlir_module_->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);

  bool has_cluster_op = false;
  main.walk([&](mlir::tf_device::ClusterFuncOp cluster_op) {
    has_cluster_op = true;
    return WalkResult::advance();
  });

  EXPECT_TRUE(has_cluster_op);
  EXPECT_EQ(
      compilation_status.Delta("tpu", "v2", "fallback_disabled", "success"), 1);
}

TEST_F(FunctionClusterTensorflowDialectTest, ClustersTFCPU) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(
      RunFunctionTf2xlaClusteringBridge(*mlir_module_, DeviceType::XLA_CPU_JIT,
                                        /*is_in_fallback_enabled_mode=*/false));

  FuncOp main = mlir_module_->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);

  EXPECT_EQ(
      compilation_status.Delta("cpu/gpu", "v2", "fallback_disabled", "success"),
      1);
}

TEST_F(FunctionClusterTensorflowDialectTest, ClustersTFGPU) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(
      RunFunctionTf2xlaClusteringBridge(*mlir_module_, DeviceType::XLA_GPU_JIT,
                                        /*is_in_fallback_enabled_mode=*/false));

  FuncOp main = mlir_module_->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);

  EXPECT_EQ(
      compilation_status.Delta("cpu/gpu", "v2", "fallback_disabled", "success"),
      1);
}

TEST_F(FunctionClusterTensorflowDialectTest, LogsFallbackMode) {
  CellReader<int64_t> compilation_status(kCompilationStreamz);

  TF_ASSERT_OK(CreateMlirModule("empty_func.mlir"));

  TF_EXPECT_OK(
      RunFunctionTf2xlaClusteringBridge(*mlir_module_, DeviceType::XLA_TPU_JIT,
                                        /*is_in_fallback_enabled_mode=*/true));

  EXPECT_EQ(
      compilation_status.Delta("tpu", "v2", "fallback_enabled", "success"), 1);
}

}  // namespace
}  // namespace v2
}  // namespace tf2xla
}  // namespace tensorflow
