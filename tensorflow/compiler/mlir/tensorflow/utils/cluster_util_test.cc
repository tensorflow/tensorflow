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

#include "tensorflow/compiler/mlir/tensorflow/utils/cluster_util.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"

namespace mlir::TF {

namespace {

constexpr StringRef kTestClusterName = "tpu0";

tsl::StatusOr<OwningOpRef<ModuleOp>> GetMlirModuleFromString(
    StringRef string, MLIRContext* context) {
  DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);
  OwningOpRef<ModuleOp> mlir_module;
  auto status =
      tensorflow::DeserializeMlirModule(string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

std::string GetDevice(Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return device_attr ? device_attr.getValue().str() : "";
}

bool CanBeIgnoredInCluster(Operation* op) {
  auto device_attr = op->getAttrOfType<StringAttr>("device");
  return !device_attr || device_attr.getValue().empty();
}

llvm::StringMap<SmallVector<Cluster>> GetClusters(ModuleOp module) {
  TF::SideEffectAnalysis side_effect_analysis(module);
  auto main_func = module.lookupSymbol<func::FuncOp>("main");
  const TF::SideEffectAnalysis::Info& info =
      side_effect_analysis.GetAnalysisForFunc(main_func);
  llvm::StringMap<SmallVector<Cluster>> clusters = BuildAllClusters(
      main_func.front(), info, GetDevice, CanBeIgnoredInCluster);
  return clusters;
}

TEST(BuildClusters, TestSingleCluster) {
  static const char* const module_with_single_cluster =
      R"(module {
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.B"(%0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.C"(%0, %1) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.D"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    func.return %3 : tensor<?xi32>
  }
}
)";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      GetMlirModuleFromString(module_with_single_cluster, &context));
  auto clusters = GetClusters(module.get());
  EXPECT_EQ(clusters.count(kTestClusterName), 1);
  EXPECT_EQ(clusters.lookup(kTestClusterName).size(), 1);
  EXPECT_EQ(clusters.lookup(kTestClusterName)[0].ops.size(), 2);
}

TEST(BuildClusters, TestMultipleClusters) {
  static const char* const module_with_two_clusters =
      R"(module {
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.B"(%0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.C"(%0, %1) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.D"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    %4 = "tf.E"(%3) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %5 = "tf.F"(%3, %4) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    func.return %5 : tensor<?xi32>
  }
}
)";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      GetMlirModuleFromString(module_with_two_clusters, &context));
  auto clusters = GetClusters(module.get());
  EXPECT_EQ(clusters.count(kTestClusterName), 1);
  EXPECT_EQ(clusters[kTestClusterName].size(), 2);
  EXPECT_EQ(clusters[kTestClusterName][0].ops.size(), 2);
  EXPECT_EQ(clusters[kTestClusterName][1].ops.size(), 2);
}

TEST(BuildClusters, TestMultipleTargets) {
  static const char* const module_with_two_clusters =
      R"(module {
func.func @main(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %0 = "tf.A"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.B"(%0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.C"(%0, %1) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.D"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    %4 = "tf.E"(%3) {device = "tpu1"} : (tensor<?xi32>) -> tensor<?xi32>
    %5 = "tf.F"(%3, %4) {device = "tpu1"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    func.return %5 : tensor<?xi32>
  }
}
)";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      GetMlirModuleFromString(module_with_two_clusters, &context));
  auto clusters = GetClusters(module.get());
  constexpr StringRef kTarget0 = "tpu0";
  EXPECT_EQ(clusters.count(kTarget0), 1);
  EXPECT_EQ(clusters[kTarget0].size(), 1);
  EXPECT_EQ(clusters[kTarget0][0].ops.size(), 2);

  constexpr StringRef kTarget1 = "tpu1";
  EXPECT_EQ(clusters.count(kTarget1), 1);
  EXPECT_EQ(clusters[kTarget1].size(), 1);
  EXPECT_EQ(clusters[kTarget1][0].ops.size(), 2);
}

TEST(BuildClusters, TestMergedClusters) {
  static const char* const module_with_single_cluster =
      R"(module {
func.func @main(%arg0: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
    %0 = "tf.Relu"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.Relu"(%0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.Add"(%0, %1) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.Relu"(%2) : (tensor<?xi32>) -> tensor<?xi32>
    %4 = "tf.Relu"(%1) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %5 = "tf.Add"(%1, %2) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    func.return %3, %5 : tensor<?xi32>, tensor<?xi32>
  }
}
)";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      GetMlirModuleFromString(module_with_single_cluster, &context));
  auto clusters = GetClusters(module.get());
  EXPECT_EQ(clusters.count(kTestClusterName), 1);
  EXPECT_EQ(clusters[kTestClusterName].size(), 1);
  EXPECT_EQ(clusters[kTestClusterName][0].ops.size(), 4);
}

TEST(BuildClusters, TestMergedClustersWithDataDependen) {
  static const char* const module_with_single_cluster =
      R"(module {
func.func @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> (tensor<?xi32>, tensor<?xi32>) {
    %0 = "tf.Relu"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
    %1 = "tf.Relu"(%0) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %2 = "tf.Add"(%0, %1) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %3 = "tf.Relu"(%arg1) {device = "tpu1"} : (tensor<?xi32>) -> tensor<?xi32>
    %4 = "tf.Add"(%3, %arg1) {device = "tpu1"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    %5 = "tf.Relu"(%4) {device = "tpu0"} : (tensor<?xi32>) -> tensor<?xi32>
    %6 = "tf.Add"(%4, %5) {device = "tpu0"} : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
    func.return %3, %5 : tensor<?xi32>, tensor<?xi32>
  }
}
)";

  MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(
      OwningOpRef<ModuleOp> module,
      GetMlirModuleFromString(module_with_single_cluster, &context));
  auto clusters = GetClusters(module.get());
  EXPECT_EQ(clusters.count(kTestClusterName), 1);
  EXPECT_EQ(clusters[kTestClusterName].size(), 1);
  EXPECT_EQ(clusters[kTestClusterName][0].ops.size(), 4);
}

}  // namespace

}  // namespace mlir::TF
