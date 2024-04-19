/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/xla_rewrite_util.h"

#include <string>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/device_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

// #include <gmock/gmock.h>
// #include <gtest/gtest.h>

namespace tensorflow {
namespace {
tsl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetMlirModuleFromString(
    llvm::StringRef string, mlir::MLIRContext* context) {
  mlir::DialectRegistry mlir_registry;
  RegisterAllTensorFlowDialects(mlir_registry);
  context->appendDialectRegistry(mlir_registry);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  auto status =
      tensorflow::DeserializeMlirModule(string, context, &mlir_module);
  if (!status.ok()) {
    return status;
  }
  return mlir_module;
}

TEST(XlaRewriteUtilTest, TestEraseClusterFuncs) {
  static const char* const module_str =
      R"(
module attributes {tf.devices = ["/job:worker/replica:0/task:0/device:CPU:0", "/job:worker/replica:0/task:0/device:GPU:0"]} {
  func.func @convert_cluster_func(%arg0: tensor<i32>) -> () {
    %2 = "tf_device.parallel_execute"() ({

      %3 = "tf_device.cluster_func"(%arg0) {device = "/job:localhost/replica:0/task:0/device:GPU:0", func = @func} : (tensor<i32>) -> tensor<i32>

      tf_device.return %3 : tensor<i32>

    }) : () -> tensor<i32>
    return
  }
  func.func @func(%arg0: tensor<i32>) -> tensor<i32> {
    return %arg0 : tensor<i32>
  }
}
)";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          GetMlirModuleFromString(module_str, &context));
  llvm::SmallVector<mlir::tf_device::ClusterFuncOp, 4> cluster_func_ops;
  module->walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
    cluster_func_ops.push_back(cluster_func);
  });
  EXPECT_EQ(cluster_func_ops.size(), 1);

  EXPECT_TRUE(mlir::succeeded(tensorflow::EraseClusterFuncs(cluster_func_ops)));

  llvm::SmallVector<mlir::tf_device::ClusterFuncOp, 4> new_cluster_func_ops;
  module->walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
    new_cluster_func_ops.push_back(cluster_func);
  });
  EXPECT_EQ(new_cluster_func_ops.size(), 0);
}

TEST(XlaRewriteUtilTest, TestWrapOpInLaunch) {
  static const char* const module_str =
      R"(
module attributes {tf.devices = {"/job:localhost/replica:0/task:0/device:CPU:0"}} {
  func.func @main() -> () {
    "tf_device.cluster"() ({
      tf_device.return
    }) {} : () -> ()
    func.return
  }
})";
  mlir::MLIRContext context;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          GetMlirModuleFromString(module_str, &context));
  mlir::tf_device::ClusterOp cluster;
  std::string device = "/job:localhost/replica:0/task:0/device:CPU:0";
  module->walk(
      [&](mlir::tf_device::ClusterOp descendant) { cluster = descendant; });
  mlir::OpBuilder builder(&context);
  auto loc = cluster->getLoc();

  // Wrap the cluster op into a Launch op
  auto launch_op = tensorflow::WrapOpInLaunch(&builder, loc, cluster, device);

  EXPECT_TRUE(llvm::isa<mlir::tf_device::LaunchOp>(launch_op));
  launch_op->erase();
}

}  // namespace
}  // namespace tensorflow
