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
#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"
#include "tsl/platform/statusor.h"

namespace {

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetMlirModuleFromString(
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

TEST(XLAShardingUtilTest, TestShapesCheckForSplitSharding) {
  // Module with sharding with static shapes.
  static const char* const module_str =
      R"(
      module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
   func.func @parallel_execute_with_tiled_input(%arg0: tensor<128x9xf32>, %arg1: tensor<128x9xf32>, %arg2: tensor<128x10xi32>, %arg3: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x9xf32>, [%arg2, %arg3] as %ri_2: tensor<128x10xi32>) {n = 2 : i32} {
      %1 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
        %identity = "tf.Identity"(%ri_1) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<128x9xf32>) -> tensor<128x9xf32>
        tf_device.return %identity : tensor<128x9xf32>
      }) {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<128x9xf32>
      %2, %3 = "tf_device.cluster_func"(%1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x9xf32>, tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
      tf_device.return %2, %3 : tensor<128x10xi32>, tensor<10x5xi1>
    }
    func.return %0#0, %1#0 : tensor<128x10xi32>, tensor<10x5xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x9xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x9xf32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<128x10xi32>, tensor<128x10xi32>) -> (tensor<128x10xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<10x5xi1>) -> tensor<10x5xi1>
    func.return %4, %3 : tensor<128x10xi32>, tensor<10x5xi1>
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
  auto cluster_func_op = cluster_func_ops[0];

  llvm::SmallVector<xla::OpSharding, 4> output_shardings;
  auto result = tensorflow::ParseAndValidateOutputSharding(2, cluster_func_op,
                                                           &output_shardings);
  ASSERT_TRUE(succeeded(result));
  ASSERT_TRUE(tensorflow::AreInputOutputShapesStaticallyKnownForSplitSharding(
      output_shardings, cluster_func_op));
}

TEST(XLAShardingUtilTest, TestShapesCheckForSplitShardingWithUnknownShapes) {
  // Module with sharding with unknown shapes
  static const char* const module_str =
      R"(
      module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
   func.func @parallel_execute_with_tiled_input(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<128x10xi32>, %arg3: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<*xf32>, [%arg2, %arg3] as %ri_2: tensor<128x10xi32>) {n = 2 : i32} {
      %1 = "tf_device.launch"() <{device = "TPU_REPLICATED_HOST_0"}> ({
        %identity = "tf.Identity"(%ri_1) {ici_weight_distribution_mlir_bridge_marker = true} : (tensor<*xf32>) -> tensor<*xf32>
        tf_device.return %identity : tensor<*xf32>
      }) {ici_weight_distribution_mlir_bridge_marker = true} : () -> tensor<*xf32>
      %2, %3 = "tf_device.cluster_func"(%1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\1A\02\01\02\22\02\00\01", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<*xf32>, tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
      tf_device.return %2, %3 : tensor<128x10xi32>, tensor<10x5xi1>
    }
    func.return %0#0, %1#0 : tensor<128x10xi32>, tensor<10x5xi1>
  }
  func.func @tpu0_func(%arg0: tensor<*xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128x10xi32>, tensor<10x5xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<*xf32>) -> (tensor<128x10xi32>, tensor<10x5xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<128x10xi32>, tensor<128x10xi32>) -> (tensor<128x10xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<10x5xi1>) -> tensor<10x5xi1>
    func.return %4, %3 : tensor<128x10xi32>, tensor<10x5xi1>
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
  auto cluster_func_op = cluster_func_ops[0];

  llvm::SmallVector<xla::OpSharding, 4> output_shardings;
  auto result = tensorflow::ParseAndValidateOutputSharding(2, cluster_func_op,
                                                           &output_shardings);
  ASSERT_TRUE(succeeded(result));
  ASSERT_FALSE(tensorflow::AreInputOutputShapesStaticallyKnownForSplitSharding(
      output_shardings, cluster_func_op));
}

TEST(XLAShardingUtilTest, NotDivisibleShardingSplitOpTest) {
  static const char* const module_str =
      R"(
  module attributes {tf.versions = {producer = 888 : i32}, tf.devices = ["/job:localhost/replica:0/task:0/device:CPU:0", "/job:localhost/replica:0/task:0/device:TPU:0", "/job:localhost/replica:0/task:0/device:TPU:1", "/job:localhost/replica:0/task:0/device:TPU_SYSTEM:0", "/job:localhost/replica:0/task:1/device:CPU:0", "/job:localhost/replica:0/task:1/device:TPU:0", "/job:localhost/replica:0/task:1/device:TPU:1", "/job:localhost/replica:0/task:1/device:TPU_SYSTEM:0"]} {
  func.func @uneven_input_sharding_disallowed(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xf32>, %arg2: tensor<*xi32>, %arg3: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %0:2, %1:2 = tf_device.replicate([%arg0, %arg1] as %ri_1: tensor<128x10xf32>, [%arg2, %arg3] as %ri_2: tensor<*xi32>) {n = 2 : i32} {
    %1, %2 = "tf_device.cluster_func"(%ri_1, %ri_2) {_xla_compile_device_type = "TPU", _replication_info = "cluster0", func = @tpu0_func, num_cores_per_replica = 2, step_marker_location = "STEP_MARK_AT_TOP_LEVEL_WHILE_LOOP", topology = "\0A\04\01\02\01\02\10\02\18\02\22\10\00\00\00\00\00\00\00\01\00\01\00\00\00\01\00\01", device_assignment = [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], input_sharding_configuration = ["\08\03\12\12\10\0b\1a\02\01\04\2a\06\0a\02\01\00\20\01\32\02\00\00\1a\02\01\04\22\04\00\01\02\03", "\08\01\1A\01\01\22\01\01"], output_sharding_configuration = ["\08\01\1A\01\01\22\01\00", ""], use_spmd_for_xla_partitioning = false} : (tensor<128x10xf32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>)
      tf_device.return %1, %2 : tensor<*xi32>, tensor<*xi1>
    }
    func.return %0#0, %1#0 : tensor<*xi32>, tensor<*xi1>
  }
  func.func @tpu0_func(%arg0: tensor<128x10xf32>, %arg1: tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi1>) {
    %1, %2 = "tf.A"(%arg0) : (tensor<128x10xf32>) -> (tensor<*xi32>, tensor<*xi1>)
    %4 = "tf.B"(%1, %arg1) : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
    %3 = "tf.XlaSharding"(%2) { _XlaSharding = "", sharding = "" } : (tensor<*xi1>) -> tensor<*xi1>
    func.return %4, %3 : tensor<*xi32>, tensor<*xi1>
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
  auto& cluster_func_op = cluster_func_ops[0];
  int num_cores_per_replica = 4;
  mlir::OpBuilder builder(&context);
  bool use_xla_nd_ops = true;

  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> input_list;
  auto result = tensorflow::ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func_op, &builder, use_xla_nd_ops,
      &input_list);
  ASSERT_TRUE(succeeded(result));

  ASSERT_EQ(input_list.size(), num_cores_per_replica);
  ASSERT_GT(input_list.front().size(), 0);
  // Get the XLASplit op generated.
  auto* op = input_list.front().front().getDefiningOp();
  ASSERT_TRUE(mlir::isa<mlir::TF::XlaSplitNDOp>(op));
  // Erase newly created Split op to avoid error during block deletion:
  // use_empty() && "Cannot destroy a value that still has uses!"
  //
  // This is needed only for the unit test.
  //
  // More explanation:
  // We create the XLASplit op in ExtractInputsForLogicalDevices function and
  // use the getResults for the op are added to input list.
  // Somehow the values from XLASplitND op are not getting released correctly in
  // the destructor for the block at the end of the test function.
  // This results in "use_empty() && 'Cannot destroy a value that still has
  // uses!'" error when the builder and block ops arr deleted when they go out
  //  of scope. Thus, added Op->destroy for the XLA split to remove it from the
  // MLIR block so the above assertion does not occur. We need this only in the
  // test code, it is not a problem in the actual pass since tpu_rewrite_pass
  // will appropriately add the values to the block.
  op->destroy();

  input_list.clear();

  // Expect error when use_xla_nd_ops is false.
  result = tensorflow::ExtractInputsForLogicalDevices(
      num_cores_per_replica, cluster_func_op, &builder, false, &input_list);
  ASSERT_TRUE(succeeded(result));
  auto* split_op = input_list.front().front().getDefiningOp();
  ASSERT_TRUE(mlir::isa<mlir::TF::SplitOp>(split_op));

  llvm::SmallVector<mlir::Value, 4> split_inputs(split_op->getOperands());
  // Constant op for the split dimension
  auto* const_op = split_inputs[0].getDefiningOp();
  ASSERT_TRUE(mlir::isa<mlir::TF::ConstOp>(const_op));
  // Pad op for the padding value to make it divisible by num_splits.
  auto* pad_op = split_inputs[1].getDefiningOp();
  ASSERT_TRUE(mlir::isa<mlir::TF::PadOp>(pad_op));
  llvm::SmallVector<mlir::Value, 4> pad_inputs(pad_op->getOperands());
  auto* const_pad_value = pad_inputs[1].getDefiningOp();
  ASSERT_TRUE(mlir::isa<mlir::TF::ConstOp>(const_pad_value));
  // Destroy the ops to avoid error during block deletion (Same as above):
  // use_empty() && "Cannot destroy a value that still has uses!"
  split_op->destroy();
  const_op->destroy();
  pad_op->destroy();
  const_pad_value->destroy();
}
}  // namespace
