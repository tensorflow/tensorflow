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

#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"

namespace mlir {
namespace TFTPU {
namespace {

// Pass that co-locates resource ops that use composite device resources
// (packed tensors) with the underlying physical TPU device.
struct TPUColocateCompositeResourceOps
    : public PassWrapper<TPUColocateCompositeResourceOps, FunctionPass> {
  void runOnFunction() override;
};

// Wraps single op in `tf_device.launch` for explicit device assignment.
void WrapOpInLaunch(OpBuilder* builder, Location loc, Operation* op,
                    llvm::StringRef device) {
  builder->setInsertionPoint(op);
  auto launch = builder->create<tf_device::LaunchOp>(
      loc, builder->getStringAttr(device), op->getResultTypes());
  launch.body().push_back(new Block);
  op->replaceAllUsesWith(launch);

  builder->setInsertionPointToEnd(&launch.GetBody());
  builder->create<tf_device::ReturnOp>(loc, op->getResults());

  // Move op inside cluster.
  op->moveBefore(launch.GetBody().getTerminator());
}

llvm::SmallVector<Operation*, 4> GetResourceOpsUsingCompositeArgsInReplicate(
    tf_device::ReplicateOp replicate) {
  llvm::SmallVector<Operation*, 4> resource_users;
  const auto add_resource_op_to_list = [&resource_users](Operation* op) {
    if (!llvm::isa<TF::AssignVariableOp, TF::ReadVariableOp>(op)) return;

    resource_users.emplace_back(op);
  };

  llvm::SmallVector<Operation*, 4> resource_users_to_visit;
  for (auto composite_arguments : replicate.GetPackedBlockArguments()) {
    for (auto resource_user : composite_arguments.getUsers())
      resource_users_to_visit.emplace_back(resource_user);
  }

  while (!resource_users_to_visit.empty()) {
    llvm::SmallVector<Operation*, 4> new_resource_users;

    for (auto resource_user : resource_users_to_visit) {
      add_resource_op_to_list(resource_user);

      // Account for pass-through identity ops.
      if (auto pass_through_identity =
              llvm::dyn_cast<TF::IdentityOp>(resource_user)) {
        for (auto identity_user : pass_through_identity.output().getUsers()) {
          new_resource_users.emplace_back(identity_user);
        }
      }
    }
    resource_users_to_visit.swap(new_resource_users);
  }

  return resource_users;
}

void ColocateCompositeResourceOpsInReplicate(
    tf_device::ReplicateOp replicate_op, OpBuilder* builder) {
  auto devices = replicate_op.devices();
  if (!devices) return;
  if (!devices.getValue().get(tensorflow::GetDeviceAliasForLogicalCore(0)))
    return;

  const auto composite_resource_users =
      GetResourceOpsUsingCompositeArgsInReplicate(replicate_op);
  for (auto resource_user : composite_resource_users) {
    WrapOpInLaunch(builder, resource_user->getLoc(), resource_user,
                   tensorflow::GetDeviceAliasForLogicalCore(0));
  }
}

void TPUColocateCompositeResourceOps::runOnFunction() {
  // Find all the executes first, since we will mutate the nodes around each
  // execute in the same tf_device.replicate op.
  llvm::SmallVector<tf_device::LaunchOp, 8> execute_launches;
  getFunction().walk([&](tf_device::LaunchOp op) {
    if (op.WrapsSingleOp() &&
        llvm::isa<TF::TPUExecuteOp, TF::TPUExecuteAndUpdateVariablesOp>(
            op.GetBody().front()))
      execute_launches.push_back(op);
  });

  OpBuilder builder(&getContext());
  for (auto execute_launch : execute_launches) {
    auto replicate = execute_launch->getParentOfType<tf_device::ReplicateOp>();
    if (!replicate) continue;

    ColocateCompositeResourceOpsInReplicate(replicate, &builder);
  }
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTPUColocateCompositeResourceOps() {
  return std::make_unique<TPUColocateCompositeResourceOps>();
}

static PassRegistration<TPUColocateCompositeResourceOps> pass(
    "tf-tpu-colocate-composite-resource-ops",
    "Colocate resource with composite device assignment to TPU device.");

}  // namespace TFTPU
}  // namespace mlir
