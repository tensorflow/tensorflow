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
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/split_into_island_per_op_pass.h"

namespace mlir {
namespace TFDevice {
namespace {
constexpr char kDeviceAttr[] = "device";

#define GEN_PASS_DEF_LAUNCHTODEVICEATTRIBUTEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_device_passes.h.inc"

struct LaunchToDeviceAttributePass
    : public impl::LaunchToDeviceAttributePassBase<
          LaunchToDeviceAttributePass> {
 public:
  explicit LaunchToDeviceAttributePass(bool legacy_graph_export) {
    legacy_graph_export_ = legacy_graph_export;
  }

  void runOnOperation() override;
};

// Assign all ops in region with specified device from launch.
LogicalResult AssignDevicesInRegion(const Dialect* tf_dialect,
                                    tf_device::LaunchOp launch,
                                    Region& region) {
  auto parallel_group_attr =
      launch->getAttrOfType<StringAttr>(TF::kParallelExecAnnotation);
  auto result = region.walk([&](Operation* op) -> WalkResult {
    if (op->getDialect() != tf_dialect) return WalkResult::advance();

    if (parallel_group_attr) {
      op->setAttr(TF::kParallelExecAnnotation, parallel_group_attr);
    }
    auto device_attr = op->getAttr(kDeviceAttr);
    if (!device_attr) {
      op->setAttr(kDeviceAttr, launch.getDeviceAttr());
      return WalkResult::advance();
    }

    if (auto device_str_attr = mlir::dyn_cast<StringAttr>(device_attr)) {
      if (device_str_attr.getValue().empty()) {
        op->setAttr(kDeviceAttr, launch.getDeviceAttr());
        return WalkResult::advance();
      } else if (device_str_attr.getValue() != launch.getDevice()) {
        return launch.emitOpError()
               << "inner op has conflicting 'device' attribute, "
                  "got '"
               << device_str_attr.getValue() << "' but expected '"
               << launch.getDevice() << "'";
      }
    } else {
      return launch.emitOpError()
             << "inner op has bad 'device' attribute, got " << device_attr;
    }

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}

LogicalResult HoistOpsAndAnnotateWithDevice(const Dialect* tf_dialect,
                                            tf_device::LaunchOp launch) {
  // Forward launch inner op results to launch op results.
  launch.replaceAllUsesWith(launch.GetBody().getTerminator()->getOperands());

  // For all inner ops, assign the launch device as a `device` attribute.
  if (failed(AssignDevicesInRegion(tf_dialect, launch, launch.getBody())))
    return failure();

  // Move all inner ops of the launch to the block containing the launch.
  auto body = launch.GetBody().without_terminator();
  Operation* launch_op = launch.getOperation();
  launch_op->getBlock()->getOperations().splice(
      launch_op->getIterator(), launch.GetBody().getOperations(), body.begin(),
      body.end());

  launch.erase();

  return success();
}

void LaunchToDeviceAttributePass::runOnOperation() {
  const Dialect* tf_dialect = getContext().getLoadedDialect("tf");
  if (!tf_dialect) {
    getOperation().emitError() << "'tf' dialect is not registered";
    return signalPassFailure();
  }

  auto result = getOperation().walk([&tf_dialect](tf_device::LaunchOp launch) {
    if (failed(HoistOpsAndAnnotateWithDevice(tf_dialect, launch)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();

  if (!legacy_graph_export_) {
    // Now, split the island into an island per op since we don't want to
    // violate the invariant imposed by the GraphExport pipeline that every
    // IslandOp perfectly wraps a single op.
    auto control_type =
        mlir::tf_executor::ControlType::get(tf_dialect->getContext());
    getOperation().walk(
        [&control_type](mlir::tf_executor::IslandOp curr_island) {
          mlir::TF::SplitIsland(curr_island, control_type);
        });
  }
}

}  // anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLaunchToDeviceAttributePass(
    bool legacy_graph_export) {
  return std::make_unique<LaunchToDeviceAttributePass>(legacy_graph_export);
}

}  // namespace TFDevice
}  // namespace mlir
