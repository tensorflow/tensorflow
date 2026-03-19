/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::BlockArgument;
using mlir::failure;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::OperationPass;
using mlir::OpOperand;
using mlir::StringAttr;
using mlir::success;
using mlir::Value;
using mlir::WalkResult;
using mlir::func::FuncOp;
using mlir::TF::ReadVariableOp;
using mlir::tf_device::ReplicateOp;

#define GEN_PASS_DEF_HOISTBROADCASTREADPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/clustering_passes.h.inc"

constexpr char kFuncDeviceAttr[] = "tf.device";
constexpr char kCpuDeviceType[] = "CPU";

struct HoistBroadcastRead
    : public impl::HoistBroadcastReadPassBase<HoistBroadcastRead> {
  void runOnOperation() override;
};

// Get the ancestor of `descendant` that is a direct child of `ancestor`.
Operation* GetAncestorBelow(Operation* descendant, Operation* ancestor) {
  Operation* parent = descendant->getParentOp();
  if (!parent) return nullptr;
  if (parent == ancestor) return descendant;
  return GetAncestorBelow(parent, ancestor);
}

// `is_cpu_read` is set to `true` iff `read` is on a resource with device type
// CPU.
LogicalResult IsCpuRead(FuncOp func, ReadVariableOp read, bool& is_cpu_read) {
  if (auto arg = mlir::dyn_cast<BlockArgument>(read->getOperand(0))) {
    if (arg.getOwner() != &(func.front())) {
      is_cpu_read = false;
      return success();
    }
    if (auto attr = func.getArgAttrOfType<StringAttr>(arg.getArgNumber(),
                                                      kFuncDeviceAttr)) {
      std::string device = attr.getValue().str();
      tensorflow::DeviceNameUtils::ParsedName parsed_name;
      if (!tensorflow::DeviceNameUtils::ParseFullName(device, &parsed_name)) {
        return read->emitOpError() << "invalid device '" << device << "'";
      }
      is_cpu_read = parsed_name.type == kCpuDeviceType;
      return success();
    }
  }
  is_cpu_read = false;
  return success();
}

// Get the reads to hoist in the `replicate`.
LogicalResult GetReads(FuncOp func, ReplicateOp replicate,
                       llvm::SmallVector<ReadVariableOp, 4>& reads) {
  for (Operation& op : replicate.getBody().front()) {
    if (auto read = llvm::dyn_cast<ReadVariableOp>(&op)) {
      bool is_cpu_read;
      if (failed(IsCpuRead(func, read, is_cpu_read))) return failure();
      if (is_cpu_read) reads.push_back(read);
    }
  }
  return success();
}

// Move reads above the `replicate`. Skip reads that come after a write to the
// same resource.
void MoveReads(ReplicateOp replicate,
               llvm::SmallVector<ReadVariableOp, 4>& reads) {
  for (ReadVariableOp read : reads) {
    Value res = read.getResource();
    Operation* scope = res.getParentBlock()->getParentOp();
    if (!scope->isProperAncestor(replicate)) continue;
    bool has_conflicting_write = false;
    for (OpOperand& use : res.getUses()) {
      Operation* using_op = use.getOwner();
      if (using_op == read) continue;
      if (!replicate->isProperAncestor(using_op)) continue;
      Operation* peer = GetAncestorBelow(using_op, replicate);
      if (read->isBeforeInBlock(peer)) continue;
      if (llvm::isa<ReadVariableOp>(peer)) continue;
      has_conflicting_write = true;
    }
    if (has_conflicting_write) continue;
    read->moveBefore(replicate);
  }
}

// Hoist `ReadVariableOp`s above the `tf_device.replicate`s.
void HoistBroadcastRead::runOnOperation() {
  FuncOp func = getOperation();

  auto result = func.walk([&](ReplicateOp replicate) {
    llvm::SmallVector<ReadVariableOp, 4> reads;
    if (failed(GetReads(func, replicate, reads)))
      return WalkResult::interrupt();
    MoveReads(replicate, reads);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateHoistBroadcastReadPass() {
  return std::make_unique<HoistBroadcastRead>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
