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

#include "llvm/Support/Casting.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDevice[] = "device";

#define GEN_PASS_DEF_COLOCATETPUCOPYWITHDYNAMICSHAPEPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ColocateTPUCopyWithDynamicShapePass
    : public impl::ColocateTPUCopyWithDynamicShapePassBase<
          ColocateTPUCopyWithDynamicShapePass> {
  void runOnOperation() override;
};

class Device : public dataflow::AbstractSparseLattice {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Device)
  using AbstractSparseLattice::AbstractSparseLattice;

  ChangeResult meet(const dataflow::AbstractSparseLattice &other) override {
    const auto *otherDevice = reinterpret_cast<const Device *>(&other);
    if (otherDevice->device_) {
      return SetDevice(otherDevice->device_);
    }
    return ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    if (device_) {
      os << "<" << device_ << ">";
    } else {
      os << "<no device>";
    }
  }

  ChangeResult SetDevice(mlir::StringAttr device) {
    bool changed = (device != device_);
    device_ = device;
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  bool hasDevice() const { return !!device_; }

  mlir::StringAttr getDevice() const { return device_; }

 private:
  mutable mlir::StringAttr device_;
};

class DeviceDataflowAnalysis
    : public ::mlir::dataflow::SparseBackwardDataFlowAnalysis<Device> {
 public:
  using ::mlir::dataflow::SparseBackwardDataFlowAnalysis<
      Device>::SparseBackwardDataFlowAnalysis;
  ~DeviceDataflowAnalysis() override = default;

  LogicalResult visitOperation(Operation *op, ArrayRef<Device *> operands,
                               ArrayRef<const Device *> results) override {
    if (llvm::isa<TF::TPUExecuteOp>(op) ||
        llvm::isa<TF::TPUExecuteAndUpdateVariablesOp>(op)) {
      auto device = op->getAttrOfType<StringAttr>(kDevice);
      for (auto *operand : operands)
        propagateIfChanged(operand, operand->SetDevice(device));
    } else {
      // Propagate device through other ops. These ops might have their
      // own device annotation, but that's fine. We only care about
      // where the TPUExecute ops live.
      StringAttr device;
      for (const Device *d : results) {
        if (d->hasDevice()) {
          device = d->getDevice();
          break;
        }
      }
      for (auto *operand : operands)
        propagateIfChanged(operand, operand->SetDevice(device));
    }
    return mlir::success();
  }
  void visitBranchOperand(OpOperand &operand) override {}
  void visitCallOperand(OpOperand &operand) override {}
  void setToExitState(Device *lattice) override {}
};

void ColocateTPUCopyWithDynamicShapePass::runOnOperation() {
  Operation *module = getOperation();

  SymbolTableCollection symbolTables;

  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<dataflow::SparseConstantPropagation>();
  solver.load<DeviceDataflowAnalysis>(symbolTables);
  if (failed(solver.initializeAndRun(module))) return signalPassFailure();

  module->walk([&](TF::TPUCopyWithDynamicShapeOp op) {
    const Device *state;
    for (auto result : op->getResults()) {
      state = solver.lookupState<Device>(result);
      if (state) break;
    }
    if (!state || !state->hasDevice()) {
      return WalkResult::advance();
    }
    op->setAttr(kDevice, state->getDevice());
    return WalkResult::advance();
  });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateColocateTPUCopyWithDynamicShapePass() {
  return std::make_unique<ColocateTPUCopyWithDynamicShapePass>();
}

}  // namespace TFTPU
}  // namespace mlir
