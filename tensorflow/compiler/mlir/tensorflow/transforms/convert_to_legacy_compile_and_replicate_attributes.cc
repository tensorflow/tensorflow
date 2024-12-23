/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass converts unified compilation and replication
// attributes into legacy attributes. For example,  _replication_info=X
// and _xla_compile_device_type=TPU should be replaced with _tpu_replicate=X.
// This ensures the unified attributes not get exposed outside of the MLIR
// bridge with V1 pipeline in some cases.

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

#define GEN_PASS_DEF_CONVERTTOLEGACYCOMPILEANDREPLICATEATTRIBUTESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ConvertToLegacyCompileAndReplicateAttributesPass
    : public impl::ConvertToLegacyCompileAndReplicateAttributesPassBase<
          ConvertToLegacyCompileAndReplicateAttributesPass> {
  void runOnOperation() override;
};

LogicalResult ConvertToLegacyAttributes(func::FuncOp func_op) {
  auto result = func_op->walk([&](mlir::Operation* op) {
    if (failed(TF::HasValidCompilationAndReplicationAttributes(*op)))
      return WalkResult::interrupt();
    if (op->hasAttr(TF::kReplicationInfoAttr)) {
      op->setAttr(TF::kTpuReplicateAttr, op->getAttr(TF::kReplicationInfoAttr));
      op->removeAttr(TF::kReplicationInfoAttr);
      op->removeAttr(TF::kCompileDeviceTypeAttr);
    }
    return mlir::WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

void ConvertToLegacyCompileAndReplicateAttributesPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  if (failed(ConvertToLegacyAttributes(func_op))) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateConvertToLegacyCompileAndReplicateAttributesPass() {
  return std::make_unique<ConvertToLegacyCompileAndReplicateAttributesPass>();
}

}  // namespace TFTPU
}  // namespace mlir
