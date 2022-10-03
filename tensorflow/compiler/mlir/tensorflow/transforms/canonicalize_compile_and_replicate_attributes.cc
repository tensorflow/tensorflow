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

// This transformation pass converts existing compilation and replication
// attributes into unified attributes. For example, A _tpu_replicate=X
// should be replaced with _xla_compile_device_type=TPU and
// _replication_info=X attributes by the conversion. An _XlaMustCompile=true
// should be replaced with _xla_compile_device_type with the value of device
// attribute.

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/core/util/device_name_utils.h"

#define DEBUG_TYPE "tf-canonicalize-compile-and-replicate-attributes"

namespace mlir {
namespace TF {

namespace {

struct CanonicalizeCompileAndReplicateAttributesPass
    : public TF::CanonicalizeCompileAndReplicateAttributesPassBase<
          CanonicalizeCompileAndReplicateAttributesPass> {
  void runOnOperation() override;
};

void CanonicalizeCompileAndReplicateAttributesPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  ModuleOp module_op = func_op->getParentOfType<ModuleOp>();
  mlir::OpBuilder builder(module_op.getContext());

  auto walk_result = func_op->walk([&](mlir::Operation* op) {
    // Convert `_tpu_replicate`.
    if (op->hasAttr(TF::kTpuReplicateAttr)) {
      op->setAttr(tensorflow::kReplicationInfoAttr,
                  op->getAttr(TF::kTpuReplicateAttr));
      op->removeAttr(tensorflow::kTpuReplicateAttr);
      op->setAttr(tensorflow::kCompileDeviceTypeAttr,
                  builder.getStringAttr(tensorflow::kTpuDevice));
    }

    // Convert `_XlaMustCompile`.
    if (op->hasAttr(tensorflow::kMustCompileAttr)) {
      bool must_compile_attr_val =
          op->getAttrOfType<BoolAttr>(tensorflow::kMustCompileAttr).getValue();
      op->removeAttr(tensorflow::kMustCompileAttr);
      if (!must_compile_attr_val) {
        if (op->hasAttr(tensorflow::kCompileDeviceTypeAttr)) {
          op->emitOpError()
              << "has both '" << tensorflow::kMustCompileAttr
              << " = false' and '" << tensorflow::kCompileDeviceTypeAttr
              << "' attribute which contradicts each other";
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      }
      if (op->hasAttr(tensorflow::kCompileDeviceTypeAttr)) {
        return mlir::WalkResult::advance();
      }
      auto device_attr = op->getAttrOfType<StringAttr>(tensorflow::kDeviceAttr);
      if (!device_attr) {
        op->setAttr(tensorflow::kCompileDeviceTypeAttr,
                    builder.getStringAttr(tensorflow::kEmptyDevice));
        return mlir::WalkResult::advance();
      }
      tensorflow::DeviceNameUtils::ParsedName parsed_name;
      tensorflow::DeviceNameUtils::ParseFullOrLocalName(device_attr.getValue(),
                                                        &parsed_name);
      auto device_type = builder.getStringAttr(parsed_name.type);
      if (failed(IsValidDeviceTypeOrEmpty(device_type))) {
        op->emitOpError() << "'" << tensorflow::kDeviceAttr << "'"
                          << " has invalid value";
        return mlir::WalkResult::interrupt();
      }
      op->setAttr(tensorflow::kCompileDeviceTypeAttr, device_type);
    }

    return mlir::WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateCanonicalizeCompileAndReplicateAttributesPass() {
  return std::make_unique<CanonicalizeCompileAndReplicateAttributesPass>();
}

}  // namespace TF
}  // namespace mlir
