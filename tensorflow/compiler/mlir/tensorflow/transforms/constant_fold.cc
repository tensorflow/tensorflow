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

#include "tensorflow/compiler/mlir/tensorflow/transforms/constant_fold.h"

#include <algorithm>

#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/eval_util.h"
#include "tensorflow/core/platform/mutex.h"

namespace mlir {
namespace TF {

LogicalResult ConstantFoldFallbackHook(
    Operation* inst, ArrayRef<Attribute> operands,
    SmallVectorImpl<Attribute>& results) {  // NOLINT
  // Instructions with side effects should not be constant folded to preserve
  // the original semantics.
  if (inst->getNumRegions() != 0 || !MemoryEffectOpInterface::hasNoEffect(inst))
    return failure();

  // If any of the result types are variants, don't try to constant fold them.
  // This creates opaque variant constants which lose information and would
  // require "raising" later.
  for (auto& type : inst->getResultTypes()) {
    if (auto tensor_type = type.dyn_cast<TensorType>()) {
      if (tensor_type.getElementType().isa<VariantType>()) {
        return failure();
      }
    }
  }

  // Do not execute function calls.
  if (llvm::isa<TF::WhileOp, TF::IfOp, CallOpInterface>(inst)) {
    return failure();
  }

  // TODO(jpienaar): Currently this persists the entire program execution. This
  // should instead be per module/set from the Graph being executed in TF (if
  // any) so that the value of variables in the context could be read.
  // Note: Sharing the context is fine as ops are side-effect free.
  auto initialize = []() {
    TF_Status* status = TF_NewStatus();
    // The TFE_Context is created without an accompanying delete due to current
    // lifetime. This does not result in memory leaks reported (see totw/110).
    TFE_ContextOptions* opts = TFE_NewContextOptions();
    auto ctx = TFE_NewContext(opts, status);
    TFE_DeleteContextOptions(opts);
    TF_DeleteStatus(status);
    return ctx;
  };
  static TFE_Context* ctx = initialize();

  // Returns directly if any of the operands is not an elements attributes.
  if (std::any_of(operands.begin(), operands.end(), [](Attribute attr) {
        return !attr || !attr.isa<ElementsAttr>();
      }))
    return failure();

  SmallVector<ElementsAttr, 4> inputs;
  inputs.reserve(operands.size());
  for (auto input : operands) {
    inputs.push_back(input.cast<ElementsAttr>());
  }

  // Avoid overlapping folds with the same context.
  // TODO(jpienaar): Avoid using global context & mutex here.
  static auto* mu = new tensorflow::mutex();
  tensorflow::mutex_lock l(*mu);
  return tensorflow::EvaluateOperation(inst, inputs, ctx, &results);
}

}  // namespace TF
}  // namespace mlir
