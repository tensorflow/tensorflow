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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/rewrite_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {
namespace {
#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_canonicalize.inc"
}  // namespace

//===----------------------------------------------------------------------===//
// _XlaHostComputeOp
//===----------------------------------------------------------------------===//

// This verifies that `_XlaHostComputeMlirOp` has a well-formed
// `host_mlir_module` attribute.
// For other attributes, there is no additional verification beyond the default.
static LogicalResult Verify(_XlaHostComputeMlirOp op) {
  // Extract the module and function.
  StringRef host_module = op.host_mlir_module();

  if (host_module.empty()) return success();

  mlir::OwningModuleRef module_for_func;
  tensorflow::Status status = tensorflow::DeserializeMlirModule(
      host_module.str(), op->getContext(), &module_for_func);
  if (!status.ok()) {
    return op.emitError()
           << "attribute 'host_mlir_module' can not be deserialized. "
           << status.error_message();
  }

  FuncOp func = module_for_func->lookupSymbol<FuncOp>("host_func");
  if (!func)
    return op.emitError()
           << "serialized module in attribute 'host_mlir_module' does not "
              "contain 'host_func' function.";

  if (op->getNumOperands() != func.getType().getNumInputs())
    return op.emitError()
           << "'host_func' has " << func.getType().getNumInputs()
           << " inputs and '_XlaHostComputeMlir' has " << op->getNumOperands()
           << " operands.  Number of operands/inputs should be the same.";

  if (op->getNumResults() != func.getType().getNumResults())
    return op.emitError() << "'host_func' has "
                          << func.getType().getNumResults()
                          << " results and '_XlaHostComputeMlir' has "
                          << op->getNumResults()
                          << " results.  Number of results should be the same.";

  return success();
}

FuncOp _XlaHostComputeMlirOp::GetHostFunc(mlir::OwningModuleRef* mlir_module) {
  if (!tensorflow::DeserializeMlirModule(host_mlir_module().str(),
                                         this->getContext(), mlir_module)
           .ok())
    return nullptr;
  return (*mlir_module)->lookupSymbol<FuncOp>("host_func");
}

}  // namespace TF
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.cc.inc"
