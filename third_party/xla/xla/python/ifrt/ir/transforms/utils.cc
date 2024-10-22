/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/transforms/utils.h"

#include "absl/log/check.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"

namespace xla {
namespace ifrt {

unsigned IfrtCallOpInfo::getHashValue(xla::ifrt::CallOp call_op) {
  llvm::hash_code hash = {};
  // Use `getInputs()/getOutputs()` instead of `getOperands()/getResults()` to
  // ensure that the control dependencies are not included in the hash.
  for (auto input_type : call_op.getInputs().getTypes()) {
    hash = llvm::hash_combine(hash, input_type);
  }
  for (auto output_type : call_op.getOutputs().getTypes()) {
    hash = llvm::hash_combine(hash, output_type);
  }
  for (mlir::NamedAttribute attr : call_op->getAttrs()) {
    // Exclude `operandSegmentSizes` because its value changes depending on
    // how many control dependencies a CallOp has.
    if (attr.getName() == "operandSegmentSizes") {
      continue;
    }
    hash = llvm::hash_combine(hash, attr);
  }
  return hash;
}

bool IfrtCallOpInfo::isEqual(xla::ifrt::CallOp lhs, xla::ifrt::CallOp rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (lhs == getEmptyKey() || lhs == getTombstoneKey() ||
      rhs == getEmptyKey() || rhs == getTombstoneKey()) {
    return false;
  }
  // Verify that the input and output types are the same.
  if (lhs.getInputs().getTypes() != rhs.getInputs().getTypes()) {
    return false;
  }
  if (lhs.getOutputs().getTypes() != rhs.getOutputs().getTypes()) {
    return false;
  }
  mlir::NamedAttrList lattrs = lhs->getAttrDictionary();
  mlir::NamedAttrList rattrs = rhs->getAttrDictionary();
  lattrs.erase("operandSegmentSizes");
  rattrs.erase("operandSegmentSizes");
  // Verify that the attributes are the same.
  return lattrs == rattrs;
}

mlir::func::FuncOp GetMainFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp func =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(module.lookupSymbol("main"));
  CHECK(func);
  return func;
}

bool IsReshard(xla::ifrt::IfrtArrayType from, xla::ifrt::IfrtArrayType to) {
  if (from.getShape() == to.getShape() &&
      from.getShardingAttr() == to.getShardingAttr() &&
      from.getDevices().size() == to.getDevices().size()) {
    return false;
  }
  return true;
}

}  // namespace ifrt
}  // namespace xla
