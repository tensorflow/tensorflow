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

#include <string>

#include "absl/log/check.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LLVM.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

unsigned IfrtCallOpInfo::getHashValue(CallOp call_op) {
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

bool IfrtCallOpInfo::isEqual(CallOp lhs, CallOp rhs) {
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

bool IsReshard(IfrtArrayType from, IfrtArrayType to) {
  if (from.getShape() == to.getShape() &&
      from.getShardingAttr() == to.getShardingAttr() &&
      from.getDevices().size() == to.getDevices().size()) {
    return false;
  }
  return true;
}

void UpdateFunctionType(mlir::func::FuncOp func_op) {
  func_op.setType(mlir::FunctionType::get(
      func_op.getContext(), func_op.getBody().getArgumentTypes(),
      func_op.getBody().front().getTerminator()->getOperandTypes()));
}

absl::StatusOr<DType> ToIfrtDType(mlir::Type type) {
  xla::PrimitiveType primitive_type = xla::ConvertMlirTypeToPrimitiveType(type);
  return ToDType(primitive_type);
}

std::string OperationToString(mlir::Operation* op,
                              const mlir::OpPrintingFlags& flags) {
  std::string out;
  {
    llvm::raw_string_ostream os(out);
    op->print(os, flags);
  }
  return out;
}

mlir::ModuleOp CloneModuleUsingBuilder(mlir::ModuleOp module,
                                       mlir::OpBuilder& builder) {
  // Create a stub for the new module.
  mlir::ModuleOp cloned_module =
      builder.create<mlir::ModuleOp>(module.getLoc(), module.getName());
  cloned_module->setAttrs(module->getAttrs());
  mlir::IRMapping mapper;
  // Clone each operation in the body of the module into the new module.
  for (mlir::Operation& op : module.getBody()->getOperations()) {
    cloned_module.getBody()->push_back(op.clone(mapper));
  }
  return cloned_module;
}

}  // namespace ifrt
}  // namespace xla
