/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.h"

namespace mlrt {
namespace compiler {

namespace {

struct MlrtInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool would_be_cloned,
                       mlir::IRMapping &mapping) const final {
    // All mlrt dialect ops can be inlined.
    return true;
  }
};

}  // namespace

MlrtDialect::MlrtDialect(mlir::MLIRContext *context)
    : mlir::Dialect(/*name=*/"mlrt", context,
                    mlir::TypeID::get<MlrtDialect>()) {
  addTypes<FutureType>();
  addTypes<PromiseType>();
  addTypes<AsyncHandleType>();
  addInterfaces<MlrtInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.cpp.inc"
      >();
}

// Parse a type registered to this dialect.
mlir::Type MlrtDialect::parseType(mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();

  if (keyword == "future") return FutureType::get(getContext());
  if (keyword == "promise") return PromiseType::get(getContext());
  if (keyword == "async_handle") return AsyncHandleType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return mlir::Type();
}

// Print a type registered to this dialect.
void MlrtDialect::printType(mlir::Type type,
                            mlir::DialectAsmPrinter &os) const {
  if (mlir::isa<FutureType>(type)) {
    os << "future";
    return;
  }

  if (mlir::isa<PromiseType>(type)) {
    os << "promise";
    return;
  }

  if (mlir::isa<AsyncHandleType>(type)) {
    os << "async_handle";
    return;
  }

  llvm_unreachable("unexpected mlrt type kind");
}

}  // namespace compiler
}  // namespace mlrt
