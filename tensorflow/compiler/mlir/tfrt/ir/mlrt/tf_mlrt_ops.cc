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
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_dialect.h"

namespace tensorflow {
namespace tf_mlrt {

namespace {

struct TensorflowMlrtInlinerInterface : public mlir::DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;
  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool would_be_cloned,
                       mlir::IRMapping &mapping) const final {
    // All tf_mlrt dialect ops can be inlined.
    return true;
  }
  // Note that CallOp and ReturnOp are handled by func; so need to implement
  // handleTerminator.
};

}  // namespace

TensorflowMlrtDialect::TensorflowMlrtDialect(mlir::MLIRContext *context)
    : mlir::Dialect(/*name=*/"tf_mlrt", context,
                    mlir::TypeID::get<TensorflowMlrtDialect>()) {
  addTypes<TFTensorType, TFDeviceType>();
  addInterfaces<TensorflowMlrtInlinerInterface>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_ops.cpp.inc"
      >();
}

// Parse a type registered to this dialect.
mlir::Type TensorflowMlrtDialect::parseType(
    mlir::DialectAsmParser &parser) const {
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword)) return mlir::Type();

  if (keyword == "tensor") return TFTensorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return mlir::Type();
}

// Print a type registered to this dialect.
void TensorflowMlrtDialect::printType(mlir::Type type,
                                      mlir::DialectAsmPrinter &os) const {
  if (type.isa<TFTensorType>()) {
    os << "tensor";
    return;
  }

  llvm_unreachable("unexpected tf_mlrt type kind");
}

}  // namespace tf_mlrt
}  // namespace tensorflow

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.cpp.inc"
#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_ops.cpp.inc"
