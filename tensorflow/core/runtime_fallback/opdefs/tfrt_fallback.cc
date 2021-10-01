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
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project

namespace tfrt {
namespace fallback {

FallbackDialect::FallbackDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfrt_fallback", context,
              TypeID::get<FallbackDialect>()) {
  addTypes<TFTensorType, TFAllocatorType>();

  addOperations<
#define GET_OP_LIST
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.cpp.inc"
      >();
}

/// Parse a type registered to this dialect.
Type FallbackDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword)) return Type();

  if (keyword == "tf_tensor") return TFTensorType::get(getContext());
  if (keyword == "tf_allocator") return TFAllocatorType::get(getContext());

  parser.emitError(parser.getNameLoc(), "unknown type: ") << keyword;
  return Type();
}

/// Print a type registered to this dialect.
void FallbackDialect::printType(Type type, DialectAsmPrinter &os) const {
  if (type.isa<TFTensorType>()) {
    os << "tf_tensor";
    return;
  }

  if (type.isa<TFAllocatorType>()) {
    os << "tf_allocator";
    return;
  }

  llvm_unreachable("unexpected fallback type kind");
}

}  // namespace fallback
}  // namespace tfrt

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/core/runtime_fallback/opdefs/tfrt_fallback.cpp.inc"
