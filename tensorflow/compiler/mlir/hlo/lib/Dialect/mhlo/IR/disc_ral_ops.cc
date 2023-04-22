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

// This file defines the operations used in the DISC RAL dialect.

#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"

namespace mlir {
namespace disc_ral {

template <typename T>
static LogicalResult Verify(T op) {
  return success();
}

//===----------------------------------------------------------------------===//
// ral Dialect Constructor
//===----------------------------------------------------------------------===//

RalDialect::RalDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context, TypeID::get<RalDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.cc.inc"
      >();
  addTypes<RalExecutionContextType>();
  context->loadDialect<memref::MemRefDialect>();
}

Type RalDialect::parseType(DialectAsmParser& parser) const {
  StringRef data_type;
  if (parser.parseKeyword(&data_type)) return Type();

  if (data_type == "context") return RalExecutionContextType::get(getContext());
  parser.emitError(parser.getNameLoc())
      << "unknown disc_ral type: " << data_type;
  return nullptr;
}

void RalDialect::printType(Type type, DialectAsmPrinter& os) const {
  if (type.isa<RalExecutionContextType>()) {
    os << "context";
    return;
  }
  os << "<unknown disc_ral type>";
}

}  // namespace disc_ral
}  // namespace mlir

#define GET_OP_CLASSES
#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.cc.inc"
