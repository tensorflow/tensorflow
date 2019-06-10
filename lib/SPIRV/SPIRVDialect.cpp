//===- LLVMDialect.cpp - MLIR SPIR-V dialect ------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/SPIRV/SPIRVDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/SPIRV/SPIRVOps.h"
#include "mlir/SPIRV/SPIRVTypes.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::spirv;

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

SPIRVDialect::SPIRVDialect(MLIRContext *context) : Dialect("spv", context) {
  addTypes<ArrayType, RuntimeArrayType>();

  addOperations<
#define GET_OP_LIST
#include "mlir/SPIRV/SPIRVOps.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mlir/SPIRV/SPIRVStructureOps.cpp.inc"
      >();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

// TODO(b/133530217): The following implements some type parsing logic. It is
// intended to be short-lived and used just before the main parser logic gets
// exposed to dialects. So there is little type checking inside.

static Type parseScalarType(StringRef spec, Builder builder) {
  return llvm::StringSwitch<Type>(spec)
      .Case("f32", builder.getF32Type())
      .Case("i32", builder.getIntegerType(32))
      .Case("f16", builder.getF16Type())
      .Case("i16", builder.getIntegerType(16))
      .Default(Type());
}

// Parses "<number> x" from the beginning of `spec`.
static bool parseNumberX(StringRef &spec, int64_t &number) {
  spec = spec.ltrim();
  if (spec.empty() || !llvm::isDigit(spec.front()))
    return false;

  number = 0;
  do {
    number = number * 10 + spec.front() - '0';
    spec = spec.drop_front();
  } while (!spec.empty() && llvm::isDigit(spec.front()));

  spec = spec.ltrim();
  if (!spec.consume_front("x"))
    return false;

  return true;
}

static Type parseVectorType(StringRef spec, Builder builder) {
  if (!spec.consume_front("vector<") || !spec.consume_back(">"))
    return Type();

  int64_t count = 0;
  if (!parseNumberX(spec, count))
    return Type();

  spec = spec.trim();
  auto scalarType = parseScalarType(spec, builder);
  if (!scalarType)
    return Type();

  return VectorType::get({count}, scalarType);
}

static Type parseArrayType(StringRef spec, Builder builder) {
  if (!spec.consume_front("array<") || !spec.consume_back(">"))
    return Type();

  Type elementType;
  int64_t count = 0;

  spec = spec.trim();
  if (!parseNumberX(spec, count))
    return Type();

  spec = spec.ltrim();
  if (spec.startswith("vector")) {
    elementType = parseVectorType(spec, builder);
  } else {
    elementType = parseScalarType(spec, builder);
  }
  if (!elementType)
    return Type();

  return ArrayType::get(elementType, count);
}

static Type parseRuntimeArrayType(StringRef spec, Builder builder) {
  if (!spec.consume_front("rtarray<") || !spec.consume_back(">"))
    return Type();

  Type elementType;
  spec = spec.trim();
  if (spec.startswith("vector")) {
    elementType = parseVectorType(spec, builder);
  } else {
    elementType = parseScalarType(spec, builder);
  }
  if (!elementType)
    return Type();

  return RuntimeArrayType::get(elementType);
}

Type SPIRVDialect::parseType(StringRef spec, Location loc) const {
  Builder builder(getContext());

  if (auto type = parseArrayType(spec, builder))
    return type;
  if (auto type = parseRuntimeArrayType(spec, builder))
    return type;

  getContext()->emitError(loc, "unknown SPIR-V type: ") << spec;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

static void print(ArrayType type, llvm::raw_ostream &os) {
  os << "array<" << type.getElementCount() << " x " << type.getElementType()
     << ">";
}

static void print(RuntimeArrayType type, llvm::raw_ostream &os) {
  os << "rtarray<" << type.getElementType() << ">";
}

void SPIRVDialect::printType(Type type, llvm::raw_ostream &os) const {
  if (auto t = type.dyn_cast<ArrayType>()) {
    print(t, os);
  } else if (auto t = type.dyn_cast<RuntimeArrayType>()) {
    print(t, os);
  } else {
    llvm_unreachable("unhandled SPIR-V type");
  }
}
