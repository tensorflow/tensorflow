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

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
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

SPIRVDialect::SPIRVDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<ArrayType, RuntimeArrayType>();

  addOperations<
#define GET_OP_LIST
#include "mlir/SPIRV/SPIRVOps.cpp.inc"
      >();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

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

Type SPIRVDialect::parseAndVerifyType(StringRef spec, Location loc) const {
  auto *context = getContext();
  auto type = mlir::parseType(spec, context);
  if (!type) {
    context->emitError(loc, "cannot parse type: ") << spec;
    return Type();
  }

  // Allow SPIR-V dialect types
  if (&type.getDialect() == this)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      context->emitError(loc, "cannot use 'bf16' to compose SPIR-V types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!llvm::is_contained(llvm::ArrayRef<unsigned>({8, 16, 32, 64}),
                            t.getWidth())) {
      context->emitError(loc,
                         "only 8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      context->emitError(loc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
  } else {
    context->emitError(loc, "cannot use ")
        << type << " to compose SPIR-V types";
    return Type();
  }

  return type;
}

Type SPIRVDialect::parseArrayType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("array<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.array delimiter <...> mismatch");
    return Type();
  }

  int64_t count = 0;
  spec = spec.trim();
  if (!parseNumberX(spec, count)) {
    context->emitError(
        loc, "expected array element count followed by 'x' but found '")
        << spec << "'";
    return Type();
  }

  Type elementType = parseAndVerifyType(spec, loc);
  if (!elementType)
    return Type();

  return ArrayType::get(elementType, count);
}

Type SPIRVDialect::parseRuntimeArrayType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("rtarray<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.rtarray delimiter <...> mismatch");
    return Type();
  }

  Type elementType = parseAndVerifyType(spec, loc);
  if (!elementType)
    return Type();

  return RuntimeArrayType::get(elementType);
}

Type SPIRVDialect::parseType(StringRef spec, Location loc) const {

  if (spec.startswith("array")) {
    return parseArrayType(spec, loc);
  }
  if (spec.startswith("rtarray")) {
    return parseRuntimeArrayType(spec, loc);
  }

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
