//===- SPIRVDialect.h - MLIR SPIR-V dialect ---------------------*- C++ -*-===//
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
// This file declares the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
#define MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_

#include "mlir/IR/Dialect.h"

namespace mlir {
namespace spirv {

enum class Decoration : uint32_t;

class SPIRVDialect : public Dialect {
public:
  explicit SPIRVDialect(MLIRContext *context);

  static StringRef getDialectNamespace() { return "spv"; }

  /// Checks if the given `type` is valid in SPIR-V dialect.
  static bool isValidType(Type type);

  /// Checks if the given `scalar type` is valid in SPIR-V dialect.
  static bool isValidScalarType(Type type);

  /// Returns the attribute name to use when specifying decorations on results
  /// of operations.
  static std::string getAttributeName(Decoration decoration);

  /// Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  /// Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  /// Provides a hook for materializing a constant to this dialect.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;
};

} // end namespace spirv
} // end namespace mlir

#endif // MLIR_DIALECT_SPIRV_SPIRVDIALECT_H_
