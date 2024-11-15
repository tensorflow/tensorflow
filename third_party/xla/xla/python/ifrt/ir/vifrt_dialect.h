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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_H_
#define XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/python/ifrt/ir/sharding_param.h"  // IWYU pragma: export
#include "xla/python/ifrt/ir/version.h"  // IWYU pragma: export

namespace xla {
namespace ifrt {

class VifrtDialect : public mlir::Dialect {
 public:
  explicit VifrtDialect(mlir::MLIRContext *context);

  static mlir::StringRef getDialectNamespace() { return "vifrt"; }

  // Parses a type registered in the VIFRT dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  // Prints a type registered in the VIFRT dialect.
  void printType(mlir::Type type, mlir::DialectAsmPrinter &os) const override;

  // Parses an attribute registered in the VIFRT dialect.
  mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
                                 mlir::Type type) const override;

  // Prints an attribute registered in the VIFRT dialect.
  void printAttribute(mlir::Attribute attr,
                      mlir::DialectAsmPrinter &os) const override;
};

class VifrtTypeConverterBase : public mlir::TypeConverter {
 public:
  VifrtTypeConverterBase() : mlir::TypeConverter() {};

  ~VifrtTypeConverterBase() override = default;
};

// Class used to manage conversions between VIFRT and Builtin types.
class VifrtTypeConverterBuiltin : public VifrtTypeConverterBase {
 public:
  // A subclass can call this method to add conversions from VIFRT to Builtin
  // types. Conversions are applied in reverse order, with the most recently
  // added conversion attempted to be applied first.
  void addVifrtToBuiltinConversions();

  // A subclass can call this method to add conversions from Builtin to VIFRT
  // types. Conversions are applied in reverse order, with the most recently
  // added conversion attempted to be applied first.
  void addBuiltinToVifrtConversions();
};

// Auto-generated VIFRT type printers and parsers.
mlir::LogicalResult printVifrtType(mlir::Type type, mlir::AsmPrinter &printer);
mlir::OptionalParseResult parseVifrtType(mlir::AsmParser &parser,
                                         llvm::StringRef *mnemonic,
                                         mlir::Type &type);

}  // namespace ifrt
}  // namespace xla

// Generated definitions.
// Attributes
#include "xla/python/ifrt/ir/vifrt_attr_interfaces.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_attrs.h.inc"
// Types
#include "xla/python/ifrt/ir/vifrt_type_interfaces.h.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_types.h.inc"
// Ops
#include "xla/python/ifrt/ir/vifrt_op_interfaces.h.inc"
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/vifrt_ops.h.inc"

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_DIALECT_H_
