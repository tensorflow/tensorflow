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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_OPS_H_
#define XLA_PYTHON_IFRT_IR_VIFRT_OPS_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Support/LLVM.h"
#include "xla/python/ifrt/ir/version.h"  // IWYU pragma: export
#include "xla/python/ifrt/ir/vifrt_types.h"  // IWYU pragma: export

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

 private:
  // Adds VIFRT types to this dialect.
  // See implementation comment for additional details.
  void addVifrtTypes();

  // Does the same this as Dialect::addTypes but without calling `registerType`.
  // See comments for `addVifrtTypes` for additional details.
  template <typename... Types>
  void addTypesWithoutRegistering() {
    (addType(Types::getTypeID(), mlir::AbstractType::get<Types>(*this)), ...);
  }
};

}  // namespace ifrt
}  // namespace xla

// Attrs
#include "xla/python/ifrt/ir/vifrt_attr_interfaces.h.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_attrs.h.inc"

// Ops
#include "xla/python/ifrt/ir/vifrt_op_interfaces.h.inc"
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/vifrt_ops.h.inc"

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_OPS_H_
