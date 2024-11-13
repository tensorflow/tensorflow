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

#include "xla/python/ifrt/ir/vifrt_ops.h"

#include <cassert>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: export
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "xla/python/ifrt/ir/vifrt_types.h"

namespace xla {
namespace ifrt {

namespace {

// Verifies if a given type or attribute is from VIFRT dialect.
template <typename TypeOrAttr>
bool isFromVifrt(TypeOrAttr t) {
  return t.getDialect().getNamespace() == VifrtDialect::getDialectNamespace();
}

template <typename TypeOrAttr>
bool allFromVifrt(llvm::ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromVifrt<TypeOrAttr>);
}

}  // namespace

// Helper functions for VIFRT printers and parsers.
static void printAttributeArray(mlir::AsmPrinter& os,
                                llvm::ArrayRef<mlir::Attribute> arrayAttr) {
  os << '[' << arrayAttr << ']';
}

// Parse attributes in brackets: [#virt.attr, #virt.attr]
mlir::ParseResult parseAttributeArray(
    mlir::AsmParser& parser, llvm::SmallVector<mlir::Attribute>& arrayAttr) {
  mlir::ArrayAttr array;
  if (mlir::failed(parser.parseAttribute(array))) {
    return mlir::failure();
  }
  arrayAttr.append(array.begin(), array.end());
  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla

#include "xla/python/ifrt/ir/vifrt_attr_interfaces.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_attrs.cc.inc"
#include "xla/python/ifrt/ir/vifrt_op_interfaces.cc.inc"
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/vifrt_ops.cc.inc"

namespace xla {
namespace ifrt {

//===----------------------------------------------------------------------===//
// VIFRT Dialect Constructor
//===----------------------------------------------------------------------===//

VifrtDialect::VifrtDialect(mlir::MLIRContext* context)
    : mlir::Dialect(getDialectNamespace(), context,
                    mlir::TypeID::get<VifrtDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/vifrt_ops.cc.inc"
      >();
  addVifrtTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/python/ifrt/ir/vifrt_attrs.cc.inc"
      >();
}

void VifrtDialect::addVifrtTypes() {
  // Following the same solution as in VHLO; Idiomatically, this functionality
  // is expressed as shown below:
  //     addTypes<
  //   #define GET_TYPEDEF_LIST
  //   #include
  //   "third_party/tensorflow/compiler/xla/python/ifrt/ir/vifrt_type_defs.cc.inc"
  //         >();
  //
  // However, Dialect::addTypes doesn't work for our situation where we want to
  // decouple the vifrt_ops and vifrt_types targets because
  // vifrt_type_defs.h.inc only includes forward declarations of `TypeStorage`
  // structs, and that's not sufficient for Dialect::addTypes to compile.
  // Therefore, we introduce this function and then reimplementing
  // Dialect::addTypes as shown below.
  addTypesWithoutRegistering<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/vifrt_type_defs.cc.inc"
      >();
  registerVifrtTypes(getContext());
}

mlir::Type VifrtDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef data_type;
  mlir::Type type;
  auto parse_result_opt = parseVifrtType(parser, &data_type, type);
  if (parse_result_opt.has_value() && mlir::succeeded(*parse_result_opt)) {
    return type;
  }
  parser.emitError(parser.getNameLoc()) << "unknown vifrt type: " << data_type;
  return nullptr;
}

void VifrtDialect::printType(mlir::Type type,
                             mlir::DialectAsmPrinter& os) const {
  if (mlir::succeeded(printVifrtType(type, os))) {
    return;
  }
  os << "<unknown vifrt type>";
}

mlir::Attribute VifrtDialect::parseAttribute(mlir::DialectAsmParser& parser,
                                             mlir::Type type) const {
  llvm::StringRef attr_tag;
  mlir::Attribute attr;
  auto parse_result = generatedAttributeParser(parser, &attr_tag, type, attr);
  if (parse_result.has_value()) {
    return attr;
  }
  parser.emitError(parser.getNameLoc(), "unknown vifrt attribute");
  return mlir::Attribute();
}

void VifrtDialect::printAttribute(mlir::Attribute attr,
                                  mlir::DialectAsmPrinter& os) const {
  mlir::LogicalResult result = generatedAttributePrinter(attr, os);
  // Avoid clang unused variable error.
  (void)result;
  assert(mlir::succeeded(result));
}

}  // namespace ifrt
}  // namespace xla
