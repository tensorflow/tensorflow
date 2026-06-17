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

#include "xla/python/ifrt/ir/vifrt_dialect.h"

#include <cassert>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: export
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "xla/python/ifrt/ir/vifrt_bytecode.h"

namespace xla {
namespace ifrt {

namespace {

// Verifies if a given type or attribute is from VIFRT dialect.
// Must be defined before importing the generated type interfaces and defs.
template <typename TypeOrAttr>
bool isFromVifrt(TypeOrAttr t) {
  return t.getDialect().getNamespace() == VifrtDialect::getDialectNamespace();
}

template <typename TypeOrAttr>
bool isFromBuiltinOrVifrt(TypeOrAttr t) {
  return t.getDialect().getNamespace() == VifrtDialect::getDialectNamespace() ||
         t.getDialect().getNamespace() ==
             mlir::BuiltinDialect::getDialectNamespace();
}

template <typename TypeOrAttr>
bool allFromVifrt(llvm::ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromVifrt<TypeOrAttr>);
}

template <typename TypeOrAttr>
bool allFromBuiltinOrVifrt(llvm::ArrayRef<TypeOrAttr> range) {
  return llvm::all_of(range, isFromBuiltinOrVifrt<TypeOrAttr>);
}

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

}  // namespace

}  // namespace ifrt
}  // namespace xla

// Print types in parentheses: (!vifrt.type, !vifrt.type)
static void printTypeArray(mlir::AsmPrinter& os,
                           mlir::ArrayRef<mlir::Type> type_array) {
  if (type_array.empty()) os << "()";
  os << type_array;
}

// Parse types in parentheses: (!vifrt.type, !vifrt.type)
mlir::ParseResult parseTypeArray(mlir::AsmParser& parser,
                                 mlir::SmallVector<mlir::Type>& type_array) {
  if (mlir::succeeded(parser.parseOptionalLParen()) &&
      mlir::succeeded(parser.parseOptionalRParen())) {
    return mlir::success();
  }
  auto parse_element = [&]() {
    return parser.parseType(type_array.emplace_back());
  };
  if (mlir::failed(parser.parseCommaSeparatedList(parse_element))) {
    return mlir::failure();
  }
  return mlir::success();
}

// Print function using: @name(arg : type, ...) -> (res_type...) { body_ops }
void printFunctionBody(mlir::OpAsmPrinter& p, mlir::Operation*,
                       mlir::Attribute name, mlir::Region& region,
                       mlir::Attribute func_type) {
  p.printSymbolName(llvm::cast<mlir::StringAttr>(name).getValue());
  p << '(';
  llvm::interleaveComma(region.getArguments(), p,
                        [&](auto arg) { p.printRegionArgument(arg); });
  p << ") -> (";
  auto fn_type = llvm::cast<xla::ifrt::VifrtFunctionV1Type>(
      llvm::cast<xla::ifrt::VifrtTypeV1Attr>(func_type).getValue());
  llvm::interleaveComma(fn_type.getOutputs(), p,
                        [&](auto res) { p.printType(res); });
  p << ") ";
  p.printRegion(region, false, true, true);
}

mlir::ParseResult parseFunctionBody(mlir::OpAsmParser& parser,
                                    mlir::Attribute& name, mlir::Region& region,
                                    mlir::Attribute& func_type) {
  mlir::StringAttr strName;
  llvm::SmallVector<mlir::OpAsmParser::Argument> args;
  llvm::SmallVector<mlir::Type> input_types;
  llvm::SmallVector<mlir::Type> res_types;
  if (mlir::failed(parser.parseSymbolName(strName)) ||
      mlir::failed(parser.parseArgumentList(
          args, mlir::AsmParser::Delimiter::Paren, true)) ||
      mlir::failed(parser.parseArrowTypeList(res_types)) ||
      mlir::failed(parser.parseRegion(region, args))) {
    return mlir::failure();
  }
  name = mlir::StringAttr::get(parser.getContext(), strName.getValue());
  for (mlir::OpAsmParser::Argument arg : args) {
    input_types.push_back(arg.type);
  }
  func_type = xla::ifrt::VifrtTypeV1Attr::get(
      parser.getContext(), xla::ifrt::VifrtFunctionV1Type::get(
                               parser.getContext(), input_types, res_types));
  return mlir::success();
}

// Attributes
#include "xla/python/ifrt/ir/vifrt_attr_interfaces.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_attrs.cc.inc"
// Types
#include "xla/python/ifrt/ir/vifrt_type_interfaces.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/vifrt_types.cc.inc"
// Ops
#include "xla/python/ifrt/ir/vifrt_op_interfaces.cc.inc"
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/vifrt_ops.cc.inc"

//===----------------------------------------------------------------------===//
// VIFRT Dialect
//===----------------------------------------------------------------------===//
namespace xla {
namespace ifrt {

VifrtDialect::VifrtDialect(mlir::MLIRContext* context)
    : mlir::Dialect(getDialectNamespace(), context,
                    mlir::TypeID::get<VifrtDialect>()) {
  addBytecodeInterface(this);
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/python/ifrt/ir/vifrt_attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/vifrt_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/vifrt_ops.cc.inc"
      >();
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

//===----------------------------------------------------------------------===//
// VIFRT Type Converter to/from Builtin
//===----------------------------------------------------------------------===//

void VifrtTypeConverterBuiltin::addBuiltinToVifrtConversions() {
  // We currently rely on the builtin types being stable, and thus we only
  // convert a subset of builtin types to VIFRT types.
  addConversion([&](mlir::FunctionType type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> converted_inputs;
    llvm::SmallVector<mlir::Type> converted_results;
    if (mlir::failed(convertTypes(type.getInputs(), converted_inputs))) {
      return {};
    }
    if (mlir::failed(convertTypes(type.getResults(), converted_results))) {
      return {};
    }
    return VifrtFunctionV1Type::get(type.getContext(), converted_inputs,
                                    converted_results);
  });
}

void VifrtTypeConverterBuiltin::addVifrtToBuiltinConversions() {
  addConversion([&](VifrtFunctionV1Type type) -> mlir::Type {
    llvm::SmallVector<mlir::Type> converted_inputs;
    llvm::SmallVector<mlir::Type> converted_results;
    if (mlir::failed(convertTypes(type.getInputs(), converted_inputs))) {
      return {};
    }
    if (mlir::failed(convertTypes(type.getOutputs(), converted_results))) {
      return {};
    }
    return mlir::FunctionType::get(type.getContext(), converted_inputs,
                                   converted_results);
  });
}

mlir::LogicalResult printVifrtType(mlir::Type type, mlir::AsmPrinter& printer) {
  return generatedTypePrinter(type, printer);
}

mlir::OptionalParseResult parseVifrtType(mlir::AsmParser& parser,
                                         llvm::StringRef* mnemonic,
                                         mlir::Type& type) {
  return generatedTypeParser(parser, mnemonic, type);
}

}  // namespace ifrt
}  // namespace xla
