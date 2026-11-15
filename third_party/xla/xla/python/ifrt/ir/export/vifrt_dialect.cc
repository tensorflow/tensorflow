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

#include "xla/python/ifrt/ir/export/vifrt_dialect.h"

#include <cassert>
#include <cstdint>

#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: export
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/export/vifrt_bytecode.h"
#include "xla/python/ifrt/ir/sharding_param.h"

namespace xla {
namespace ifrt {

namespace {

// Verifies if a given type or attribute is from VIFRT dialect.
// Must be defined before importing the generated type interfaces and defs.
template <typename TypeOrAttr>
bool isFromBuiltinOrVifrt(TypeOrAttr t) {
  return t != nullptr && (t.getDialect().getNamespace() ==
                              VifrtDialect::getDialectNamespace() ||
                          t.getDialect().getNamespace() ==
                              mlir::BuiltinDialect::getDialectNamespace());
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
  auto emit_error = [&]() { return parser.emitError(parser.getNameLoc()); };
  xla::ifrt::VifrtFunctionV1Type vifrt_func_type =
      xla::ifrt::VifrtFunctionV1Type::getChecked(
          emit_error, parser.getContext(),
          llvm::ArrayRef<mlir::Type>(input_types),
          llvm::ArrayRef<mlir::Type>(res_types));
  if (vifrt_func_type == nullptr) {
    return mlir::failure();
  }
  func_type = xla::ifrt::VifrtTypeV1Attr::getChecked(
      emit_error, parser.getContext(), mlir::Type(vifrt_func_type));
  if (func_type == nullptr) {
    return mlir::failure();
  }
  return mlir::success();
}

// Attributes
#include "xla/python/ifrt/ir/export/vifrt_attr_interfaces.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/export/vifrt_attrs.cc.inc"
// Types
#include "xla/python/ifrt/ir/export/vifrt_type_interfaces.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/export/vifrt_types.cc.inc"
// Ops
#include "xla/python/ifrt/ir/export/vifrt_op_interfaces.cc.inc"
#define GET_OP_CLASSES
#include "xla/python/ifrt/ir/export/vifrt_ops.cc.inc"

//===----------------------------------------------------------------------===//
// VIFRT Attribute and Type Verifiers
//===----------------------------------------------------------------------===//
namespace xla {
namespace ifrt {
namespace {

// Returns the number of elements in a validated `VifrtIntervalV1Attr`.
int GetIntervalSize(VifrtIntervalV1Attr interval) {
  return 1 + (interval.getEnd() - interval.getStart() - 1) / interval.getStep();
}

}  // namespace

mlir::LogicalResult VifrtDevicesV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int> ids) {
  if (ids.empty()) {
    return emitError() << "Device list cannot be empty";
  }
  llvm::DenseSet<int> device_set;
  for (int id : ids) {
    if (id < 0) {
      return emitError() << "Device list has negative logical id " << id;
    }
    if (auto [unused_it, inserted] = device_set.insert(id); !inserted) {
      return emitError() << "Device list has duplicate logical id " << id;
    }
  }
  return mlir::success();
}

mlir::LogicalResult VifrtShardingParamV2Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ShardingParam sharding) {
  return sharding.verify(emitError);
}

mlir::LogicalResult VifrtIntervalV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int start,
    int end, int step) {
  if (start < 0 || end < 0) {
    return emitError() << "start, end must be zero or positive";
  }
  if (step <= 0) {
    return emitError() << "step must be positive";
  }
  if (start >= end) {
    return emitError() << "interval is empty";
  }
  return mlir::success();
}

mlir::LogicalResult VifrtMappingV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    VifrtIntervalV1Attr from_shards, VifrtIntervalV1Attr to_shards) {
  if (from_shards == nullptr || to_shards == nullptr) {
    return emitError() << "expected non-null interval attributes";
  }
  const int from_size = GetIntervalSize(from_shards);
  const int to_size = GetIntervalSize(to_shards);
  if (from_size != to_size) {
    return emitError() << "from has " << from_size << " and to has " << to_size
                       << ", but they must have the same number of shards.";
  }
  return mlir::success();
}

mlir::LogicalResult VifrtArrayMappingV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int32_t in_array_index, int32_t out_array_index, mlir::ArrayAttr mappings) {
  if (in_array_index < 0 || out_array_index < 0) {
    return emitError()
           << "in_array_index and out_array_index must be non-negative";
  }
  if (mappings == nullptr || mappings.empty() ||
      !llvm::all_of(mappings.getValue(), llvm::IsaPred<VifrtMappingV1Attr>)) {
    return emitError() << "expected non-empty array of VifrtMappingV1Attr";
  }
  return mlir::success();
}

mlir::LogicalResult VifrtTypeV1Attr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type value) {
  if (!isFromBuiltinOrVifrt(value)) {
    return emitError() << "expected VIFRT or builtin type";
  }
  return mlir::success();
}

mlir::LogicalResult VifrtArrayV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, mlir::Attribute sharding_attr,
    VifrtDevicesV1Attr devices_attr, mlir::StringAttr memory_kind_attr,
    mlir::StringAttr layout_attr) {
  if (shape == nullptr || devices_attr == nullptr ||
      memory_kind_attr == nullptr || layout_attr == nullptr) {
    return emitError() << "expected non-null VifrtArrayV1Type parameters";
  }
  if (sharding_attr == nullptr ||
      !llvm::isa<VifrtShardingParamV2Attr, VifrtUnspecifiedShardingV1Attr>(
          sharding_attr)) {
    return emitError() << "expected VIFRT sharding attribute";
  }
  if (shape.getEncoding() != nullptr) {
    return emitError() << "Unexpected tensor encoding on array shape: "
                       << shape;
  }
  if (!shape.hasStaticShape()) {
    return emitError() << "Array shape must be static, got " << shape;
  }
  int64_t num_elements = 1;
  for (int64_t dim : shape.getShape()) {
    if (__builtin_mul_overflow(num_elements, dim, &num_elements)) {
      return emitError()
             << "Array shape element count overflows 64-bit signed integer: "
             << shape;
    }
  }
  if (llvm::isa<VifrtTokenV1Type>(shape.getElementType())) {
    if (!shape.getShape().empty()) {
      return emitError() << "Token type cannot have non-empty shape";
    }
  } else if (!llvm::isa<VifrtStringV1Type>(shape.getElementType()) &&
             shape.getElementType().getDialect().getNamespace() !=
                 mlir::BuiltinDialect::getDialectNamespace()) {
    return emitError() << "Unsupported array element type: "
                       << shape.getElementType();
  }
  if (layout_attr.getValue() != kVifrtDefaultString) {
    absl::StatusOr<xla::LayoutMode> layout_mode =
        xla::LayoutMode::FromString(layout_attr.str());
    if (!layout_mode.ok()) {
      return emitError() << "Invalid layout mode: "
                         << layout_mode.status().message();
    }
  }
  if (auto sharding_param_attr =
          llvm::dyn_cast<VifrtShardingParamV2Attr>(sharding_attr)) {
    return sharding_param_attr.getSharding().CanApplyTo(emitError, shape,
                                                        devices_attr.getIds());
  }
  return mlir::success();
}

mlir::LogicalResult VifrtFunctionV1Type::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> inputs, llvm::ArrayRef<mlir::Type> outputs) {
  if (!allFromBuiltinOrVifrt(inputs) || !allFromBuiltinOrVifrt(outputs)) {
    return emitError() << "expected VIFRT or builtin types";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// VIFRT Dialect
//===----------------------------------------------------------------------===//

VifrtDialect::VifrtDialect(mlir::MLIRContext* context)
    : mlir::Dialect(getDialectNamespace(), context,
                    mlir::TypeID::get<VifrtDialect>()) {
  addBytecodeInterface(this);
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/python/ifrt/ir/export/vifrt_attrs.cc.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/export/vifrt_types.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/export/vifrt_ops.cc.inc"
      >();
}

mlir::Type VifrtDialect::parseType(mlir::DialectAsmParser& parser) const {
  llvm::StringRef data_type;
  mlir::Type type;
  mlir::OptionalParseResult parse_result_opt =
      parseVifrtType(parser, &data_type, type);
  if (parse_result_opt.has_value()) {
    if (mlir::succeeded(*parse_result_opt)) {
      return type;
    }
    return nullptr;
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
  mlir::OptionalParseResult parse_result =
      generatedAttributeParser(parser, &attr_tag, type, attr);
  if (parse_result.has_value()) {
    if (mlir::succeeded(*parse_result)) {
      return attr;
    }
    return mlir::Attribute();
  }
  parser.emitError(parser.getNameLoc())
      << "unknown vifrt attribute: " << attr_tag;
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
    mlir::Location loc = mlir::UnknownLoc::get(type.getContext());
    return VifrtFunctionV1Type::getChecked(
        [&]() { return mlir::emitError(loc); }, type.getContext(),
        llvm::ArrayRef<mlir::Type>(converted_inputs),
        llvm::ArrayRef<mlir::Type>(converted_results));
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
