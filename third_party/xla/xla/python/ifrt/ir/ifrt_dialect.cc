/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_dialect.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"  // IWYU pragma: keep
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/python/ifrt/ir/constants.h"
#include "xla/python/ifrt/ir/ifrt_interfaces.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/sharding_param.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"

// Generated definitions.
#include "xla/python/ifrt/ir/ifrt_dialect.cc.inc"
#define GET_TYPEDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_types.cc.inc"
#define GET_ATTRDEF_CLASSES
#include "xla/python/ifrt/ir/ifrt_attrs.cc.inc"

namespace xla {
namespace ifrt {

class IfrtAsmDialectInterface : public mlir::OpAsmDialectInterface {
 public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(mlir::Attribute attr,
                       llvm::raw_ostream& os) const override;
};

void IfrtDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "xla/python/ifrt/ir/ifrt_types.cc.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "xla/python/ifrt/ir/ifrt_attrs.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "xla/python/ifrt/ir/ifrt_ops.cc.inc"
      >();
  addInterfaces<IfrtAsmDialectInterface>();
}

IfrtAsmDialectInterface::AliasResult IfrtAsmDialectInterface::getAlias(
    mlir::Attribute attr, llvm::raw_ostream& os) const {
  if (llvm::isa<IfrtShardingParamAttr>(attr)) {
    os << "sp";
    return AliasResult::FinalAlias;
  }
  if (auto devices = llvm::dyn_cast<IfrtDevicesAttr>(attr);
      devices != nullptr && devices.getIds().size() > 4) {
    os << "devices";
    return AliasResult::FinalAlias;
  }
  if (auto mapping = llvm::dyn_cast<IfrtArrayMappingAttr>(attr);
      mapping != nullptr && mapping.getMappings().size() > 2) {
    os << "array_mapping";
    return AliasResult::FinalAlias;
  }
  return AliasResult::NoAlias;
}

namespace {

enum class RegionAttrKind { kArg, kResult };

// Verifies the common invariants shared by `ifrt.function` and
// `ifrt.reshard_function` attributes and returns the `func::FuncOp` on success.
mlir::FailureOr<mlir::func::FuncOp> VerifyIfrtFunctionCommon(
    mlir::Operation* op, mlir::NamedAttribute attr) {
  mlir::func::FuncOp func_op = llvm::dyn_cast<mlir::func::FuncOp>(op);
  if (func_op == nullptr) {
    return op->emitOpError() << "has `" << attr.getName().getValue()
                             << "` attr but is not a function";
  }
  if (!mlir::isa<mlir::UnitAttr>(attr.getValue())) {
    return op->emitOpError() << "has `" << attr.getName().getValue()
                             << "` attr that is not a UnitAttr";
  }
  if (op->hasAttr(kIfrtFunctionAttrName) &&
      op->hasAttr(kIfrtReshardFunctionAttrName)) {
    return op->emitOpError()
           << "cannot have both `" << kIfrtFunctionAttrName << "` and `"
           << kIfrtReshardFunctionAttrName << "` attrs";
  }
  if (!llvm::isa_and_nonnull<mlir::ModuleOp>(op->getParentOp())) {
    return op->emitOpError()
           << "with `" << attr.getName().getValue()
           << "` attr must be nested directly in a `builtin.module` op";
  }
  if (op->getNumRegions() != 1 || !func_op.getBody().hasOneBlock()) {
    return op->emitOpError() << "with `" << attr.getName().getValue()
                             << "` attr must have a single-block body";
  }
  mlir::TypeAttr func_type_attr = func_op.getFunctionTypeAttr();
  if (func_type_attr == nullptr || func_type_attr.getValue() == nullptr ||
      !mlir::isa<mlir::FunctionType>(func_type_attr.getValue())) {
    return op->emitOpError() << "with `" << attr.getName().getValue()
                             << "` attr must have a valid `FunctionType`";
  }
  mlir::Block& block = func_op.getBody().front();
  if (block.empty() || !llvm::isa<mlir::func::ReturnOp>(block.back())) {
    return op->emitOpError() << "with `" << attr.getName().getValue()
                             << "` attr must end with `func.return`";
  }
  return func_op;
}

// Verifies an `ifrt.function` operation attribute on `op`.
mlir::LogicalResult VerifyIfrtFunctionAttr(mlir::Operation* op,
                                           mlir::NamedAttribute attr) {
  mlir::FailureOr<mlir::func::FuncOp> func_op =
      VerifyIfrtFunctionCommon(op, attr);
  if (mlir::failed(func_op)) {
    return mlir::failure();
  }
  mlir::FunctionType func_type = func_op->getFunctionType();
  for (mlir::Type input_type : func_type.getInputs()) {
    if (!llvm::isa<IfrtArrayType, IfrtControlType>(input_type)) {
      return func_op->emitOpError()
             << "with `" << kIfrtFunctionAttrName
             << "` attr must have only `!ifrt.array` or `!ifrt.control` "
                "argument types, got "
             << input_type;
    }
  }
  for (mlir::Type result_type : func_type.getResults()) {
    if (!llvm::isa<IfrtArrayType, IfrtControlType>(result_type)) {
      return func_op->emitOpError()
             << "with `" << kIfrtFunctionAttrName
             << "` attr must have only `!ifrt.array` or `!ifrt.control` "
                "result types, got "
             << result_type;
    }
  }
  return mlir::success();
}

// Verifies an `ifrt.reshard_function` operation attribute on `op`.
mlir::LogicalResult VerifyIfrtReshardFunctionAttr(mlir::Operation* op,
                                                  mlir::NamedAttribute attr) {
  mlir::FailureOr<mlir::func::FuncOp> func_op =
      VerifyIfrtFunctionCommon(op, attr);
  if (mlir::failed(func_op)) {
    return mlir::failure();
  }
  mlir::FunctionType func_type = func_op->getFunctionType();
  for (mlir::Type input_type : func_type.getInputs()) {
    if (!llvm::isa<IfrtArrayType>(input_type)) {
      return func_op->emitOpError()
             << "with `" << kIfrtReshardFunctionAttrName
             << "` attr must have only `!ifrt.array` argument types, got "
             << input_type;
    }
  }
  for (mlir::Type result_type : func_type.getResults()) {
    if (!llvm::isa<IfrtArrayType>(result_type)) {
      return func_op->emitOpError()
             << "with `" << kIfrtReshardFunctionAttrName
             << "` attr must have only `!ifrt.array` result types, got "
             << result_type;
    }
  }
  for (mlir::Operation& body_op : func_op->getBody().front()) {
    if (!llvm::isa<ReshardOp, CopyArraysOp, RemapArraysOp, BitcastArraysOp,
                   AssembleOp, DisassembleOp, mlir::func::ReturnOp>(&body_op)) {
      return func_op->emitOpError()
             << "with `" << kIfrtReshardFunctionAttrName
             << "` attr may only contain resharding ops and `func.return`, "
                "got `"
             << body_op.getName() << "`";
    }
  }
  return mlir::success();
}

// Verifies that `attr` holds a non-empty `mlir::StringAttr`.
mlir::LogicalResult VerifyNonEmptyStringAttr(mlir::Operation* op,
                                             mlir::NamedAttribute attr) {
  mlir::StringAttr str_attr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
  if (str_attr == nullptr || str_attr.getValue().empty()) {
    return op->emitOpError() << "has `" << attr.getName().getValue()
                             << "` attr that is not a non-empty StringAttr";
  }
  return mlir::success();
}

// Verifies `ifrt.sharding`, `ifrt.devices`, and `ifrt.memory_kind` attributes
// attached to `func.func` region arguments or results. Other `ifrt.*`
// attributes are allowed, but not verified.
mlir::LogicalResult VerifyIfrtArgOrResultAttr(mlir::Operation* op,
                                              mlir::NamedAttribute attr,
                                              RegionAttrKind kind) {
  const llvm::StringRef attr_name = attr.getName().getValue();
  if (attr_name != kIfrtShardingAttrName && attr_name != kIfrtDevicesAttrName &&
      attr_name != kIfrtMemoryKindAttrName) {
    return mlir::success();
  }
  const llvm::StringRef attr_kind =
      kind == RegionAttrKind::kArg ? "arg" : "result";
  if (!llvm::isa<mlir::func::FuncOp>(op)) {
    return op->emitOpError() << "has `" << attr_name << "` " << attr_kind
                             << " attr but is not a function";
  }
  if (attr_name == kIfrtShardingAttrName) {
    if (!mlir::isa<IfrtShardingAttrInterface>(attr.getValue())) {
      return op->emitOpError()
             << "has `" << kIfrtShardingAttrName << "` " << attr_kind
             << " attr that is not an `IfrtShardingAttrInterface`";
    }
    return mlir::success();
  }
  if (attr_name == kIfrtDevicesAttrName) {
    if (!mlir::isa<IfrtDevicesAttr>(attr.getValue())) {
      return op->emitOpError()
             << "has `" << kIfrtDevicesAttrName << "` " << attr_kind
             << " attr that is not an `IfrtDevicesAttr`";
    }
    return mlir::success();
  }
  mlir::StringAttr str_attr = llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
  if (str_attr == nullptr || str_attr.getValue().empty()) {
    return op->emitOpError()
           << "has `" << kIfrtMemoryKindAttrName << "` " << attr_kind
           << " attr that is not a non-empty StringAttr";
  }
  return mlir::success();
}

}  // namespace

mlir::LogicalResult IfrtDialect::verifyOperationAttribute(
    mlir::Operation* op, mlir::NamedAttribute attr) {
  if (attr.getName() == kIfrtFunctionAttrName) {
    return VerifyIfrtFunctionAttr(op, attr);
  }
  if (attr.getName() == kIfrtReshardFunctionAttrName) {
    return VerifyIfrtReshardFunctionAttr(op, attr);
  }
  if (attr.getName() == kIfrtLocalViewAttrName) {
    if (!llvm::isa<CallOp, CallLoadedExecutableOp, mlir::ModuleOp>(op)) {
      return op->emitOpError()
             << "has `" << kIfrtLocalViewAttrName
             << "` attr but is not a `ifrt.Call`, `ifrt.CallLoadedExecutable`, "
                "or `builtin.module` op";
    }
    if (!mlir::isa<mlir::UnitAttr>(attr.getValue())) {
      return op->emitOpError() << "has `" << kIfrtLocalViewAttrName
                               << "` attr that is not a UnitAttr";
    }
    return mlir::success();
  }
  if (attr.getName() == kIfrtModuleTypeAttrName) {
    if (!llvm::isa<CallOp, CallLoadedExecutableOp>(op)) {
      return op->emitOpError() << "has `" << kIfrtModuleTypeAttrName
                               << "` attr but is not a `ifrt.Call` or "
                                  "`ifrt.CallLoadedExecutable` op";
    }
    mlir::StringAttr str_attr =
        llvm::dyn_cast<mlir::StringAttr>(attr.getValue());
    if (str_attr == nullptr ||
        (str_attr.getValue() != kIfrtModuleTypeXla &&
         str_attr.getValue() != kIfrtModuleTypeMpmdReshard)) {
      return op->emitOpError()
             << "has `" << kIfrtModuleTypeAttrName
             << "` attr that must be a StringAttr with value `"
             << kIfrtModuleTypeXla << "` or `" << kIfrtModuleTypeMpmdReshard
             << "`";
    }
    return mlir::success();
  }
  if (attr.getName() == kIfrtCompileOptionsKey ||
      attr.getName() == kIfrtMeshNameAttrName) {
    return VerifyNonEmptyStringAttr(op, attr);
  }
  if (attr.getName() == kIfrtNumDevicesAttrName) {
    mlir::IntegerAttr int_attr =
        llvm::dyn_cast<mlir::IntegerAttr>(attr.getValue());
    if (int_attr == nullptr || !int_attr.getType().isSignlessInteger(32) ||
        int_attr.getInt() <= 0) {
      return op->emitOpError()
             << "has `" << kIfrtNumDevicesAttrName
             << "` attr that is not a positive i32 IntegerAttr";
    }
    return mlir::success();
  }
  // Allow other `ifrt.*` attributes.
  return mlir::success();
}

mlir::LogicalResult IfrtDialect::verifyRegionArgAttribute(
    mlir::Operation* op, unsigned regionIndex, unsigned argIndex,
    mlir::NamedAttribute attr) {
  if (attr.getName() == kIfrtDonatedArgAttrName) {
    mlir::func::FuncOp func_op = llvm::dyn_cast<mlir::func::FuncOp>(op);
    if (func_op == nullptr) {
      return op->emitOpError() << "has `" << kIfrtDonatedArgAttrName
                               << "` arg attr but is not a function";
    }
    if (!mlir::isa<mlir::UnitAttr>(attr.getValue())) {
      return op->emitOpError() << "has `" << kIfrtDonatedArgAttrName
                               << "` arg attr that is not a UnitAttr";
    }
    if (!op->hasAttr(kIfrtFunctionAttrName)) {
      return op->emitOpError()
             << "has `" << kIfrtDonatedArgAttrName << "` arg attr but not has `"
             << kIfrtFunctionAttrName << "` attr";
    }
    if (!mlir::isa<IfrtArrayType>(
            func_op.getFunctionType().getInput(argIndex))) {
      return op->emitOpError()
             << "has `" << kIfrtDonatedArgAttrName << "` arg attr on argument #"
             << argIndex << " that is not an `!ifrt.array`";
    }
    return mlir::success();
  }
  return VerifyIfrtArgOrResultAttr(op, attr, RegionAttrKind::kArg);
}

mlir::LogicalResult IfrtDialect::verifyRegionResultAttribute(
    mlir::Operation* op, unsigned regionIndex, unsigned resultIndex,
    mlir::NamedAttribute attr) {
  if (attr.getName() == kIfrtDonatedArgAttrName) {
    return op->emitOpError()
           << "has `" << kIfrtDonatedArgAttrName
           << "` result attr, which is only valid on arguments";
  }
  return VerifyIfrtArgOrResultAttr(op, attr, RegionAttrKind::kResult);
}

//===----------------------------------------------------------------------===//
// IfrtShardingParamAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtShardingParamAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ShardingParam sharding_param) {
  return sharding_param.verify(emitError);
}

mlir::LogicalResult IfrtShardingParamAttr::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  return getSharding().CanApplyTo(emitError, shape, device_ids);
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtShardingParamAttr::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  return getSharding().GlobalShapeFromLocalShape(local_shape);
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtShardingParamAttr::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  return getSharding().LocalShapeFromGlobalShape(global_shape);
}

// Returns the number of devices the sharding applies to.
int IfrtShardingParamAttr::NumDevices() const {
  return getSharding().NumDevices();
};

//===----------------------------------------------------------------------===//
// IfrtUnspecifiedShardingAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtUnspecifiedShardingAttr::CanApplyTo(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, llvm::ArrayRef<int> device_ids) const {
  // The unspecified sharding can be applied to any array.
  return mlir::success();
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtUnspecifiedShardingAttr::GlobalShapeFromLocalShape(
    llvm::ArrayRef<int64_t> local_shape) const {
  // Unspecified sharding does not change the shape.
  llvm::SmallVector<int64_t> global_shape(local_shape.begin(),
                                          local_shape.end());
  return global_shape;
}

absl::StatusOr<llvm::SmallVector<int64_t>>
IfrtUnspecifiedShardingAttr::LocalShapeFromGlobalShape(
    llvm::ArrayRef<int64_t> global_shape) const {
  // Unspecified sharding does not change the shape.
  llvm::SmallVector<int64_t> local_shape(global_shape.begin(),
                                         global_shape.end());
  return local_shape;
}

int IfrtUnspecifiedShardingAttr::NumDevices() const { return 0; }

//===----------------------------------------------------------------------===//
// IfrtArrayType
//===----------------------------------------------------------------------===//

// Returns an array of logical device ids.
llvm::ArrayRef<int> IfrtArrayType::getDevices() const {
  return getDevicesAttr().getIds();
}

mlir::LogicalResult IfrtArrayType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::RankedTensorType shape, IfrtShardingAttrInterface sharding_attr,
    IfrtDevicesAttr devices_attr, mlir::StringAttr memory_kind_attr,
    mlir::StringAttr layout_attr) {
  if (shape == nullptr || sharding_attr == nullptr || devices_attr == nullptr) {
    return emitError() << "expected non-null IfrtArrayType parameters";
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
    if (dim == 0) {
      continue;
    }
    if (__builtin_mul_overflow(num_elements, dim, &num_elements) ||
        num_elements > Shape::kMaxNumElements) {
      return emitError()
             << "Array shape element count exceeds maximum supported elements ("
             << Shape::kMaxNumElements << "): " << shape;
    }
  }
  if (llvm::isa<IfrtTokenType>(shape.getElementType())) {
    if (!shape.getShape().empty()) {
      return emitError() << "Token type cannot have non-empty shape";
    }
  } else if (!llvm::isa<IfrtStringType>(shape.getElementType()) &&
             shape.getElementType().getDialect().getNamespace() !=
                 mlir::BuiltinDialect::getDialectNamespace()) {
    return emitError() << "Unsupported array element type: "
                       << shape.getElementType();
  }
  if (memory_kind_attr != nullptr && memory_kind_attr.getValue().empty()) {
    return emitError() << "Memory kind cannot be empty";
  }
  if (layout_attr != nullptr) {
    absl::StatusOr<xla::LayoutMode> layout_mode =
        xla::LayoutMode::FromString(layout_attr.str());
    if (!layout_mode.ok()) {
      return emitError() << "Invalid layout mode: "
                         << layout_mode.status().message();
    }
  }
  return sharding_attr.CanApplyTo(emitError, shape, devices_attr.getIds());
}

xla::ifrt::MemoryKind IfrtArrayType::MemoryKind() const {
  static const absl::NoDestructor<std::string> default_memory_kind(
      absl::StrCat(xla::ifrt::MemoryKind()));
  mlir::StringAttr memory_kind_attr = getMemoryKindAttr();
  return memory_kind_attr == nullptr ||
                 memory_kind_attr.getValue() == *default_memory_kind
             ? xla::ifrt::MemoryKind()
             : xla::ifrt::MemoryKind(memory_kind_attr.str());
};

// TODO(icgog): Migrate to xla::ifrt::Layout.
xla::LayoutMode IfrtArrayType::LayoutMode() const {
  if (auto layout_attr = getLayoutAttr()) {
    auto layout_mode = xla::LayoutMode::FromString(layout_attr.str());
    CHECK_OK(layout_mode) << "Invalid layout mode: " << layout_attr.str();
    return *layout_mode;
  }
  return xla::LayoutMode(xla::LayoutMode::Mode::kDefault);
}

absl::StatusOr<llvm::SmallVector<int64_t>> IfrtArrayType::LocalShape() const {
  return getShardingAttr().LocalShapeFromGlobalShape(getShape().getShape());
}

bool IfrtArrayType::HasSameLayout(IfrtArrayType other) const {
  if (getLayoutAttr() == other.getLayoutAttr()) {
    return true;
  }
  xla::LayoutMode lhs_mode = LayoutMode();
  xla::LayoutMode rhs_mode = other.LayoutMode();
  if (lhs_mode.mode != rhs_mode.mode) {
    return false;
  }
  if (lhs_mode.mode == xla::LayoutMode::Mode::kUserSpecified) {
    return lhs_mode.user_layout == rhs_mode.user_layout;
  }
  return true;
}

void IfrtArrayType::print(mlir::AsmPrinter& odsPrinter) const {
  mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getShape());
  odsPrinter << ", ";
  odsPrinter.printStrippedAttrOrType(getShardingAttr());
  odsPrinter << ", ";
  odsPrinter.printStrippedAttrOrType(getDevicesAttr());
  if (getMemoryKindAttr()) {
    odsPrinter << ", memory_kind = ";
    odsPrinter.printStrippedAttrOrType(getMemoryKindAttr());
  }
  if (getLayoutAttr()) {
    odsPrinter << ", layout = ";
    odsPrinter.printStrippedAttrOrType(getLayoutAttr());
  }
  odsPrinter << ">";
}

mlir::FailureOr<mlir::StringAttr> parseMemoryKindAttr(
    mlir::AsmParser& odsParser) {
  if (mlir::failed(odsParser.parseOptionalKeyword("memory_kind"))) {
    return mlir::failure();
  }
  if (mlir::failed(odsParser.parseEqual())) {
    return mlir::failure();
  }
  auto memory_kind_attr_or =
      mlir::FieldParser<mlir::StringAttr>::parse(odsParser);
  if (mlir::failed(memory_kind_attr_or)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse Ifrt_ArrayType parameter 'memory_kind_attr' which "
        "is to be a `mlir::StringAttr`");
    return mlir::failure();
  }
  return memory_kind_attr_or;
}

mlir::FailureOr<mlir::StringAttr> parseLayoutAttr(mlir::AsmParser& odsParser) {
  if (mlir::failed(odsParser.parseOptionalKeyword("layout"))) {
    return mlir::failure();
  }
  if (mlir::failed(odsParser.parseEqual())) {
    return mlir::failure();
  }
  auto layout_attr_or = mlir::FieldParser<mlir::StringAttr>::parse(odsParser);
  if (mlir::failed(layout_attr_or)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse Ifrt_ArrayType parameter 'layout_attr' which is to be "
        "a `mlir::StringAttr`");
    return mlir::failure();
  }
  return layout_attr_or;
}

mlir::Type IfrtArrayType::parse(mlir::AsmParser& odsParser) {
  mlir::Builder odsBuilder(odsParser.getContext());

  if (mlir::failed(odsParser.parseLess())) {
    return {};
  }

  auto shape_or = mlir::FieldParser<mlir::RankedTensorType>::parse(odsParser);
  if (mlir::failed(shape_or)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
                        "failed to parse Ifrt_ArrayType parameter 'shape' "
                        "which is to be a `mlir::RankedTensorType`");
    return {};
  }

  if (mlir::failed(odsParser.parseComma())) {
    return {};
  }

  auto sharding_attr_or =
      mlir::FieldParser<IfrtShardingAttrInterface>::parse(odsParser);
  if (mlir::failed(sharding_attr_or)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse Ifrt_ArrayType parameter 'sharding_attr' which is to "
        "be a `IfrtShardingAttrInterface`");
    return {};
  }

  if (mlir::failed(odsParser.parseComma())) {
    return {};
  }

  auto devices_attr_or = mlir::FieldParser<IfrtDevicesAttr>::parse(odsParser);
  if (mlir::failed(devices_attr_or)) {
    odsParser.emitError(
        odsParser.getCurrentLocation(),
        "failed to parse Ifrt_ArrayType parameter 'devices_attr' which is to "
        "be a `IfrtDevicesAttr`");
    return {};
  }

  mlir::FailureOr<mlir::StringAttr> memory_kind_attr_or;
  mlir::FailureOr<mlir::StringAttr> layout_attr_or;
  if (mlir::succeeded(odsParser.parseOptionalComma())) {
    memory_kind_attr_or = parseMemoryKindAttr(odsParser);
    if (mlir::failed(memory_kind_attr_or)) {
      layout_attr_or = parseLayoutAttr(odsParser);
      if (mlir::failed(layout_attr_or)) {
        odsParser.emitError(
            odsParser.getCurrentLocation(),
            "failed to parse Ifrt_ArrayType optional attributes");
        return {};
      }
    }
    if (mlir::succeeded(odsParser.parseOptionalComma())) {
      layout_attr_or = parseLayoutAttr(odsParser);
      if (mlir::failed(layout_attr_or)) {
        odsParser.emitError(
            odsParser.getCurrentLocation(),
            "failed to parse Ifrt_ArrayType `layout` attributes");
        return {};
      }
    }
  }

  if (mlir::failed(odsParser.parseGreater())) {
    return {};
  }

  return odsParser.getChecked<IfrtArrayType>(
      odsParser.getCurrentLocation(), odsParser.getContext(), *shape_or,
      *sharding_attr_or, *devices_attr_or,
      memory_kind_attr_or.value_or(nullptr), layout_attr_or.value_or(nullptr));
}

//===----------------------------------------------------------------------===//
// IfrtDevicesAttr
//===----------------------------------------------------------------------===//

IfrtDevicesAttr::operator llvm::ArrayRef<int>() const { return getIds(); }

mlir::LogicalResult IfrtDevicesAttr::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
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

//===----------------------------------------------------------------------===//
// IfrtIntervalAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtIntervalAttr::verify(
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

int IfrtIntervalAttr::size() const {
  return 1 + (getEnd() - getStart() - 1) / getStep();
}

//===----------------------------------------------------------------------===//
// IfrtMappingAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtMappingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    IfrtIntervalAttr from_shards, IfrtIntervalAttr to_shards) {
  if (from_shards == nullptr || to_shards == nullptr) {
    return emitError() << "expected non-null interval attributes";
  }
  // Verify that from and to contains the same number of shards.
  if (from_shards.size() != to_shards.size()) {
    return emitError() << "from has " << from_shards.size() << " and to has "
                       << to_shards.size()
                       << ", but they must have the same number of shards.";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// IfrtArrayMappingAttr
//===----------------------------------------------------------------------===//

mlir::LogicalResult IfrtArrayMappingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int32_t in_array_index, int32_t out_array_index, mlir::ArrayAttr mappings) {
  if (in_array_index < 0 || out_array_index < 0) {
    return emitError()
           << "in_array_index and out_array_index must be non-negative";
  }
  if (mappings == nullptr || mappings.empty() ||
      !llvm::all_of(mappings.getValue(), llvm::IsaPred<IfrtMappingAttr>)) {
    return emitError() << "expected non-empty array of IfrtMappingAttr";
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

bool IsIfrtFunction(mlir::Operation* op) {
  return op->hasAttr(kIfrtFunctionAttrName) ||
         op->hasAttr(kIfrtReshardFunctionAttrName);
}

IfrtArrayType GetArrayType(mlir::Type type) {
  auto ifrt_array_type = mlir::dyn_cast<IfrtArrayType>(type);
  CHECK(ifrt_array_type != nullptr)
      << "Expecte 'IfrtArrayType', but got `" << mlir::debugString(type) << "`";
  return ifrt_array_type;
}

IfrtArrayType GetArrayType(mlir::Value value) {
  return GetArrayType(value.getType());
}

IfrtShardingParamAttr GetShardingParamAttr(IfrtArrayType array_type) {
  auto sharding_attr =
      mlir::dyn_cast<IfrtShardingParamAttr>(array_type.getShardingAttr());
  CHECK(sharding_attr != nullptr)
      << "Array type sharding attribute: " << mlir::debugString(array_type)
      << " if not of type `IfrtShardingParamAttr`";
  return sharding_attr;
}

bool IsUnspecifiedSharding(IfrtShardingAttrInterface sharding_attr) {
  return mlir::isa<IfrtUnspecifiedShardingAttr>(sharding_attr);
}

mlir::LogicalResult VerifyPerShardShapesAreEqual(mlir::Operation* op,
                                                 IfrtArrayType lhs,
                                                 IfrtArrayType rhs) {
  absl::StatusOr<llvm::SmallVector<int64_t>> lhs_shard_shape = lhs.LocalShape();
  if (!lhs_shard_shape.ok()) {
    return op->emitOpError() << "unable to get per-shard shape of array " << lhs
                             << ": " << lhs_shard_shape.status().message();
  }
  absl::StatusOr<llvm::SmallVector<int64_t>> rhs_shard_shape = rhs.LocalShape();
  if (!rhs_shard_shape.ok()) {
    return op->emitOpError() << "unable to get per-shard shape of array " << rhs
                             << ": " << rhs_shard_shape.status().message();
  }
  if (*lhs_shard_shape != *rhs_shard_shape) {
    return op->emitOpError()
           << "Arrays have different per-shard shapes: " << lhs << " vs. "
           << rhs;
  }
  return mlir::success();
}

}  // namespace ifrt
}  // namespace xla
