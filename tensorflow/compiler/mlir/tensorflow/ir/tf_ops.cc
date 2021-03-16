/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Identifier.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DecodeAttributesInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

// These are currently aliases and the alias will be removed, verified
// equivalent until then.
// TODO(b/178519687): Remvoe once addressed.
static_assert(std::is_same<tensorflow::int64, std::int64_t>::value,
              "tensorflow::int64 is expected to match std::int64_t");

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TF Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct TFConstantFoldInterface : public DialectFoldInterface {
  TFConstantFoldInterface(Dialect *dialect) : DialectFoldInterface(dialect) {}
  LogicalResult fold(Operation *op, ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results) const final {
    return TensorFlowDialect::constantFold(op, operands, results);
  }
};

struct TFDecodeAttributesInterface : public DialectDecodeAttributesInterface {
  TFDecodeAttributesInterface(Dialect *dialect)
      : DialectDecodeAttributesInterface(dialect) {}
  LogicalResult decode(OpaqueElementsAttr input, ElementsAttr &output) const {
    return TensorFlowDialect::decode(input, output);
  }
};

struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Returns if it's legal to inline 'callable' into the 'call', where 'call' is
  // a TF operation.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Check that the TF call operation is one that is legal to inline.
    return !isa<TPUPartitionedCallOp>(call);
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF operation.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
    // Allow inlining in regions attached to region based control flow
    // operations only if the src region is a single block region
    return isa<IfRegionOp, WhileRegionOp>(dest->getParentOp()) &&
           llvm::hasSingleElement(*src);
  }

  // Returns true if its legal to inline a TF operation `op` into the `dest`
  // region.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
    // An op is legal to inline if either of the following conditions is true:
    // (a) Its legal to duplicate the Op.
    // (b) The Op is inside a single use function. If that function is inlined,
    //     post inlining, the function will be dead and eliminated from the IR.
    //     So there won't be any code duplication.
    // plus the function caller op can be replaced by inlined ops.
    return !wouldBeCloned || TensorFlowDialect::CanDuplicate(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

// Returns true if the op can be duplicated.
bool TensorFlowDialect::CanDuplicate(Operation *op) {
  // If the op is marked with the cannot duplicate trait, it cannot be
  // duplicated.
  if (op->hasTrait<OpTrait::TF::CannotDuplicate>()) return false;

  // If the op has no memory side effects, it can be duplicated.
  if (MemoryEffectOpInterface::hasNoEffect(op)) return true;

  // If the op is marked stateless using the `is_stateless` attribute, that
  // attribute determines if the op can be duplicated.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return is_stateless.getValue();

  // Otherwise, assume ops can be duplicated by default if its registered, else
  // it cannot be for unknown ops.
  return op->isRegistered();
}

// Returns true if the op can have side effects.
bool TensorFlowDialect::CanHaveSideEffects(Operation *op) {
  // If the op has no memory side effects, it has no side effects
  if (MemoryEffectOpInterface::hasNoEffect(op)) return false;

  // If the op is marked stateless using the `is_stateless` attribute, then
  // it has no side effects.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return !is_stateless.getValue();

  // Terminators defined in the TF dialect do not have side effects.
  if (op->hasTrait<OpTrait::IsTerminator>()) return false;

  // Otherwise assume that the op can have side effects.
  return true;
}

std::vector<TensorFlowDialect::AdditionalOpFunction>
    *TensorFlowDialect::GetAdditionalOperationHooks() {
  static auto *const additional_operation_hooks =
      new std::vector<TensorFlowDialect::AdditionalOpFunction>();
  return additional_operation_hooks;
}

TensorFlowDialect::ConstantFoldHook TensorFlowDialect::constant_fold_hook_;
TensorFlowDialect::DecodeConstantHook TensorFlowDialect::decode_constant_hook_;

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context, TypeID::get<TensorFlowDialect>()) {
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_all_ops.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.cc.inc"
      >();
  registerTypes();
  addInterfaces<TFInlinerInterface, TFDecodeAttributesInterface,
                TFConstantFoldInterface>();
  registerAttributes();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  for (const auto &hook : *GetAdditionalOperationHooks()) {
    hook(*this);
  }
}

namespace {

ShapeAttr ParseShapeAttr(MLIRContext *context, StringRef spec, Location loc) {
  auto emit_error = [&, spec]() {
    emitError(loc, "invalid TensorFlow shape attribute: ") << spec;
    return nullptr;
  };

  if (!spec.consume_front("shape<")) return emit_error();

  if (spec.consume_front("*>"))
    return mlir::TF::ShapeAttr::get(context, llvm::None);

  SmallVector<int64_t, 4> shape;
  while (!spec.consume_front(">")) {
    int64_t dim;

    if (spec.consume_front("?"))
      dim = -1;
    else if (spec.consumeInteger(10, dim) || dim < 0)
      return emit_error();

    spec.consume_front("x");

    shape.push_back(dim);
  }

  return mlir::TF::ShapeAttr::get(context, llvm::makeArrayRef(shape));
}

void PrintShapeAttr(ShapeAttr attr, DialectAsmPrinter &os) {  // NOLINT
  os << "shape";

  os << "<";
  if (attr.hasRank()) {
    auto print_dim = [&](int64_t dim) {
      if (dim > -1)
        os << dim;
      else
        os << "?";
    };
    llvm::interleave(attr.getShape(), os, print_dim, "x");
  } else {
    os << "*";
  }
  os << ">";
}

// Parses a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
//
// where the first element is a SymbolRefAttr and the second element is a
// DictionaryAttr.
FuncAttr ParseFuncAttr(MLIRContext *context, StringRef spec, Location loc) {
  auto emit_error = [&, spec]() {
    emitError(loc, "invalid TensorFlow func attribute: ") << spec;
    return nullptr;
  };

  if (!spec.consume_front("func<")) return emit_error();

  size_t func_name_num_read = 0;
  Attribute func_name_attr =
      mlir::parseAttribute(spec, context, func_name_num_read);
  if (!func_name_attr || !func_name_attr.isa<SymbolRefAttr>())
    return emit_error();
  spec = spec.drop_front(func_name_num_read);

  if (!spec.consume_front(", ")) return emit_error();

  size_t func_attrs_num_read = 0;
  Attribute func_attrs_attr =
      mlir::parseAttribute(spec, context, func_attrs_num_read);
  if (!func_attrs_attr || !func_attrs_attr.isa<DictionaryAttr>())
    return emit_error();
  spec = spec.drop_front(func_attrs_num_read);

  if (!spec.consume_front(">")) return emit_error();

  return mlir::TF::FuncAttr::get(context, func_name_attr.cast<SymbolRefAttr>(),
                                 func_attrs_attr.cast<DictionaryAttr>());
}

// Prints a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
void PrintFuncAttr(FuncAttr attr, DialectAsmPrinter &os) {
  os << "func<" << attr.GetName() << ", " << attr.GetAttrs() << ">";
}

}  // namespace

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
  auto spec = parser.getFullSymbolSpec();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  if (spec.startswith("shape")) return ParseShapeAttr(getContext(), spec, loc);

  if (spec.startswith("func")) return ParseFuncAttr(getContext(), spec, loc);

  return (emitError(loc, "unknown TensorFlow attribute: " + spec), nullptr);
}

void TensorFlowDialect::printAttribute(Attribute attr,
                                       DialectAsmPrinter &os) const {
  if (auto shape_attr = attr.dyn_cast<ShapeAttr>())
    PrintShapeAttr(shape_attr, os);
  else if (auto func_attr = attr.dyn_cast<FuncAttr>())
    PrintFuncAttr(func_attr, os);
  else
    llvm_unreachable("unexpected tensorflow attribute type");
}

// Parses a type registered to this dialect.
Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
  StringRef data;
  if (parser.parseKeyword(&data)) return Type();

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  if (data == name) return tftype##Type::get(getContext());
// Custom TensorFlow types are handled separately at the end as they do partial
// match.
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

  llvm::SMLoc loc = parser.getNameLoc();
  if (data.startswith("resource")) {
    Type ret = ParseResourceType(parser);
    if (!ret) parser.emitError(loc, "invalid resource type");
    return ret;
  }
  if (data.startswith("variant")) {
    Type ret = ParseVariantType(parser);
    if (!ret) parser.emitError(loc, "invalid variant type");
    return ret;
  }
  return (parser.emitError(loc, "unknown TensorFlow type: " + data), nullptr);
}

// Prints a type registered to this dialect.
void TensorFlowDialect::printType(Type ty, DialectAsmPrinter &os) const {
  assert(ty.isa<TensorFlowType>());
#define HANDLE_TF_TYPE(tftype, enumerant, name)        \
  if (auto derived_ty = ty.dyn_cast<tftype##Type>()) { \
    os << name;                                        \
    return;                                            \
  }
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name) \
  if (auto derived_ty = ty.dyn_cast<tftype##Type>()) { \
    Print##tftype##Type(derived_ty, os);               \
    return;                                            \
  }
// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

  llvm_unreachable("unexpected tensorflow type kind");
}

namespace {
template <typename TypeWithSubtype>
Type ParseTypeWithSubtype(MLIRContext *context, DialectAsmParser &parser) {
  // Default type without inferred subtypes.
  if (failed(parser.parseOptionalLess())) return TypeWithSubtype::get(context);

  // Most types with subtypes have only one subtype.
  SmallVector<TensorType, 1> subtypes;
  do {
    TensorType tensor_ty;
    if (parser.parseType(tensor_ty)) return Type();

    // Each of the subtypes should be a valid TensorFlow type.
    // TODO(jpienaar): Remove duplication.
    if (!IsValidTFTensorType(tensor_ty)) {
      parser.emitError(parser.getNameLoc()) << "invalid subtype: " << tensor_ty;
      return Type();
    }
    subtypes.push_back(tensor_ty);
  } while (succeeded(parser.parseOptionalComma()));

  if (parser.parseGreater()) return Type();

  return TypeWithSubtype::get(subtypes, context);
}

template <typename TypeWithSubtype>
void PrintTypeWithSubtype(StringRef type, TypeWithSubtype ty,
                          DialectAsmPrinter &os) {
  os << type;
  ArrayRef<TensorType> subtypes = ty.getSubtypes();
  if (subtypes.empty()) return;

  os << "<";
  interleaveComma(subtypes, os);
  os << ">";
}
}  // anonymous namespace

Type TensorFlowDialect::ParseResourceType(DialectAsmParser &parser) const {
  return ParseTypeWithSubtype<ResourceType>(getContext(), parser);
}

void TensorFlowDialect::PrintResourceType(ResourceType ty,
                                          DialectAsmPrinter &os) const {
  return PrintTypeWithSubtype("resource", ty, os);
}

Type TensorFlowDialect::ParseVariantType(DialectAsmParser &parser) const {
  return ParseTypeWithSubtype<VariantType>(getContext(), parser);
}

void TensorFlowDialect::PrintVariantType(VariantType ty,
                                         DialectAsmPrinter &os) const {
  return PrintTypeWithSubtype("variant", ty, os);
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
  return builder.create<ConstOp>(loc, type, value);
}

}  // namespace TF
}  // namespace mlir
