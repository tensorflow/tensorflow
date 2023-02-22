/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/types/dialect.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/escaping.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

#define GET_ATTRDEF_CLASSES
#include "tensorflow/core/ir/types/attributes.cc.inc"
#include "tensorflow/core/ir/types/attributes_enum.cc.inc"

#define GET_TYPEDEF_CLASSES
#include "tensorflow/core/ir/types/types.cc.inc"

// Generated definitions.
#include "tensorflow/core/ir/types/dialect.cpp.inc"

namespace mlir {
namespace tf_type {

//===----------------------------------------------------------------------===//
// TFType dialect.
//===----------------------------------------------------------------------===//

// Dialect construction: there is one instance per context and it registers its
// operations, types, and interfaces here.
void TFTypeDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tensorflow/core/ir/types/attributes.cc.inc"
      >();
  addTypes<ControlType, OpaqueTensorType,
#define HANDLE_TF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
#include "tensorflow/core/ir/types/types.def"
           >();
}

namespace {
template <typename TypeWithSubtype>
Type ParseTypeWithSubtype(MLIRContext* context, DialectAsmParser& parser) {
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
                          DialectAsmPrinter& os) {
  os << type;
  ArrayRef<TensorType> subtypes = ty.getSubtypes();
  if (subtypes.empty()) return;

  os << "<";
  interleaveComma(subtypes, os);
  os << ">";
}
Type ParseResourceType(MLIRContext* context, DialectAsmParser& parser) {
  return ParseTypeWithSubtype<ResourceType>(context, parser);
}

void PrintResourceType(ResourceType ty, DialectAsmPrinter& os) {
  return PrintTypeWithSubtype("resource", ty, os);
}

Type ParseVariantType(MLIRContext* context, DialectAsmParser& parser) {
  return ParseTypeWithSubtype<VariantType>(context, parser);
}

void PrintVariantType(VariantType ty, DialectAsmPrinter& os) {
  return PrintTypeWithSubtype("variant", ty, os);
}

}  // namespace

// Entry point for Type parsing, TableGen generated code will handle the
// dispatch to the individual classes.
Type TFTypeDialect::parseType(DialectAsmParser& parser) const {
  StringRef type_tag;
  llvm::SMLoc loc = parser.getNameLoc();

  Type genType;
  auto parse_result = generatedTypeParser(parser, &type_tag, genType);
  if (parse_result.has_value()) return genType;

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  if (type_tag == name) return tftype##Type::get(getContext());
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE: intended redundant include.
#include "tensorflow/core/ir/types/types.def"

  if (type_tag.startswith("resource")) {
    Type ret = ParseResourceType(getContext(), parser);
    if (!ret) parser.emitError(loc, "invalid resource type");
    return ret;
  }
  if (type_tag.startswith("variant")) {
    Type ret = ParseVariantType(getContext(), parser);
    if (!ret) parser.emitError(loc, "invalid variant type");
    return ret;
  }

  parser.emitError(parser.getNameLoc(),
                   "unknown type in TF graph dialect: " + type_tag);
  return {};
}

// Entry point for Type parsing, TableGen generated code will handle the
// dispatch to the individual classes.
void TFTypeDialect::printType(Type type, DialectAsmPrinter& printer) const {
#define HANDLE_TF_TYPE(tftype, enumerant, name)          \
  if (auto derived_ty = type.dyn_cast<tftype##Type>()) { \
    printer << name;                                     \
    return;                                              \
  }
#define HANDLE_CUSTOM_TF_TYPE(tftype, enumerant, name)   \
  if (auto derived_ty = type.dyn_cast<tftype##Type>()) { \
    Print##tftype##Type(derived_ty, printer);            \
    return;                                              \
  }
// NOLINTNEXTLINE: intended redundant include.
#include "tensorflow/core/ir/types/types.def"

  if (failed(generatedTypePrinter(type, printer)))
    llvm::report_fatal_error("unexpected tensorflow graph type kind");
}

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

Attribute VersionAttr::parse(AsmParser& parser, Type) {
  if (failed(parser.parseLess())) return {};

  int32_t producer, min_consumer;
  if (parser.parseKeyword("producer", " in tf_type version") ||
      parser.parseEqual() || parser.parseInteger(producer) ||
      parser.parseComma() ||
      parser.parseKeyword("min_consumer", " in tf_type version") ||
      parser.parseEqual() || parser.parseInteger(min_consumer))
    return {};

  SmallVector<int32_t, 4> bad_consumers;
  if (!parser.parseOptionalComma()) {
    if (parser.parseKeyword("bad_consumers", " in tf_type version") ||
        parser.parseEqual() || parser.parseLSquare())
      return {};
    do {
      int32_t bad_consumer;
      if (parser.parseInteger(bad_consumer)) return {};
      bad_consumers.push_back(bad_consumer);
    } while (!parser.parseOptionalComma());
    if (parser.parseRSquare()) return {};
  }
  if (failed(parser.parseGreater())) return {};

  return VersionAttr::get(parser.getContext(), producer, min_consumer,
                          bad_consumers);
}

void VersionAttr::print(AsmPrinter& printer) const {
  llvm::raw_ostream& os = printer.getStream();
  os << "<producer = " << getProducer()
     << ", min_consumer = " << getMinConsumer();
  ArrayRef<int32_t> badConsumers = getBadConsumers();
  if (!badConsumers.empty()) {
    os << ", bad_consumers = [";
    llvm::interleaveComma(badConsumers, os);
    os << "]";
  }
  os << ">";
}

FailureOr<FullTypeAttr> RawFullTypeAttrParser(AsmParser& parser) {
  SmallVector<FullTypeAttr> args;

  // Parse variable 'type_id'
  llvm::StringRef type_id_str;
  if (failed(parser.parseKeyword(&type_id_str))) {
    parser.emitError(
        parser.getCurrentLocation(),
        "failed to parse TFType_FullTypeAttr parameter keyword for "
        "'type_id'");
    return failure();
  }
  std::optional<FullTypeId> type_id = symbolizeFullTypeId(type_id_str);
  if (!type_id) {
    parser.emitError(parser.getCurrentLocation(),
                     "failed to parse TFType_FullTypeAttr parameter "
                     "'type_id'");
    return failure();
  }

  // Parse variable 'args'
  if (parser.parseCommaSeparatedList(AsmParser::Delimiter::OptionalLessGreater,
                                     [&]() {
                                       FailureOr<tf_type::FullTypeAttr> arg =
                                           RawFullTypeAttrParser(parser);
                                       if (failed(arg)) return failure();
                                       args.push_back(*arg);
                                       return success();
                                     }))
    return failure();

  // Parse variable 'attr'
  Attribute attr;
  parser.parseOptionalAttribute(attr);
  return FullTypeAttr::get(parser.getContext(), static_cast<int32_t>(*type_id),
                           args, attr);
}

Attribute FullTypeAttr::parse(AsmParser& parser, Type odsType) {
  if (failed(parser.parseLess())) return {};
  FailureOr<tf_type::FullTypeAttr> ret = RawFullTypeAttrParser(parser);
  if (succeeded(ret) && failed(parser.parseGreater())) return {};
  return ret.value_or(FullTypeAttr());
}

static void RawFullTypeAttrPrint(FullTypeAttr tfattr, AsmPrinter& printer) {
  printer << stringifyFullTypeId(tf_type::FullTypeId(tfattr.getTypeId()));
  if (!tfattr.getArgs().empty()) {
    printer << "<";
    llvm::interleaveComma(tfattr.getArgs(), printer, [&](Attribute arg) {
      if (auto t = arg.dyn_cast<FullTypeAttr>())
        RawFullTypeAttrPrint(t, printer);
      else
        printer << "<<INVALID ARG>>";
    });
    printer << ">";
  }
  if (tfattr.getAttr()) {
    printer << ' ';
    printer.printStrippedAttrOrType(tfattr.getAttr());
  }
}

void FullTypeAttr::print(AsmPrinter& printer) const {
  printer << "<";
  RawFullTypeAttrPrint(*this, printer);
  printer << ">";
}

// Print a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
// or
//   #tf.func<"", {attr = "value"}>
// in case of null symbol ref.
void FuncAttr::print(AsmPrinter& os) const {
  if (getName().getRootReference().getValue().empty())
    os << "<\"\", " << getAttrs() << ">";
  else
    os << "<" << getName() << ", " << getAttrs() << ">";
}

// Parses a #tf.func attribute of the following format:
//
//   #tf.func<@symbol, {attr = "value"}>
//
// where the first element is a SymbolRefAttr and the second element is a
// DictionaryAttr.
Attribute FuncAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  llvm::SMLoc loc = parser.getCurrentLocation();
  Attribute name, dict;
  if (failed(parser.parseAttribute(name))) {
    parser.emitError(loc) << "expected symbol while parsing tf.func attribute";
    return {};
  }
  if (auto func_name_str = name.dyn_cast<StringAttr>()) {
    if (!func_name_str.getValue().empty()) {
      parser.emitError(loc)
          << "expected empty string or symbol while parsing tf.func "
             "attribute";
      return {};
    }
    name = SymbolRefAttr::get(parser.getContext(), "");
  }
  if (!name.isa<SymbolRefAttr>()) {
    parser.emitError(loc) << "expected symbol while parsing tf.func attribute";
    return {};
  }
  if (failed(parser.parseComma())) return {};
  loc = parser.getCurrentLocation();
  if (failed(parser.parseAttribute(dict)) || !dict.isa<DictionaryAttr>()) {
    parser.emitError(loc)
        << "expected Dictionary attribute while parsing tf.func attribute";
    return {};
  }
  if (failed(parser.parseGreater())) return {};
  return FuncAttr::get(parser.getContext(), name.cast<SymbolRefAttr>(),
                       dict.cast<DictionaryAttr>());
}

void PlaceholderAttr::print(AsmPrinter& os) const {
  os << "<" << StringAttr::get(getContext(), getValue()) << ">";
}

Attribute PlaceholderAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};
  std::string content;
  if (failed(parser.parseOptionalString(&content))) {
    parser.emitError(parser.getCurrentLocation())
        << "expected string while parsing tf.placeholder attribute";
    return {};
  }
  if (failed(parser.parseGreater())) return {};
  return PlaceholderAttr::get(parser.getContext(), content);
}

void ShapeAttr::print(AsmPrinter& os) const {
  os << "<";
  if (hasRank()) {
    auto print_dim = [&](int64_t dim) {
      if (dim != ShapedType::kDynamic)
        os << dim;
      else
        os << "?";
    };
    llvm::interleave(getShape(), os, print_dim, "x");
  } else {
    os << "*";
  }
  os << ">";
}

Attribute ShapeAttr::parse(AsmParser& parser, Type type) {
  if (failed(parser.parseLess())) return {};

  if (succeeded(parser.parseOptionalStar())) {
    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getCurrentLocation())
          << "expected `>` after `*` when parsing a tf.shape "
             "attribute";
      return {};
    }
    return ShapeAttr::get(parser.getContext(), std::nullopt);
  }

  SmallVector<int64_t> shape;
  if (failed(parser.parseOptionalGreater())) {
    auto parse_element = [&]() {
      shape.emplace_back();
      llvm::SMLoc loc = parser.getCurrentLocation();
      if (succeeded(parser.parseOptionalQuestion())) {
        shape.back() = ShapedType::kDynamic;
      } else if (failed(parser.parseInteger(shape.back()))) {
        parser.emitError(loc)
            << "expected an integer or `?` when parsing a tf.shape attribute";
        return failure();
      }
      return success();
    };
    if (failed(parse_element())) return {};
    while (failed(parser.parseOptionalGreater())) {
      if (failed(parser.parseXInDimensionList()) || failed(parse_element()))
        return {};
    }
  }
  return ShapeAttr::get(parser.getContext(), llvm::ArrayRef(shape));
}

// Get or create a shape attribute.
ShapeAttr ShapeAttr::get(MLIRContext* context,
                         llvm::Optional<ArrayRef<int64_t>> shape) {
  if (shape) return Base::get(context, *shape, /*unranked=*/false);

  return Base::get(context, ArrayRef<int64_t>(), /*unranked=*/true);
}

// Get or create a shape attribute.
ShapeAttr ShapeAttr::get(MLIRContext* context, ShapedType shaped_type) {
  if (shaped_type.hasRank())
    return Base::get(context, shaped_type.getShape(), /*unranked=*/false);

  return Base::get(context, ArrayRef<int64_t>(), /*unranked=*/true);
}

llvm::Optional<ArrayRef<int64_t>> ShapeAttr::getValue() const {
  if (hasRank()) return getShape();
  return std::nullopt;
}

bool ShapeAttr::hasRank() const { return !getImpl()->unranked; }

int64_t ShapeAttr::getRank() const {
  assert(hasRank());
  return getImpl()->shape.size();
}

bool ShapeAttr::hasStaticShape() const {
  if (!hasRank()) return false;

  for (auto dim : getShape()) {
    if (dim < 0) return false;
  }

  return true;
}

namespace {
// Returns the shape of the given value if it's ranked; returns std::nullopt
// otherwise.
Optional<ArrayRef<int64_t>> GetShape(Value value) {
  auto shaped_type = value.getType().cast<ShapedType>();
  if (shaped_type.hasRank()) return shaped_type.getShape();
  return std::nullopt;
}

// Merges cast compatible shapes and returns a more refined shape. The two
// shapes are cast compatible if they have the same rank and at each dimension,
// either both have same size or one of them is dynamic. Returns false if the
// given shapes are not cast compatible. The refined shape is same or more
// precise than the two input shapes.
bool GetCastCompatibleShape(ArrayRef<int64_t> a_shape,
                            ArrayRef<int64_t> b_shape,
                            SmallVectorImpl<int64_t>* refined_shape) {
  if (a_shape.size() != b_shape.size()) return false;
  int64_t rank = a_shape.size();
  refined_shape->reserve(rank);
  for (auto dims : llvm::zip(a_shape, b_shape)) {
    int64_t dim1 = std::get<0>(dims);
    int64_t dim2 = std::get<1>(dims);

    if (ShapedType::isDynamic(dim1)) {
      refined_shape->push_back(dim2);
      continue;
    }
    if (ShapedType::isDynamic(dim2)) {
      refined_shape->push_back(dim1);
      continue;
    }
    if (dim1 == dim2) {
      refined_shape->push_back(dim1);
      continue;
    }
    return false;
  }
  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Utility iterators
//===----------------------------------------------------------------------===//

OperandShapeIterator::OperandShapeIterator(Operation::operand_iterator it)
    : llvm::mapped_iterator<Operation::operand_iterator,
                            llvm::Optional<ArrayRef<int64_t>> (*)(Value)>(
          it, &GetShape) {}

ResultShapeIterator::ResultShapeIterator(Operation::result_iterator it)
    : llvm::mapped_iterator<Operation::result_iterator,
                            llvm::Optional<ArrayRef<int64_t>> (*)(Value)>(
          it, &GetShape) {}

//===----------------------------------------------------------------------===//
// TF types helper functions
//===----------------------------------------------------------------------===//

bool TensorFlowType::classof(Type type) {
  return llvm::isa<TFTypeDialect>(type.getDialect());
}
bool TensorFlowRefType::classof(Type type) {
  return type.isa<
#define HANDLE_TF_TYPE(tftype, enumerant, name)
#define HANDLE_TF_REF_TYPE(tftype, enumerant, name) tftype##Type,
#define HANDLE_LAST_TF_TYPE(tftype, enumerant, name) tftype##Type
// NOLINTNEXTLINE
#include "tensorflow/core/ir/types/types.def"
      >();
}

TensorFlowType TensorFlowRefType::get(Type type) {
  MLIRContext* ctx = type.getContext();
  type = getElementTypeOrSelf(type);
  if (type.isF16()) {
    return HalfRefType::get(ctx);
  } else if (type.isF32()) {
    return FloatRefType::get(ctx);
  } else if (type.isF64()) {
    return DoubleRefType::get(ctx);
  } else if (type.isBF16()) {
    return Bfloat16RefType::get(ctx);
  } else if (type.isFloat8E4M3FN()) {
    return Float8E4M3FNRefType::get(ctx);
  } else if (type.isFloat8E5M2()) {
    return Float8E5M2RefType::get(ctx);
  } else if (auto complex_type = type.dyn_cast<ComplexType>()) {
    Type etype = complex_type.getElementType();
    if (etype.isF32()) {
      return Complex64RefType::get(ctx);
    } else if (etype.isF64()) {
      return Complex128RefType::get(ctx);
    }
    llvm_unreachable("unexpected complex type");
  } else if (auto itype = type.dyn_cast<IntegerType>()) {
    switch (itype.getWidth()) {
      case 1:
        return BoolRefType::get(ctx);
      case 8:
        return itype.isUnsigned() ? TensorFlowType(Uint8RefType::get(ctx))
                                  : Int8RefType::get(ctx);
      case 16:
        return itype.isUnsigned() ? TensorFlowType(Uint16RefType::get(ctx))
                                  : Int16RefType::get(ctx);
      case 32:
        return itype.isUnsigned() ? TensorFlowType(Uint32RefType::get(ctx))
                                  : Int32RefType::get(ctx);
      case 64:
        return itype.isUnsigned() ? TensorFlowType(Uint64RefType::get(ctx))
                                  : Int64RefType::get(ctx);
      default:
        llvm_unreachable("unexpected integer type");
    }
  }
#define HANDLE_TF_TYPE(tftype, enumerant, name)        \
  if (auto derived_ty = type.dyn_cast<tftype##Type>()) \
    return tftype##RefType::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/core/ir/types/types.def"
  llvm_unreachable("unexpected type kind");
}

Type TensorFlowRefType::RemoveRef() {
  MLIRContext* ctx = getContext();
  if (isa<HalfRefType>()) return FloatType::getF16(ctx);
  if (isa<FloatRefType>()) return FloatType::getF32(ctx);
  if (isa<DoubleRefType>()) return FloatType::getF64(ctx);
  if (isa<Bfloat16RefType>()) return FloatType::getBF16(ctx);
  if (isa<Float8E4M3FNType>()) return FloatType::getFloat8E4M3FN(ctx);
  if (isa<Float8E5M2Type>()) return FloatType::getFloat8E5M2(ctx);
  if (isa<BoolRefType>()) return IntegerType::get(ctx, 1);
  if (isa<Int8RefType>()) return IntegerType::get(ctx, 8);
  if (isa<Int16RefType>()) return IntegerType::get(ctx, 16);
  if (isa<Int32RefType>()) return IntegerType::get(ctx, 32);
  if (isa<Int64RefType>()) return IntegerType::get(ctx, 64);
  if (isa<Uint8RefType>())
    return IntegerType::get(ctx, 8, IntegerType::Unsigned);
  if (isa<Uint16RefType>())
    return IntegerType::get(ctx, 16, IntegerType::Unsigned);
  if (isa<Uint32RefType>())
    return IntegerType::get(ctx, 32, IntegerType::Unsigned);
  if (isa<Uint64RefType>())
    return IntegerType::get(ctx, 64, IntegerType::Unsigned);
  if (isa<Complex64RefType>()) return ComplexType::get(FloatType::getF32(ctx));
  if (isa<Complex128RefType>()) return ComplexType::get(FloatType::getF64(ctx));
#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  if (isa<tftype##RefType>()) return tftype##Type::get(ctx);

#define HANDLE_TF_REF_TYPE(tftype, enumerant, name)
// NOLINTNEXTLINE
#include "tensorflow/core/ir/types/types.def"
  llvm_unreachable("unexpected tensorflow ref type kind");
}

bool TensorFlowTypeWithSubtype::classof(Type type) {
  return type.isa<ResourceType, VariantType>();
}

Type TensorFlowTypeWithSubtype::RemoveSubtypes() {
  MLIRContext* ctx = getContext();
  if (isa<VariantType>()) return VariantType::get(ctx);
  if (isa<ResourceType>()) return ResourceType::get(ctx);
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

TensorFlowTypeWithSubtype TensorFlowTypeWithSubtype::clone(
    ArrayRef<TensorType> new_subtypes) {
  MLIRContext* ctx = getContext();
  if (isa<VariantType>())
    return VariantType::get(new_subtypes, ctx)
        .cast<TensorFlowTypeWithSubtype>();
  if (isa<ResourceType>())
    return ResourceType::get(new_subtypes, ctx)
        .cast<TensorFlowTypeWithSubtype>();
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

ArrayRef<TensorType> TensorFlowTypeWithSubtype::GetSubtypes() {
  if (auto variant_type = dyn_cast<VariantType>())
    return variant_type.getSubtypes();
  if (auto resource_type = dyn_cast<ResourceType>())
    return resource_type.getSubtypes();
  llvm_unreachable("unexpected tensorflow type with subtypes kind");
}

// TODO(jpienaar): BroadcastCompatible and HasCompatibleElementTypes have
// similar structure that could be extracted into helper method.
bool BroadcastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto types : llvm::zip(lhs, rhs)) {
    // Drop ref types because they don't affect broadcast compatibility. E.g.,
    // `tensor<!tf_type.f32ref>` and `tensor<f32>` should be considered
    // broadcast compatible.
    auto lhs_type = DropRefType(std::get<0>(types));
    auto rhs_type = DropRefType(std::get<1>(types));

    // This should be true for all TF ops:
    auto lhs_tt = lhs_type.dyn_cast<TensorType>();
    auto rhs_tt = rhs_type.dyn_cast<TensorType>();
    if (!lhs_tt || !rhs_tt) {
      if (lhs_type != rhs_type) return false;
      continue;
    }

    // Verify matching element types. These should be identical, except for
    // variant type where unknown subtype is considered compatible with all
    // subtypes.
    auto lhs_et = lhs_tt.getElementType();
    auto rhs_et = rhs_tt.getElementType();
    if (lhs_et != rhs_et) {
      // If either does not have subtypes, then the element types don't match.
      auto lhs_wst = lhs_et.dyn_cast<TensorFlowTypeWithSubtype>();
      auto rhs_wst = rhs_et.dyn_cast<TensorFlowTypeWithSubtype>();
      if (!lhs_wst || !rhs_wst) return false;

      // Consider the subtype of variant types.
      auto lhs_wst_st = lhs_wst.GetSubtypes();
      auto rhs_wst_st = rhs_wst.GetSubtypes();
      if (!lhs_wst_st.empty() && !rhs_wst_st.empty()) {
        for (auto subtypes : llvm::zip(lhs_wst_st, rhs_wst_st)) {
          if (!BroadcastCompatible(std::get<0>(subtypes),
                                   std::get<1>(subtypes)))
            return false;
        }
      }
    }

    auto lhs_rt = lhs_type.dyn_cast<RankedTensorType>();
    auto rhs_rt = rhs_type.dyn_cast<RankedTensorType>();
    if (!lhs_rt || !rhs_rt) return true;
    SmallVector<int64_t, 4> shape;
    return OpTrait::util::getBroadcastedShape(lhs_rt.getShape(),
                                              rhs_rt.getShape(), shape);
  }
  return true;
}

// Given two types `a` and `b`, returns a refined type which is cast compatible
// with both `a` and `b` and is equal to or more precise than both of them. It
// returns empty Type if the input types are not cast compatible.
//
// The two types are considered cast compatible if they have dynamically equal
// shapes and element type. For element types that do not have subtypes, they
// must be equal. However for TensorFlow types such as Resource and Variant,
// that also have subtypes, we recursively check for subtype compatibility for
// Resource types and assume all variant types are cast compatible. If either
// one of `a` or `b` have empty subtypes, they are considered cast compatible.
//
// The returned type is same or more precise than the input types. For example,
// if `a` and `b` are cast compatible types tensor<2x?x?xf32> and
// tensor<?x4x?xf32> respectively, the returned type is tensor<2x4x?xf32>.
//
// Provides option to ignore ref types on 'a'. This is useful for TF ops that
// might allow operands to either be same as result type or be a ref type
// corresponding to it.
Type GetCastCompatibleType(Type a, Type b, bool may_ignore_ref_type_a) {
  // Fast path if everything is equal.
  if (a == b) return b;

  auto a_tt = a.dyn_cast<TensorType>();
  auto b_tt = b.dyn_cast<TensorType>();

  // If only one of a or b is a tensor type, they are incompatible.
  if (static_cast<bool>(a_tt) ^ static_cast<bool>(b_tt)) return nullptr;

  // For non-tensor types, we do not need to worry about shape and can return
  // early.
  if (!a_tt && !b_tt) {
    // Remove ref types.
    if (may_ignore_ref_type_a) {
      if (auto ref_type = a.dyn_cast<TensorFlowRefType>()) {
        a = ref_type.RemoveRef();
        if (a == b) return a;
      }
    }
    if (a.getTypeID() != b.getTypeID()) return nullptr;

    // If either is not a type that contain subtypes then the types are not cast
    // compatible.
    auto a_wst = a.dyn_cast<TensorFlowTypeWithSubtype>();
    auto b_wst = b.dyn_cast<TensorFlowTypeWithSubtype>();
    if (!a_wst || !b_wst) return nullptr;

    // For Variant types we are more permissive right now and accept all pairs
    // of Variant types. If we are more constrainted and check compatibility of
    // subtypes, we might reject valid graphs.
    // TODO(prakalps): Variant doesn't have a subtype, we assign it
    // one, so we should only assign it one when we know the subtype. Then we
    // can be more constrained and check subtypes for cast compatibility as
    // well.
    if (a.isa<VariantType>()) return a;
    if (b.isa<VariantType>()) return b;

    // For Resource types, we recursively check the subtypes for cast
    // compatibility, if possible. Otherwise treat them as compatible.
    auto a_wst_st = a_wst.GetSubtypes();
    auto b_wst_st = b_wst.GetSubtypes();
    if (a_wst_st.empty()) return b;
    if (b_wst_st.empty()) return a;
    if (a_wst_st.size() != b_wst_st.size()) return nullptr;
    SmallVector<TensorType, 4> refined_subtypes;
    for (auto subtypes : llvm::zip(a_wst_st, b_wst_st)) {
      Type refined_st =
          GetCastCompatibleType(std::get<0>(subtypes), std::get<1>(subtypes),
                                /*may_ignore_ref_type_a=*/false);
      if (!refined_st) return nullptr;
      refined_subtypes.push_back(refined_st.cast<TensorType>());
    }

    return ResourceType::get(refined_subtypes, a.getContext());
  }

  // For tensor types, check compatibility of both element type and shape.
  Type refined_element_ty = GetCastCompatibleType(
      a_tt.getElementType(), b_tt.getElementType(), may_ignore_ref_type_a);
  if (!refined_element_ty) return nullptr;

  if (!a_tt.hasRank() && !b_tt.hasRank()) {
    return UnrankedTensorType::get(refined_element_ty);
  }
  if (!a_tt.hasRank()) {
    return RankedTensorType::get(b_tt.getShape(), refined_element_ty);
  }
  if (!b_tt.hasRank()) {
    return RankedTensorType::get(a_tt.getShape(), refined_element_ty);
  }

  SmallVector<int64_t, 4> refined_shape;
  if (!GetCastCompatibleShape(a_tt.getShape(), b_tt.getShape(), &refined_shape))
    return nullptr;

  return RankedTensorType::get(refined_shape, refined_element_ty);
}

bool HasCompatibleElementTypes(Type lhs, Type rhs,
                               bool may_ignore_ref_type_lhs) {
  return GetCastCompatibleType(lhs, rhs, may_ignore_ref_type_lhs) != nullptr;
}

bool AreCastCompatible(TypeRange types) {
  Type common = types.front();
  for (auto type : types.drop_front()) {
    Type refined_type =
        GetCastCompatibleType(common, type, /*may_ignore_ref_type_a=*/false);
    if (!refined_type) return false;
    common = refined_type;
  }
  return true;
}

bool ArraysAreCastCompatible(TypeRange lhs, TypeRange rhs) {
  if (lhs.size() != rhs.size()) return false;
  for (auto pair : llvm::zip(lhs, rhs)) {
    auto lhs_i = std::get<0>(pair);
    auto rhs_i = std::get<1>(pair);
    if (!AreCastCompatible({lhs_i, rhs_i})) return false;
  }
  return true;
}

// Returns the corresponding TensorFlow or standard type from TensorFlowRef
// type.
static Type GetDefaultTypeOf(TensorFlowRefType type) {
  return type.RemoveRef();
}

// Assumes a function `GetDefaultTypeOf(ComposedType)` that returns the default
// type for a composed type (such as a ref type or a type with subtypes).
template <typename ComposedType>
Type DropTypeHelper(Type ty) {
  Type element_ty = getElementTypeOrSelf(ty);
  auto composed_type = element_ty.dyn_cast<ComposedType>();
  if (!composed_type) return ty;

  Type default_ty = GetDefaultTypeOf(composed_type);
  if (auto ranked_ty = ty.dyn_cast<RankedTensorType>()) {
    return RankedTensorType::get(ranked_ty.getShape(), default_ty);
  } else if (ty.dyn_cast<UnrankedTensorType>()) {
    return UnrankedTensorType::get(default_ty);
  } else {
    return default_ty;
  }
}

Type DropSubTypes(Type ty) {
  return DropTypeHelper<TensorFlowTypeWithSubtype>(ty);
}

Type DropRefType(Type ty) { return DropTypeHelper<TensorFlowRefType>(ty); }

Type DropRefAndSubTypes(Type ty) { return DropRefType(DropSubTypes(ty)); }

Attribute TensorProtoAttr::parse(AsmParser& parser, Type type) {
  if (parser.parseColon()) {
    return nullptr;
  }

  std::string data;
  if (parser.parseString(&data)) {
    return nullptr;
  }
  if (data.size() < 2 || data.substr(0, 2) != "0x") {
    parser.emitError(parser.getNameLoc(), "Hex string doesn't start with `0x`");
    return nullptr;
  }

  std::string bytes_data = absl::HexStringToBytes(data.substr(2));
  return TensorProtoAttr::get(type, bytes_data);
}

void TensorProtoAttr::print(mlir::AsmPrinter& printer) const {
  StringRef bytes_str = getValue();
  printer << " : \"0x" << llvm::toHex(bytes_str) << "\"";
}

}  // namespace tf_type
}  // namespace mlir
