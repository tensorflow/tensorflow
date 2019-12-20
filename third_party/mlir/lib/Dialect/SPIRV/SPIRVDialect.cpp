//===- LLVMDialect.cpp - MLIR SPIR-V dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SPIR-V dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/SPIRVTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/StringExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace spirv {
#include "mlir/Dialect/SPIRV/SPIRVOpUtils.inc"
} // namespace spirv
} // namespace mlir

using namespace mlir;
using namespace mlir::spirv;

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

/// Returns true if the given region contains spv.Return or spv.ReturnValue ops.
static inline bool containsReturn(Region &region) {
  return llvm::any_of(region, [](Block &block) {
    Operation *terminator = block.getTerminator();
    return isa<spirv::ReturnOp>(terminator) ||
           isa<spirv::ReturnValueOp>(terminator);
  });
}

namespace {
/// This class defines the interface for inlining within the SPIR-V dialect.
struct SPIRVInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// Returns true if the given region 'src' can be inlined into the region
  /// 'dest' that is attached to an operation registered to the current dialect.
  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &) const final {
    // Return true here when inlining into spv.selection and spv.loop
    // operations.
    auto op = dest->getParentOp();
    return isa<spirv::SelectionOp>(op) || isa<spirv::LoopOp>(op);
  }

  /// Returns true if the given operation 'op', that is registered to this
  /// dialect, can be inlined into the region 'dest' that is attached to an
  /// operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest,
                       BlockAndValueMapping &) const final {
    // TODO(antiagainst): Enable inlining structured control flows with return.
    if ((isa<spirv::SelectionOp>(op) || isa<spirv::LoopOp>(op)) &&
        containsReturn(op->getRegion(0)))
      return false;
    // TODO(antiagainst): we need to filter OpKill here to avoid inlining it to
    // a loop continue construct:
    // https://github.com/KhronosGroup/SPIRV-Headers/issues/86
    // However OpKill is fragment shader specific and we don't support it yet.
    return true;
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op, Block *newDest) const final {
    if (auto returnOp = dyn_cast<spirv::ReturnOp>(op)) {
      OpBuilder(op).create<spirv::BranchOp>(op->getLoc(), newDest);
      op->erase();
    } else if (auto retValOp = dyn_cast<spirv::ReturnValueOp>(op)) {
      llvm_unreachable("unimplemented spv.ReturnValue in inliner");
    }
  }

  /// Handle the given inlined terminator by replacing it with a new operation
  /// as necessary.
  void handleTerminator(Operation *op,
                        ArrayRef<Value *> valuesToRepl) const final {
    // Only spv.ReturnValue needs to be handled here.
    auto retValOp = dyn_cast<spirv::ReturnValueOp>(op);
    if (!retValOp)
      return;

    // Replace the values directly with the return operands.
    assert(valuesToRepl.size() == 1 &&
           "spv.ReturnValue expected to only handle one result");
    valuesToRepl.front()->replaceAllUsesWith(retValOp.value());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

SPIRVDialect::SPIRVDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<ArrayType, ImageType, PointerType, RuntimeArrayType, StructType>();

  // Add SPIR-V ops.
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/SPIRV/SPIRVOps.cpp.inc"
      >();

  addInterfaces<SPIRVInlinerInterface>();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
}

std::string SPIRVDialect::getAttributeName(Decoration decoration) {
  return convertToSnakeCase(stringifyDecoration(decoration));
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

// Forward declarations.
template <typename ValTy>
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                      DialectAsmParser &parser);
template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                    DialectAsmParser &parser);

template <>
Optional<uint64_t> parseAndVerify<uint64_t>(SPIRVDialect const &dialect,
                                            DialectAsmParser &parser);

static bool isValidSPIRVIntType(IntegerType type) {
  return llvm::is_contained(ArrayRef<unsigned>({1, 8, 16, 32, 64}),
                            type.getWidth());
}

bool SPIRVDialect::isValidScalarType(Type type) {
  if (type.isa<FloatType>()) {
    return !type.isBF16();
  }
  if (auto intType = type.dyn_cast<IntegerType>()) {
    return isValidSPIRVIntType(intType);
  }
  return false;
}

static bool isValidSPIRVVectorType(VectorType type) {
  return type.getRank() == 1 &&
         SPIRVDialect::isValidScalarType(type.getElementType()) &&
         type.getNumElements() >= 2 && type.getNumElements() <= 4;
}

bool SPIRVDialect::isValidType(Type type) {
  // Allow SPIR-V dialect types
  if (type.getKind() >= Type::FIRST_SPIRV_TYPE &&
      type.getKind() <= TypeKind::LAST_SPIRV_TYPE) {
    return true;
  }
  if (SPIRVDialect::isValidScalarType(type)) {
    return true;
  }
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    return isValidSPIRVVectorType(vectorType);
  }
  return false;
}

static Type parseAndVerifyType(SPIRVDialect const &dialect,
                               DialectAsmParser &parser) {
  Type type;
  llvm::SMLoc typeLoc = parser.getCurrentLocation();
  if (parser.parseType(type))
    return Type();

  // Allow SPIR-V dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      parser.emitError(typeLoc, "cannot use 'bf16' to compose SPIR-V types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!isValidSPIRVIntType(t)) {
      parser.emitError(typeLoc,
                       "only 1/8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      parser.emitError(typeLoc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
    if (t.getNumElements() > 4) {
      parser.emitError(
          typeLoc, "vector length has to be less than or equal to 4 but found ")
          << t.getNumElements();
      return Type();
    }
  } else {
    parser.emitError(typeLoc, "cannot use ")
        << type << " to compose SPIR-V types";
    return Type();
  }

  return type;
}

// element-type ::= integer-type
//                | floating-point-type
//                | vector-type
//                | spirv-type
//
// array-type ::= `!spv.array<` integer-literal `x` element-type
//                (`[` integer-literal `]`)? `>`
static Type parseArrayType(SPIRVDialect const &dialect,
                           DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 1> countDims;
  llvm::SMLoc countLoc = parser.getCurrentLocation();
  if (parser.parseDimensionList(countDims, /*allowDynamic=*/false))
    return Type();
  if (countDims.size() != 1) {
    parser.emitError(countLoc,
                     "expected single integer for array element count");
    return Type();
  }

  // According to the SPIR-V spec:
  // "Length is the number of elements in the array. It must be at least 1."
  int64_t count = countDims[0];
  if (count == 0) {
    parser.emitError(countLoc, "expected array length greater than 0");
    return Type();
  }

  Type elementType = parseAndVerifyType(dialect, parser);
  if (!elementType)
    return Type();

  ArrayType::LayoutInfo layoutInfo = 0;
  if (succeeded(parser.parseOptionalLSquare())) {
    llvm::SMLoc layoutLoc = parser.getCurrentLocation();
    auto layout = parseAndVerify<ArrayType::LayoutInfo>(dialect, parser);
    if (!layout)
      return Type();

    if (!(layoutInfo = layout.getValue())) {
      parser.emitError(layoutLoc, "ArrayStride must be greater than zero");
      return Type();
    }

    if (parser.parseRSquare())
      return Type();
  }

  if (parser.parseGreater())
    return Type();
  return ArrayType::get(elementType, count, layoutInfo);
}

// TODO(ravishankarm) : Reorder methods to be utilities first and parse*Type
// methods in alphabetical order
//
// storage-class ::= `UniformConstant`
//                 | `Uniform`
//                 | `Workgroup`
//                 | <and other storage classes...>
//
// pointer-type ::= `!spv.ptr<` element-type `,` storage-class `>`
static Type parsePointerType(SPIRVDialect const &dialect,
                             DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  auto pointeeType = parseAndVerifyType(dialect, parser);
  if (!pointeeType)
    return Type();

  StringRef storageClassSpec;
  llvm::SMLoc storageClassLoc = parser.getCurrentLocation();
  if (parser.parseComma() || parser.parseKeyword(&storageClassSpec))
    return Type();

  auto storageClass = symbolizeStorageClass(storageClassSpec);
  if (!storageClass) {
    parser.emitError(storageClassLoc, "unknown storage class: ")
        << storageClassSpec;
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return PointerType::get(pointeeType, *storageClass);
}

// runtime-array-type ::= `!spv.rtarray<` element-type `>`
static Type parseRuntimeArrayType(SPIRVDialect const &dialect,
                                  DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type elementType = parseAndVerifyType(dialect, parser);
  if (!elementType)
    return Type();

  if (parser.parseGreater())
    return Type();
  return RuntimeArrayType::get(elementType);
}

// Specialize this function to parse each of the parameters that define an
// ImageType. By default it assumes this is an enum type.
template <typename ValTy>
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect,
                                      DialectAsmParser &parser) {
  StringRef enumSpec;
  llvm::SMLoc enumLoc = parser.getCurrentLocation();
  if (parser.parseKeyword(&enumSpec)) {
    return llvm::None;
  }

  auto val = spirv::symbolizeEnum<ValTy>()(enumSpec);
  if (!val)
    parser.emitError(enumLoc, "unknown attribute: '") << enumSpec << "'";
  return val;
}

template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect,
                                    DialectAsmParser &parser) {
  // TODO(ravishankarm): Further verify that the element type can be sampled
  auto ty = parseAndVerifyType(dialect, parser);
  if (!ty)
    return llvm::None;
  return ty;
}

template <typename IntTy>
static Optional<IntTy> parseAndVerifyInteger(SPIRVDialect const &dialect,
                                             DialectAsmParser &parser) {
  IntTy offsetVal = std::numeric_limits<IntTy>::max();
  if (parser.parseInteger(offsetVal))
    return llvm::None;
  return offsetVal;
}

template <>
Optional<uint64_t> parseAndVerify<uint64_t>(SPIRVDialect const &dialect,
                                            DialectAsmParser &parser) {
  return parseAndVerifyInteger<uint64_t>(dialect, parser);
}

// Functor object to parse a comma separated list of specs. The function
// parseAndVerify does the actual parsing and verification of individual
// elements. This is a functor since parsing the last element of the list
// (termination condition) needs partial specialization.
template <typename ParseType, typename... Args> struct parseCommaSeparatedList {
  Optional<std::tuple<ParseType, Args...>>
  operator()(SPIRVDialect const &dialect, DialectAsmParser &parser) const {
    auto parseVal = parseAndVerify<ParseType>(dialect, parser);
    if (!parseVal)
      return llvm::None;

    auto numArgs = std::tuple_size<std::tuple<Args...>>::value;
    if (numArgs != 0 && failed(parser.parseComma()))
      return llvm::None;
    auto remainingValues = parseCommaSeparatedList<Args...>{}(dialect, parser);
    if (!remainingValues)
      return llvm::None;
    return std::tuple_cat(std::tuple<ParseType>(parseVal.getValue()),
                          remainingValues.getValue());
  }
};

// Partial specialization of the function to parse a comma separated list of
// specs to parse the last element of the list.
template <typename ParseType> struct parseCommaSeparatedList<ParseType> {
  Optional<std::tuple<ParseType>> operator()(SPIRVDialect const &dialect,
                                             DialectAsmParser &parser) const {
    if (auto value = parseAndVerify<ParseType>(dialect, parser))
      return std::tuple<ParseType>(value.getValue());
    return llvm::None;
  }
};

// dim ::= `1D` | `2D` | `3D` | `Cube` | <and other SPIR-V Dim specifiers...>
//
// depth-info ::= `NoDepth` | `IsDepth` | `DepthUnknown`
//
// arrayed-info ::= `NonArrayed` | `Arrayed`
//
// sampling-info ::= `SingleSampled` | `MultiSampled`
//
// sampler-use-info ::= `SamplerUnknown` | `NeedSampler` |  `NoSampler`
//
// format ::= `Unknown` | `Rgba32f` | <and other SPIR-V Image formats...>
//
// image-type ::= `!spv.image<` element-type `,` dim `,` depth-info `,`
//                              arrayed-info `,` sampling-info `,`
//                              sampler-use-info `,` format `>`
static Type parseImageType(SPIRVDialect const &dialect,
                           DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  auto value =
      parseCommaSeparatedList<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                              ImageSamplingInfo, ImageSamplerUseInfo,
                              ImageFormat>{}(dialect, parser);
  if (!value)
    return Type();

  if (parser.parseGreater())
    return Type();
  return ImageType::get(value.getValue());
}

// Parse decorations associated with a member.
static ParseResult parseStructMemberDecorations(
    SPIRVDialect const &dialect, DialectAsmParser &parser,
    ArrayRef<Type> memberTypes,
    SmallVectorImpl<StructType::LayoutInfo> &layoutInfo,
    SmallVectorImpl<StructType::MemberDecorationInfo> &memberDecorationInfo) {

  // Check if the first element is offset.
  llvm::SMLoc layoutLoc = parser.getCurrentLocation();
  StructType::LayoutInfo layout = 0;
  OptionalParseResult layoutParseResult = parser.parseOptionalInteger(layout);
  if (layoutParseResult.hasValue()) {
    if (failed(*layoutParseResult))
      return failure();

    if (layoutInfo.size() != memberTypes.size() - 1) {
      return parser.emitError(
          layoutLoc, "layout specification must be given for all members");
    }
    layoutInfo.push_back(layout);
  }

  // Check for no spirv::Decorations.
  if (succeeded(parser.parseOptionalRSquare()))
    return success();

  // If there was a layout, make sure to parse the comma.
  if (layoutParseResult.hasValue() && parser.parseComma())
    return failure();

  // Check for spirv::Decorations.
  do {
    auto memberDecoration = parseAndVerify<spirv::Decoration>(dialect, parser);
    if (!memberDecoration)
      return failure();

    memberDecorationInfo.emplace_back(
        static_cast<uint32_t>(memberTypes.size() - 1),
        memberDecoration.getValue());
  } while (succeeded(parser.parseOptionalComma()));

  return parser.parseRSquare();
}

// struct-member-decoration ::= integer-literal? spirv-decoration*
// struct-type ::= `!spv.struct<` spirv-type (`[` struct-member-decoration `]`)?
//                     (`, ` spirv-type (`[` struct-member-decoration `]`)? `>`
static Type parseStructType(SPIRVDialect const &dialect,
                            DialectAsmParser &parser) {
  if (parser.parseLess())
    return Type();

  if (succeeded(parser.parseOptionalGreater()))
    return StructType::getEmpty(dialect.getContext());

  SmallVector<Type, 4> memberTypes;
  SmallVector<StructType::LayoutInfo, 4> layoutInfo;
  SmallVector<StructType::MemberDecorationInfo, 4> memberDecorationInfo;

  do {
    Type memberType;
    if (parser.parseType(memberType))
      return Type();
    memberTypes.push_back(memberType);

    if (succeeded(parser.parseOptionalLSquare())) {
      if (parseStructMemberDecorations(dialect, parser, memberTypes, layoutInfo,
                                       memberDecorationInfo)) {
        return Type();
      }
    }
  } while (succeeded(parser.parseOptionalComma()));

  if (!layoutInfo.empty() && memberTypes.size() != layoutInfo.size()) {
    parser.emitError(parser.getNameLoc(),
                     "layout specification must be given for all members");
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return StructType::get(memberTypes, layoutInfo, memberDecorationInfo);
}

// spirv-type ::= array-type
//              | element-type
//              | image-type
//              | pointer-type
//              | runtime-array-type
//              | struct-type
Type SPIRVDialect::parseType(DialectAsmParser &parser) const {
  StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return Type();

  if (keyword == "array")
    return parseArrayType(*this, parser);
  if (keyword == "image")
    return parseImageType(*this, parser);
  if (keyword == "ptr")
    return parsePointerType(*this, parser);
  if (keyword == "rtarray")
    return parseRuntimeArrayType(*this, parser);
  if (keyword == "struct")
    return parseStructType(*this, parser);

  parser.emitError(parser.getNameLoc(), "unknown SPIR-V type: ") << keyword;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

static void print(ArrayType type, DialectAsmPrinter &os) {
  os << "array<" << type.getNumElements() << " x " << type.getElementType();
  if (type.hasLayout()) {
    os << " [" << type.getArrayStride() << "]";
  }
  os << ">";
}

static void print(RuntimeArrayType type, DialectAsmPrinter &os) {
  os << "rtarray<" << type.getElementType() << ">";
}

static void print(PointerType type, DialectAsmPrinter &os) {
  os << "ptr<" << type.getPointeeType() << ", "
     << stringifyStorageClass(type.getStorageClass()) << ">";
}

static void print(ImageType type, DialectAsmPrinter &os) {
  os << "image<" << type.getElementType() << ", " << stringifyDim(type.getDim())
     << ", " << stringifyImageDepthInfo(type.getDepthInfo()) << ", "
     << stringifyImageArrayedInfo(type.getArrayedInfo()) << ", "
     << stringifyImageSamplingInfo(type.getSamplingInfo()) << ", "
     << stringifyImageSamplerUseInfo(type.getSamplerUseInfo()) << ", "
     << stringifyImageFormat(type.getImageFormat()) << ">";
}

static void print(StructType type, DialectAsmPrinter &os) {
  os << "struct<";
  auto printMember = [&](unsigned i) {
    os << type.getElementType(i);
    SmallVector<spirv::Decoration, 0> decorations;
    type.getMemberDecorations(i, decorations);
    if (type.hasLayout() || !decorations.empty()) {
      os << " [";
      if (type.hasLayout()) {
        os << type.getOffset(i);
        if (!decorations.empty())
          os << ", ";
      }
      auto each_fn = [&os](spirv::Decoration decoration) {
        os << stringifyDecoration(decoration);
      };
      interleaveComma(decorations, os, each_fn);
      os << "]";
    }
  };
  interleaveComma(llvm::seq<unsigned>(0, type.getNumElements()), os,
                  printMember);
  os << ">";
}

void SPIRVDialect::printType(Type type, DialectAsmPrinter &os) const {
  switch (type.getKind()) {
  case TypeKind::Array:
    print(type.cast<ArrayType>(), os);
    return;
  case TypeKind::Pointer:
    print(type.cast<PointerType>(), os);
    return;
  case TypeKind::RuntimeArray:
    print(type.cast<RuntimeArrayType>(), os);
    return;
  case TypeKind::Image:
    print(type.cast<ImageType>(), os);
    return;
  case TypeKind::Struct:
    print(type.cast<StructType>(), os);
    return;
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}

//===----------------------------------------------------------------------===//
// Constant
//===----------------------------------------------------------------------===//

Operation *SPIRVDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  if (!ConstantOp::isBuildableWith(type))
    return nullptr;

  return builder.create<spirv::ConstantOp>(loc, type, value);
}
