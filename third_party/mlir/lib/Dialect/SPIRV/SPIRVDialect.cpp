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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/Support/StringExtras.h"
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
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect, Location loc,
                                      StringRef spec);
template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect, Location loc,
                                    StringRef spec);

template <>
Optional<uint64_t> parseAndVerify<uint64_t>(SPIRVDialect const &dialect,
                                            Location loc, StringRef spec);

// Parses "<number> x" from the beginning of `spec`.
static bool parseNumberX(StringRef &spec, int64_t &number) {
  spec = spec.ltrim();
  if (spec.empty() || !llvm::isDigit(spec.front()))
    return false;

  number = 0;
  do {
    number = number * 10 + spec.front() - '0';
    spec = spec.drop_front();
  } while (!spec.empty() && llvm::isDigit(spec.front()));

  spec = spec.ltrim();
  if (!spec.consume_front("x"))
    return false;

  return true;
}

static bool isValidSPIRVIntType(IntegerType type) {
  return llvm::is_contained(llvm::ArrayRef<unsigned>({1, 8, 16, 32, 64}),
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

static Type parseAndVerifyType(SPIRVDialect const &dialect, StringRef spec,
                               Location loc) {
  spec = spec.trim();
  auto *context = dialect.getContext();
  size_t numCharsRead = 0;
  auto type = mlir::parseType(spec.trim(), context, numCharsRead);
  if (!type) {
    emitError(loc, "cannot parse type: ") << spec;
    return Type();
  }
  if (numCharsRead < spec.size()) {
    emitError(loc, "unexpected additional tokens '")
        << spec.substr(numCharsRead) << "' after parsing type: " << type;
    return Type();
  }

  // Allow SPIR-V dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      emitError(loc, "cannot use 'bf16' to compose SPIR-V types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!isValidSPIRVIntType(t)) {
      emitError(loc, "only 1/8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      emitError(loc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
    if (t.getNumElements() > 4) {
      emitError(loc,
                "vector length has to be less than or equal to 4 but found ")
          << t.getNumElements();
      return Type();
    }
  } else {
    emitError(loc, "cannot use ") << type << " to compose SPIR-V types";
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
static Type parseArrayType(SPIRVDialect const &dialect, StringRef spec,
                           Location loc) {
  if (!spec.consume_front("array<") || !spec.consume_back(">")) {
    emitError(loc, "spv.array delimiter <...> mismatch");
    return Type();
  }

  int64_t count = 0;
  spec = spec.trim();
  if (!parseNumberX(spec, count)) {
    emitError(loc, "expected array element count followed by 'x' but found '")
        << spec << "'";
    return Type();
  }

  // According to the SPIR-V spec:
  // "Length is the number of elements in the array. It must be at least 1."
  if (!count) {
    emitError(loc, "expected array length greater than 0");
    return Type();
  }

  if (spec.trim().empty()) {
    emitError(loc, "expected element type");
    return Type();
  }

  ArrayType::LayoutInfo layoutInfo = 0;
  size_t lastLSquare;

  // Handle case when element type is not a trivial type
  auto lastRDelimiter = spec.rfind('>');
  if (lastRDelimiter != StringRef::npos) {
    lastLSquare = spec.find('[', lastRDelimiter);
  } else {
    lastLSquare = spec.rfind('[');
  }

  if (lastLSquare != StringRef::npos) {
    auto layoutSpec = spec.substr(lastLSquare);
    layoutSpec = layoutSpec.trim();
    if (!layoutSpec.consume_front("[") || !layoutSpec.consume_back("]")) {
      emitError(loc, "expected array stride within '[' ']' in '")
          << layoutSpec << "'";
      return Type();
    }
    layoutSpec = layoutSpec.trim();
    auto layout =
        parseAndVerify<ArrayType::LayoutInfo>(dialect, loc, layoutSpec);
    if (!layout) {
      return Type();
    }

    if (!(layoutInfo = layout.getValue())) {
      emitError(loc, "ArrayStride must be greater than zero");
      return Type();
    }

    spec = spec.substr(0, lastLSquare);
  }

  Type elementType = parseAndVerifyType(dialect, spec, loc);
  if (!elementType)
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
static Type parsePointerType(SPIRVDialect const &dialect, StringRef spec,
                             Location loc) {
  if (!spec.consume_front("ptr<") || !spec.consume_back(">")) {
    emitError(loc, "spv.ptr delimiter <...> mismatch");
    return Type();
  }

  // Split into pointee type and storage class
  StringRef scSpec, ptSpec;
  std::tie(ptSpec, scSpec) = spec.rsplit(',');
  if (scSpec.empty()) {
    emitError(loc,
              "expected comma to separate pointee type and storage class in '")
        << spec << "'";
    return Type();
  }

  scSpec = scSpec.trim();
  auto storageClass = symbolizeStorageClass(scSpec);
  if (!storageClass) {
    emitError(loc, "unknown storage class: ") << scSpec;
    return Type();
  }

  if (ptSpec.trim().empty()) {
    emitError(loc, "expected pointee type");
    return Type();
  }

  auto pointeeType = parseAndVerifyType(dialect, ptSpec, loc);
  if (!pointeeType)
    return Type();

  return PointerType::get(pointeeType, *storageClass);
}

// runtime-array-type ::= `!spv.rtarray<` element-type `>`
static Type parseRuntimeArrayType(SPIRVDialect const &dialect, StringRef spec,
                                  Location loc) {
  if (!spec.consume_front("rtarray<") || !spec.consume_back(">")) {
    emitError(loc, "spv.rtarray delimiter <...> mismatch");
    return Type();
  }

  if (spec.trim().empty()) {
    emitError(loc, "expected element type");
    return Type();
  }

  Type elementType = parseAndVerifyType(dialect, spec, loc);
  if (!elementType)
    return Type();

  return RuntimeArrayType::get(elementType);
}

// Specialize this function to parse each of the parameters that define an
// ImageType. By default it assumes this is an enum type.
template <typename ValTy>
static Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect, Location loc,
                                      StringRef spec) {
  auto val = spirv::symbolizeEnum<ValTy>()(spec);
  if (!val) {
    emitError(loc, "unknown attribute: '") << spec << "'";
  }
  return val;
}

template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect, Location loc,
                                    StringRef spec) {
  // TODO(ravishankarm): Further verify that the element type can be sampled
  auto ty = parseAndVerifyType(dialect, spec, loc);
  if (!ty) {
    return llvm::None;
  }
  return ty;
}

template <typename IntTy>
static Optional<IntTy> parseAndVerifyInteger(SPIRVDialect const &dialect,
                                             Location loc, StringRef spec) {
  IntTy offsetVal = std::numeric_limits<IntTy>::max();
  spec = spec.trim();
  if (spec.consumeInteger(10, offsetVal)) {
    return llvm::None;
  }
  spec = spec.trim();
  if (!spec.empty()) {
    return llvm::None;
  }
  return offsetVal;
}

template <>
Optional<uint64_t> parseAndVerify<uint64_t>(SPIRVDialect const &dialect,
                                            Location loc, StringRef spec) {
  return parseAndVerifyInteger<uint64_t>(dialect, loc, spec);
}

// Functor object to parse a comma separated list of specs. The function
// parseAndVerify does the actual parsing and verification of individual
// elements. This is a functor since parsing the last element of the list
// (termination condition) needs partial specialization.
template <typename ParseType, typename... Args> struct parseCommaSeparatedList {
  Optional<std::tuple<ParseType, Args...>>
  operator()(SPIRVDialect const &dialect, Location loc, StringRef spec) const {
    auto numArgs = std::tuple_size<std::tuple<Args...>>::value;
    StringRef parseSpec, restSpec;
    std::tie(parseSpec, restSpec) = spec.split(',');

    parseSpec = parseSpec.trim();
    if (numArgs != 0 && restSpec.empty()) {
      emitError(loc, "expected more parameters for image type '")
          << parseSpec << "'";
      return llvm::None;
    }

    auto parseVal = parseAndVerify<ParseType>(dialect, loc, parseSpec);
    if (!parseVal) {
      return llvm::None;
    }

    auto remainingValues =
        parseCommaSeparatedList<Args...>{}(dialect, loc, restSpec);
    if (!remainingValues) {
      return llvm::None;
    }
    return std::tuple_cat(std::tuple<ParseType>(parseVal.getValue()),
                          remainingValues.getValue());
  }
};

// Partial specialization of the function to parse a comma separated list of
// specs to parse the last element of the list.
template <typename ParseType> struct parseCommaSeparatedList<ParseType> {
  Optional<std::tuple<ParseType>>
  operator()(SPIRVDialect const &dialect, Location loc, StringRef spec) const {
    spec = spec.trim();
    auto value = parseAndVerify<ParseType>(dialect, loc, spec);
    if (!value) {
      return llvm::None;
    }
    return std::tuple<ParseType>(value.getValue());
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
static Type parseImageType(SPIRVDialect const &dialect, StringRef spec,
                           Location loc) {
  if (!spec.consume_front("image<") || !spec.consume_back(">")) {
    emitError(loc, "spv.image delimiter <...> mismatch");
    return Type();
  }

  auto value =
      parseCommaSeparatedList<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                              ImageSamplingInfo, ImageSamplerUseInfo,
                              ImageFormat>{}(dialect, loc, spec);
  if (!value) {
    return Type();
  }

  return ImageType::get(value.getValue());
}

// Parse decorations associated with a member.
static ParseResult parseStructMemberDecorations(
    SPIRVDialect const &dialect, Location loc, StringRef spec,
    ArrayRef<Type> memberTypes,
    SmallVectorImpl<StructType::LayoutInfo> &layoutInfo,
    SmallVectorImpl<StructType::MemberDecorationInfo> &memberDecorationInfo) {
  spec = spec.trim();
  auto memberInfo = spec.split(',');
  // Check if the first element is offset.
  auto layout =
      parseAndVerify<StructType::LayoutInfo>(dialect, loc, memberInfo.first);
  if (layout) {
    if (layoutInfo.size() != memberTypes.size() - 1) {
      return emitError(loc,
                       "layout specification must be given for all members");
    }
    layoutInfo.push_back(layout.getValue());
    spec = memberInfo.second.trim();
  }

  // Check for spirv::Decorations.
  while (!spec.empty()) {
    memberInfo = spec.split(',');
    auto memberDecoration =
        parseAndVerify<spirv::Decoration>(dialect, loc, memberInfo.first);
    if (!memberDecoration) {
      return failure();
    }
    memberDecorationInfo.emplace_back(
        static_cast<uint32_t>(memberTypes.size() - 1),
        memberDecoration.getValue());
    spec = memberInfo.second.trim();
  }
  return success();
}

// struct-member-decoration ::= integer-literal? spirv-decoration*
// struct-type ::= `!spv.struct<` spirv-type (`[` struct-member-decoration `]`)?
//                     (`, ` spirv-type (`[` struct-member-decoration `]`)?
static Type parseStructType(SPIRVDialect const &dialect, StringRef spec,
                            Location loc) {
  if (!spec.consume_front("struct<") || !spec.consume_back(">")) {
    emitError(loc, "spv.struct delimiter <...> mismatch");
    return Type();
  }

  SmallVector<Type, 4> memberTypes;
  SmallVector<StructType::LayoutInfo, 4> layoutInfo;
  SmallVector<StructType::MemberDecorationInfo, 4> memberDecorationInfo;

  auto *context = dialect.getContext();
  while (!spec.empty()) {
    spec = spec.trim();
    size_t pos = 0;
    auto memberType = mlir::parseType(spec, context, pos);
    if (!memberType) {
      emitError(loc, "cannot parse type from '") << spec << "'";
    }
    memberTypes.push_back(memberType);

    spec = spec.substr(pos).trim();
    if (spec.consume_front("[")) {
      auto rSquare = spec.find(']');
      if (rSquare == StringRef::npos) {
        emitError(loc, "missing matching ']' in ") << spec;
        return Type();
      }
      if (parseStructMemberDecorations(dialect, loc, spec.substr(0, rSquare),
                                       memberTypes, layoutInfo,
                                       memberDecorationInfo)) {
        return Type();
      }
      spec = spec.substr(rSquare + 1).trim();
    }

    // Handle comma.
    if (!spec.consume_front(",")) {
      // End of decorations list.
      break;
    }
  }
  spec = spec.trim();
  if (!spec.empty()) {
    emitError(loc, "unexpected substring '")
        << spec << "' while parsing StructType";
    return Type();
  }
  if (!layoutInfo.empty() && memberTypes.size() != layoutInfo.size()) {
    emitError(loc, "layout specification must be given for all members");
    return Type();
  }
  if (memberTypes.empty()) {
    return StructType::getEmpty(dialect.getContext());
  }
  return StructType::get(memberTypes, layoutInfo, memberDecorationInfo);
}

// spirv-type ::= array-type
//              | element-type
//              | image-type
//              | pointer-type
//              | runtime-array-type
//              | struct-type
Type SPIRVDialect::parseType(StringRef spec, Location loc) const {
  if (spec.startswith("array"))
    return parseArrayType(*this, spec, loc);
  if (spec.startswith("image"))
    return parseImageType(*this, spec, loc);
  if (spec.startswith("ptr"))
    return parsePointerType(*this, spec, loc);
  if (spec.startswith("rtarray"))
    return parseRuntimeArrayType(*this, spec, loc);
  if (spec.startswith("struct"))
    return parseStructType(*this, spec, loc);

  emitError(loc, "unknown SPIR-V type: ") << spec;
  return Type();
}

//===----------------------------------------------------------------------===//
// Type Printing
//===----------------------------------------------------------------------===//

static void print(ArrayType type, llvm::raw_ostream &os) {
  os << "array<" << type.getNumElements() << " x " << type.getElementType();
  if (type.hasLayout()) {
    os << " [" << type.getArrayStride() << "]";
  }
  os << ">";
}

static void print(RuntimeArrayType type, llvm::raw_ostream &os) {
  os << "rtarray<" << type.getElementType() << ">";
}

static void print(PointerType type, llvm::raw_ostream &os) {
  os << "ptr<" << type.getPointeeType() << ", "
     << stringifyStorageClass(type.getStorageClass()) << ">";
}

static void print(ImageType type, llvm::raw_ostream &os) {
  os << "image<" << type.getElementType() << ", " << stringifyDim(type.getDim())
     << ", " << stringifyImageDepthInfo(type.getDepthInfo()) << ", "
     << stringifyImageArrayedInfo(type.getArrayedInfo()) << ", "
     << stringifyImageSamplingInfo(type.getSamplingInfo()) << ", "
     << stringifyImageSamplerUseInfo(type.getSamplerUseInfo()) << ", "
     << stringifyImageFormat(type.getImageFormat()) << ">";
}

static void print(StructType type, llvm::raw_ostream &os) {
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

void SPIRVDialect::printType(Type type, llvm::raw_ostream &os) const {
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
