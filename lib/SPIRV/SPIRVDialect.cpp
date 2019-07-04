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

#include "mlir/SPIRV/SPIRVDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Parser.h"
#include "mlir/SPIRV/SPIRVOps.h"
#include "mlir/SPIRV/SPIRVTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace spirv {
#include "mlir/SPIRV/SPIRVOpUtils.inc"
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

  addOperations<
#define GET_OP_LIST
#include "mlir/SPIRV/SPIRVOps.cpp.inc"
      >();

  // Allow unknown operations because SPIR-V is extensible.
  allowUnknownOperations();
}

//===----------------------------------------------------------------------===//
// Type Parsing
//===----------------------------------------------------------------------===//

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

static Type parseAndVerifyType(SPIRVDialect const &dialect, StringRef spec,
                               Location loc) {
  spec = spec.trim();
  auto *context = dialect.getContext();
  auto type = mlir::parseType(spec.trim(), context);
  if (!type) {
    emitError(loc, "cannot parse type: ") << spec;
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
    if (!llvm::is_contained(llvm::ArrayRef<unsigned>({8, 16, 32, 64}),
                            t.getWidth())) {
      emitError(loc, "only 8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      emitError(loc, "only 1-D vector allowed but found ") << t;
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
// array-type ::= `!spv.array<` integer-literal `x` element-type `>`
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

  if (spec.trim().empty()) {
    emitError(loc, "expected element type");
    return Type();
  }

  Type elementType = parseAndVerifyType(dialect, spec, loc);
  if (!elementType)
    return Type();

  return ArrayType::get(elementType, count);
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

template <>
Optional<spirv::StructType::LayoutInfo>
parseAndVerify(SPIRVDialect const &dialect, Location loc, StringRef spec) {
  uint64_t offsetVal = std::numeric_limits<uint64_t>::max();
  if (!spec.consume_front("[")) {
    emitError(loc, "expected '[' while parsing layout specification in '")
        << spec << "'";
    return llvm::None;
  }
  if (spec.consumeInteger(10, offsetVal)) {
    emitError(
        loc,
        "expected unsigned integer to specify offset of member in struct: '")
        << spec << "'";
    return llvm::None;
  }
  spec = spec.trim();
  if (!spec.consume_front("]")) {
    emitError(loc, "missing ']' in decorations spec: '") << spec << "'";
    return llvm::None;
  }
  if (spec != "") {
    emitError(loc, "unexpected extra tokens in layout information: '")
        << spec << "'";
    return llvm::None;
  }
  return spirv::StructType::LayoutInfo{offsetVal};
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

// Method to parse one member of a struct (including Layout information)
static ParseResult
parseStructElement(SPIRVDialect const &dialect, StringRef spec, Location loc,
                   SmallVectorImpl<Type> &memberTypes,
                   SmallVectorImpl<StructType::LayoutInfo> &layoutInfo) {
  // Check for a '[' <layoutInfo> ']'
  auto lastLSquare = spec.rfind('[');
  auto typeSpec = spec.substr(0, lastLSquare);
  auto layoutSpec = (lastLSquare == StringRef::npos ? StringRef("")
                                                    : spec.substr(lastLSquare));
  auto type = parseAndVerify<Type>(dialect, loc, typeSpec);
  if (!type) {
    return failure();
  }
  memberTypes.push_back(type.getValue());
  if (layoutSpec.empty()) {
    return success();
  }
  if (layoutInfo.size() != memberTypes.size() - 1) {
    emitError(loc, "layout specification must be given for all members");
    return failure();
  }
  auto layout =
      parseAndVerify<StructType::LayoutInfo>(dialect, loc, layoutSpec);
  if (!layout) {
    return failure();
  }
  layoutInfo.push_back(layout.getValue());
  return success();
}

// Helper method to record the position of the corresponding '>' for every '<'
// encountered when parsing the string left to right. The relative position of
// '>' w.r.t to the '<' is recorded.
static bool
computeMatchingRAngles(Location loc, StringRef const &spec,
                       SmallVectorImpl<size_t> &matchingRAngleOffset) {
  SmallVector<size_t, 4> openBrackets;
  for (size_t i = 0, e = spec.size(); i != e; ++i) {
    if (spec[i] == '<') {
      openBrackets.push_back(i);
    } else if (spec[i] == '>') {
      if (openBrackets.empty()) {
        emitError(loc, "unbalanced '<' in '") << spec << "'";
        return false;
      }
      matchingRAngleOffset.push_back(i - openBrackets.pop_back_val());
    }
  }
  return true;
}

static ParseResult
parseStructHelper(SPIRVDialect const &dialect, StringRef spec, Location loc,
                  ArrayRef<size_t> matchingRAngleOffset,
                  SmallVectorImpl<Type> &memberTypes,
                  SmallVectorImpl<StructType::LayoutInfo> &layoutInfo) {
  // Check if the occurrence of ',' or '<' is before. If former, split using
  // ','. If latter, split using matching '>' to get the entire type
  // description
  auto firstComma = spec.find(',');
  auto firstLAngle = spec.find('<');
  if (firstLAngle == StringRef::npos && firstComma == StringRef::npos) {
    return parseStructElement(dialect, spec, loc, memberTypes, layoutInfo);
  }
  if (firstLAngle == StringRef::npos || firstComma < firstLAngle) {
    // Parse the type before the ','
    if (parseStructElement(dialect, spec.substr(0, firstComma), loc,
                           memberTypes, layoutInfo)) {
      return failure();
    }
    return parseStructHelper(dialect, spec.substr(firstComma + 1).ltrim(), loc,
                             matchingRAngleOffset, memberTypes, layoutInfo);
  }
  auto matchingRAngle = matchingRAngleOffset.front() + firstLAngle;
  // Find the next ',' or '>'
  auto endLoc = std::min(spec.find(',', matchingRAngle + 1), spec.size());
  if (parseStructElement(dialect, spec.substr(0, endLoc), loc, memberTypes,
                         layoutInfo)) {
    return failure();
  }
  auto rest = spec.substr(endLoc + 1).ltrim();
  if (rest.empty()) {
    return success();
  }
  if (rest.front() == ',') {
    return parseStructHelper(
        dialect, rest.drop_front().trim(), loc,
        ArrayRef<size_t>(std::next(matchingRAngleOffset.begin()),
                         matchingRAngleOffset.end()),
        memberTypes, layoutInfo);
  }
  emitError(loc, "unexpected string : '") << rest << "'";
  return failure();
}

// struct-type ::= `!spv.struct<` spirv-type (` [` integer-literal `]`)?
//                 (`, ` spirv-type ( ` [` integer-literal `] ` )? )*
static Type parseStructType(SPIRVDialect const &dialect, StringRef spec,
                            Location loc) {
  if (!spec.consume_front("struct<") || !spec.consume_back(">")) {
    emitError(loc, "spv.struct delimiter <...> mismatch");
    return Type();
  }

  if (spec.trim().empty()) {
    emitError(loc, "expected SPIR-V type");
    return Type();
  }

  SmallVector<Type, 4> memberTypes;
  SmallVector<StructType::LayoutInfo, 4> layoutInfo;
  SmallVector<size_t, 4> matchingRAngleOffset;
  if (!computeMatchingRAngles(loc, spec, matchingRAngleOffset) ||
      parseStructHelper(dialect, spec, loc, matchingRAngleOffset, memberTypes,
                        layoutInfo)) {
    return Type();
  }
  if (layoutInfo.empty()) {
    return StructType::get(memberTypes);
  }
  if (memberTypes.size() != layoutInfo.size()) {
    emitError(loc, "layout specification must be given for all members");
    return Type();
  }
  return StructType::get(memberTypes, layoutInfo);
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
  os << "array<" << type.getElementCount() << " x " << type.getElementType()
     << ">";
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
  std::string sep = "";
  for (size_t i = 0, e = type.getNumMembers(); i != e; ++i) {
    os << sep << type.getMemberType(i);
    if (type.hasLayout()) {
      os << " [" << type.getOffset(i) << "]";
    }
    sep = ", ";
  }
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
