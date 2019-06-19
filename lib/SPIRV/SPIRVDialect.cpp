//===- LLVMDialect.cpp - MLIR SPIR-V dialect ------------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
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

#include <type_traits>

using namespace mlir;
using namespace mlir::spirv;

//===----------------------------------------------------------------------===//
// SPIR-V Dialect
//===----------------------------------------------------------------------===//

SPIRVDialect::SPIRVDialect(MLIRContext *context)
    : Dialect(getDialectNamespace(), context) {
  addTypes<ArrayType, ImageType, PointerType, RuntimeArrayType>();

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

static Type parseAndVerifyTypeImpl(SPIRVDialect const &dialect, Location loc,
                                   StringRef spec) {
  auto *context = dialect.getContext();
  auto type = mlir::parseType(spec, context);
  if (!type) {
    context->emitError(loc, "cannot parse type: ") << spec;
    return Type();
  }

  // Allow SPIR-V dialect types
  if (&type.getDialect() == &dialect)
    return type;

  // Check other allowed types
  if (auto t = type.dyn_cast<FloatType>()) {
    if (type.isBF16()) {
      context->emitError(loc, "cannot use 'bf16' to compose SPIR-V types");
      return Type();
    }
  } else if (auto t = type.dyn_cast<IntegerType>()) {
    if (!llvm::is_contained(llvm::ArrayRef<unsigned>({8, 16, 32, 64}),
                            t.getWidth())) {
      context->emitError(loc,
                         "only 8/16/32/64-bit integer type allowed but found ")
          << type;
      return Type();
    }
  } else if (auto t = type.dyn_cast<VectorType>()) {
    if (t.getRank() != 1) {
      context->emitError(loc, "only 1-D vector allowed but found ") << t;
      return Type();
    }
  } else {
    context->emitError(loc, "cannot use ")
        << type << " to compose SPIR-V types";
    return Type();
  }

  return type;
}

Type SPIRVDialect::parseAndVerifyType(StringRef spec, Location loc) const {
  return parseAndVerifyTypeImpl(*this, loc, spec);
}

// element-type ::= integer-type
//                | floating-point-type
//                | vector-type
//                | spirv-type
//
// array-type ::= `!spv.array<` integer-literal `x` element-type `>`
Type SPIRVDialect::parseArrayType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("array<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.array delimiter <...> mismatch");
    return Type();
  }

  int64_t count = 0;
  spec = spec.trim();
  if (!parseNumberX(spec, count)) {
    context->emitError(
        loc, "expected array element count followed by 'x' but found '")
        << spec << "'";
    return Type();
  }

  if (spec.trim().empty()) {
    context->emitError(loc, "expected element type");
    return Type();
  }

  Type elementType = parseAndVerifyType(spec, loc);
  if (!elementType)
    return Type();

  return ArrayType::get(elementType, count);
}

// storage-class ::= `UniformConstant`
//                 | `Uniform`
//                 | `Workgroup`
//                 | <and other storage classes...>
//
// pointer-type ::= `!spv.ptr<` element-type `,` storage-class `>`
Type SPIRVDialect::parsePointerType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("ptr<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.ptr delimiter <...> mismatch");
    return Type();
  }

  // Split into pointee type and storage class
  StringRef scSpec, ptSpec;
  std::tie(ptSpec, scSpec) = spec.rsplit(',');
  if (scSpec.empty()) {
    context->emitError(
        loc, "expected comma to separate pointee type and storage class in '")
        << spec << "'";
    return Type();
  }

  scSpec = scSpec.trim();
  auto storageClass = symbolizeStorageClass(scSpec);
  if (!storageClass) {
    context->emitError(loc, "unknown storage class: ") << scSpec;
    return Type();
  }

  if (ptSpec.trim().empty()) {
    context->emitError(loc, "expected pointee type");
    return Type();
  }

  auto pointeeType = parseAndVerifyType(ptSpec, loc);
  if (!pointeeType)
    return Type();

  return PointerType::get(pointeeType, *storageClass);
}

// runtime-array-type ::= `!spv.rtarray<` element-type `>`
Type SPIRVDialect::parseRuntimeArrayType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("rtarray<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.rtarray delimiter <...> mismatch");
    return Type();
  }

  if (spec.trim().empty()) {
    context->emitError(loc, "expected element type");
    return Type();
  }

  Type elementType = parseAndVerifyType(spec, loc);
  if (!elementType)
    return Type();

  return RuntimeArrayType::get(elementType);
}

// Specialize this function to parse each of the parameters that define an
// ImageType
template <typename ValTy>
Optional<ValTy> parseAndVerify(SPIRVDialect const &dialect, Location loc,
                               StringRef spec) {
  auto *context = dialect.getContext();
  context->emitError(loc, "unexpected parameter while parsing '")
      << spec << "'";
  return llvm::None;
}

template <>
Optional<Type> parseAndVerify<Type>(SPIRVDialect const &dialect, Location loc,
                                    StringRef spec) {
  // TODO(ravishankarm): Further verify that the element type can be sampled
  return parseAndVerifyTypeImpl(dialect, loc, spec);
}

template <>
Optional<Dim> parseAndVerify<Dim>(SPIRVDialect const &dialect, Location loc,
                                  StringRef spec) {
  auto dim = symbolizeDim(spec);
  if (!dim) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown Dim in Image type: '") << spec << "'";
  }
  return dim;
}

template <>
Optional<ImageDepthInfo>
parseAndVerify<ImageDepthInfo>(SPIRVDialect const &dialect, Location loc,
                               StringRef spec) {
  auto depth = symbolizeImageDepthInfo(spec);
  if (!depth) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown ImageDepthInfo in Image type: '")
        << spec << "'";
  }
  return depth;
}

template <>
Optional<ImageArrayedInfo>
parseAndVerify<ImageArrayedInfo>(SPIRVDialect const &dialect, Location loc,
                                 StringRef spec) {
  auto arrayedInfo = symbolizeImageArrayedInfo(spec);
  if (!arrayedInfo) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown ImageArrayedInfo in Image type: '")
        << spec << "'";
  }
  return arrayedInfo;
}

template <>
Optional<ImageSamplingInfo>
parseAndVerify<ImageSamplingInfo>(SPIRVDialect const &dialect, Location loc,
                                  StringRef spec) {
  auto samplingInfo = symbolizeImageSamplingInfo(spec);
  if (!samplingInfo) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown ImageSamplingInfo in Image type: '")
        << spec << "'";
  }
  return samplingInfo;
}

template <>
Optional<ImageSamplerUseInfo>
parseAndVerify<ImageSamplerUseInfo>(SPIRVDialect const &dialect, Location loc,
                                    StringRef spec) {
  auto samplerUseInfo = symbolizeImageSamplerUseInfo(spec);
  if (!samplerUseInfo) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown ImageSamplerUseInfo in Image type: '")
        << spec << "'";
  }
  return samplerUseInfo;
}

template <>
Optional<ImageFormat> parseAndVerify<ImageFormat>(SPIRVDialect const &dialect,
                                                  Location loc,
                                                  StringRef spec) {
  auto format = symbolizeImageFormat(spec);
  if (!format) {
    auto *context = dialect.getContext();
    context->emitError(loc, "unknown ImageFormat in Image type: '")
        << spec << "'";
  }
  return format;
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
    auto *context = dialect.getContext();
    std::tie(parseSpec, restSpec) = spec.split(',');

    parseSpec = parseSpec.trim();
    if (numArgs != 0 && restSpec.empty()) {
      context->emitError(loc, "expected more parameters for image type '")
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
Type SPIRVDialect::parseImageType(StringRef spec, Location loc) const {
  auto *context = getContext();
  if (!spec.consume_front("image<") || !spec.consume_back(">")) {
    context->emitError(loc, "spv.image delimiter <...> mismatch");
    return Type();
  }

  auto value =
      parseCommaSeparatedList<Type, Dim, ImageDepthInfo, ImageArrayedInfo,
                              ImageSamplingInfo, ImageSamplerUseInfo,
                              ImageFormat>{}(*this, loc, spec);
  if (!value) {
    return Type();
  }

  return ImageType::get(value.getValue());
}

Type SPIRVDialect::parseType(StringRef spec, Location loc) const {
  if (spec.startswith("array"))
    return parseArrayType(spec, loc);
  if (spec.startswith("image"))
    return parseImageType(spec, loc);
  if (spec.startswith("ptr"))
    return parsePointerType(spec, loc);
  if (spec.startswith("rtarray"))
    return parseRuntimeArrayType(spec, loc);

  getContext()->emitError(loc, "unknown SPIR-V type: ") << spec;
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
  case TypeKind::ImageType:
    print(type.cast<ImageType>(), os);
    return;
  default:
    llvm_unreachable("unhandled SPIR-V type");
  }
}
