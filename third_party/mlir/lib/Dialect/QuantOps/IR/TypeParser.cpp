//===- TypeParser.h - Quantization Type Parser ------------------*- C++ -*-===//
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

#include "mlir/Dialect/QuantOps/QuantOps.h"
#include "mlir/Dialect/QuantOps/QuantTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace quant {

/// Print a floating point value in a way that the parser will be able to
/// round-trip losslessly.
static void printStabilizedFloat(const APFloat &apValue, raw_ostream &os) {
  // We would like to output the FP constant value in exponential notation,
  // but we cannot do this if doing so will lose precision.  Check here to
  // make sure that we only output it in exponential format if we can parse
  // the value back and get the same value.
  bool isInf = apValue.isInfinity();
  bool isNaN = apValue.isNaN();
  if (!isInf && !isNaN) {
    SmallString<128> strValue;
    apValue.toString(strValue, 6, 0, false);

    // Check to make sure that the stringized number is not some string like
    // "Inf" or NaN, that atof will accept, but the lexer will not.  Check
    // that the string matches the "[-+]?[0-9]" regex.
    assert(((strValue[0] >= '0' && strValue[0] <= '9') ||
            ((strValue[0] == '-' || strValue[0] == '+') &&
             (strValue[1] >= '0' && strValue[1] <= '9'))) &&
           "[-+]?[0-9] regex does not match!");
    // Reparse stringized version!
    if (APFloat(apValue.getSemantics(), strValue).bitwiseIsEqual(apValue)) {
      os << strValue;
      return;
    }
  }

  SmallVector<char, 16> str;
  apValue.toString(str);
  os << str;
}

namespace {

enum class TokenKind {
  error,
  eof,
  l_brace,
  r_brace,
  l_angle,
  r_angle,
  colon,
  comma,
  alpha_ident,
  integer_literal,
  float_literal,
};

struct Token {
  TokenKind kind;
  StringRef spelling;
};

class Lexer {
public:
  Lexer(StringRef source) : curBuffer(source), curPtr(curBuffer.begin()) {}

  Token lexToken();

private:
  Token formToken(TokenKind kind, const char *tokStart) {
    return Token{kind, StringRef(tokStart, curPtr - tokStart)};
  }

  Token emitError(const char *loc, const Twine &message) {
    return formToken(TokenKind::error, loc);
  }

  bool isEnd() const { return curPtr == curBuffer.end(); }

  // Lexer implementation methods
  Token lexalpha_ident(const char *tokStart);
  Token lexNumber(const char *tokStart);

  StringRef curBuffer;
  const char *curPtr;
};

} // namespace

Token Lexer::lexToken() {
  // Ignore whitespace.
  while (!isEnd()) {
    switch (*curPtr) {
    case ' ':
    case '\t':
    case '\n':
    case '\r':
      ++curPtr;
      continue;
    default:
      break;
    }
    break;
  }

  if (isEnd()) {
    return Token{TokenKind::eof, ""};
  }

  const char *tokStart = curPtr;
  switch (*curPtr++) {
  default:
    if (isalpha(*tokStart)) {
      return lexalpha_ident(tokStart);
    }
    if (isdigit(*tokStart)) {
      return lexNumber(tokStart);
    }

    return emitError(tokStart, "unexpected character");

  case '<':
    return formToken(TokenKind::l_angle, tokStart);
  case '>':
    return formToken(TokenKind::r_angle, tokStart);
  case '{':
    return formToken(TokenKind::l_brace, tokStart);
  case '}':
    return formToken(TokenKind::r_brace, tokStart);
  case ':':
    return formToken(TokenKind::colon, tokStart);
  case ',':
    return formToken(TokenKind::comma, tokStart);
  case '-':
    return lexNumber(tokStart);
  case '+':
    return lexNumber(tokStart);
  }
}

/// Lex a bare alpha identifier. Since this DSL often contains identifiers with
/// trailing numeric components, this only matches alphas. It is up to the
/// parser to handle identifiers that can be mixed alphanum.
///
///   alpha-ident ::= (letter)(letter)*
Token Lexer::lexalpha_ident(const char *tokStart) {
  while (!isEnd() && isalpha(*curPtr)) {
    ++curPtr;
  }
  return formToken(TokenKind::alpha_ident, tokStart);
}

/// Lex a number.
///
///   integer-literal ::= [-+]?digit+
///   float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
Token Lexer::lexNumber(const char *tokStart) {
  // Leading '+', '-' or digit has already been consumed.
  while (!isEnd() && isdigit(*curPtr)) {
    ++curPtr;
  }
  // If not a decimal point, treat as integer.
  if (isEnd() || *curPtr != '.') {
    return formToken(TokenKind::integer_literal, tokStart);
  }
  ++curPtr;

  // Skip over [0-9]*([eE][-+]?[0-9]+)?
  // Leading digits.
  while (!isEnd() && isdigit(*curPtr)) {
    ++curPtr;
  }

  // [eE][-+]?[0-9]+
  if (!isEnd() && (*curPtr == 'e' || *curPtr == 'E')) {
    auto remaining = curBuffer.end() - curPtr;
    if (remaining > 2 && isdigit(curPtr[1])) {
      // Lookahead 2 for digit.
      curPtr += 2;
      while (!isEnd() && isdigit(*curPtr)) {
        ++curPtr;
      }
    } else if (remaining > 3 && (curPtr[1] == '-' || curPtr[1] == '+') &&
               isdigit(curPtr[2])) {
      // Lookahead 3 for [+-] digit.
      curPtr += 3;
      while (!isEnd() && isdigit(*curPtr)) {
        ++curPtr;
      }
    }
  }
  return formToken(TokenKind::float_literal, tokStart);
} // end namespace

// --- TypeParser ---
namespace {

class TypeParser {
public:
  TypeParser(StringRef source, MLIRContext *context, Location location)
      : context(context), location(location), lexer(source),
        curToken(lexer.lexToken()) {}

  /// Attempts to parse the source as a type, returning the unknown
  /// type on error.
  Type parseType();

private:
  /// Unconditionally consumes the current token.
  void consumeToken() {
    assert(curToken.kind != TokenKind::eof &&
           "should not advance past EOF or errors");
    curToken = lexer.lexToken();
  }

  /// Unconditionally consumes the current token, asserting that it is of the
  /// specified kind.
  void consumeToken(TokenKind kind) {
    assert(curToken.kind == kind && "consumed an unexpected token");
    consumeToken();
  }

  /// Conditionally consumes a token if of the specified kind.
  /// Returns true if consumed.
  bool consumeIf(TokenKind kind) {
    if (curToken.kind == kind) {
      consumeToken();
      return true;
    }
    return false;
  }

  /// Emits an error at the current location with a message.
  void emitError(const Twine &message) {
    // TODO: All errors show up at the beginning of the extended type location.
    // Figure out how to make this location relative to where the error occurred
    // in this instance.
    mlir::emitError(location, message);
  }

  // Parsers.
  Type parseAnyType();
  Type parseUniformType();
  IntegerType parseStorageType(bool &isSigned);
  bool parseStorageRange(IntegerType storageType, bool isSigned,
                         int64_t &storageTypeMin, int64_t &storageTypeMax);
  FloatType parseExpressedType();
  bool parseQuantParams(double &scale, int64_t &zeroPoint);

  MLIRContext *context;
  Location location;
  Lexer lexer;

  // The next token that has not yet been consumed.
  Token curToken;
};

} // namespace

Type TypeParser::parseType() {
  // All types start with an identifier that we switch on.
  if (curToken.kind == TokenKind::alpha_ident) {
    StringRef typeNameSpelling = curToken.spelling;
    consumeToken();

    Type result;
    if (typeNameSpelling == "uniform") {
      result = parseUniformType();
      if (!result) {
        return nullptr;
      }
    } else if (typeNameSpelling == "any") {
      result = parseAnyType();
      if (!result) {
        return nullptr;
      }
    } else {
      return (emitError("unknown quantized type " + typeNameSpelling), nullptr);
    }

    // Make sure the entire input was consumed.
    if (curToken.kind != TokenKind::eof) {
      return (emitError("unrecognized token: " + curToken.spelling), nullptr);
    }

    return result;
  } else {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }
}

/// Parses a UniformQuantizedType.
///
///   uniform_per_layer ::= `any<` storage-spec (expressed-type-spec)?`>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
Type TypeParser::parseAnyType() {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;

  // Type specification.
  if (!consumeIf(TokenKind::l_angle)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Optional expressed type.
  if (consumeIf(TokenKind::colon)) {
    expressedType = parseExpressedType();
    if (!expressedType) {
      return nullptr;
    }
  }

  if (!consumeIf(TokenKind::r_angle)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }

  return AnyQuantizedType::getChecked(typeFlags, storageType, expressedType,
                                      storageTypeMin, storageTypeMax, location);
}

/// Parses a UniformQuantizedType.
///
///   uniform_type ::= uniform_per_layer
///                  | uniform_per_axis
///   uniform_per_layer ::= `uniform<` storage-spec expressed-type-spec
///                          `,` scale-zero `>`
///   uniform_per_axis ::= `uniform<` storage-spec expressed-type-spec
///                        axis-spec `,` scale-zero-list `>`
///   storage-spec ::= storage-type (`<` storage-range `>`)?
///   storage-range ::= integer-literal `:` integer-literal
///   storage-type ::= (`i` | `u`) integer-literal
///   expressed-type-spec ::= `:` `f` integer-literal
///   axis-spec ::= `:` integer-literal
///   scale-zero ::= float-literal `:` integer-literal
///   scale-zero-list ::= `{` scale-zero (`,` scale-zero)* `}`
Type TypeParser::parseUniformType() {
  IntegerType storageType;
  FloatType expressedType;
  unsigned typeFlags = 0;
  int64_t storageTypeMin;
  int64_t storageTypeMax;
  bool isPerAxis = false;
  int32_t quantizedDimension;
  SmallVector<double, 1> scales;
  SmallVector<int64_t, 1> zeroPoints;

  // Type specification.
  if (!consumeIf(TokenKind::l_angle)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }

  // Storage type.
  bool isSigned = false;
  storageType = parseStorageType(isSigned);
  if (!storageType) {
    return nullptr;
  }
  if (isSigned) {
    typeFlags |= QuantizationFlags::Signed;
  }

  // Storage type range.
  if (parseStorageRange(storageType, isSigned, storageTypeMin,
                        storageTypeMax)) {
    return nullptr;
  }

  // Expressed type.
  if (!consumeIf(TokenKind::colon)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }
  expressedType = parseExpressedType();
  if (!expressedType) {
    return nullptr;
  }

  // Optionally parse quantized dimension for per-axis quantization.
  if (consumeIf(TokenKind::colon)) {
    if (curToken.kind != TokenKind::integer_literal) {
      return (emitError("expected quantized dimension"), nullptr);
    }
    if (curToken.spelling.getAsInteger(10, quantizedDimension)) {
      return (emitError("illegal quantized dimension: " + curToken.spelling),
              nullptr);
    }
    consumeToken(TokenKind::integer_literal);
    isPerAxis = true;
  }

  // Comma leading into range_spec.
  if (!consumeIf(TokenKind::comma)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }

  // Parameter specification.
  // For per-axis, ranges are in a {} delimitted list.
  if (isPerAxis) {
    if (!consumeIf(TokenKind::l_brace)) {
      return (emitError("unrecognized token: " + curToken.spelling), nullptr);
    }
  }

  // Parse scales/zeroPoints.
  do {
    scales.resize(scales.size() + 1);
    zeroPoints.resize(zeroPoints.size() + 1);
    if (parseQuantParams(scales.back(), zeroPoints.back())) {
      return nullptr;
    }
  } while (isPerAxis && consumeIf(TokenKind::comma));

  if (isPerAxis) {
    if (!consumeIf(TokenKind::r_brace)) {
      return (emitError("unrecognized token: " + curToken.spelling), nullptr);
    }
  }

  if (!consumeIf(TokenKind::r_angle)) {
    return (emitError("unrecognized token: " + curToken.spelling), nullptr);
  }

  if (!isPerAxis && scales.size() > 1) {
    return (emitError("multiple scales/zeroPoints provided, but "
                      "quantizedDimension wasn't specified"),
            nullptr);
  }

  if (isPerAxis) {
    ArrayRef<double> scalesRef(scales.begin(), scales.end());
    ArrayRef<int64_t> zeroPointsRef(zeroPoints.begin(), zeroPoints.end());
    return UniformQuantizedPerAxisType::getChecked(
        typeFlags, storageType, expressedType, scalesRef, zeroPointsRef,
        quantizedDimension, storageTypeMin, storageTypeMax, location);
  }

  return UniformQuantizedType::getChecked(
      typeFlags, storageType, expressedType, scales.front(), zeroPoints.front(),
      storageTypeMin, storageTypeMax, location);
}

IntegerType TypeParser::parseStorageType(bool &isSigned) {
  // Parse storage type (alpha_ident, integer_literal).
  StringRef storageTypePrefix = curToken.spelling;
  unsigned storageTypeWidth;
  if (curToken.kind != TokenKind::alpha_ident) {
    return (emitError("expected storage type prefix"), nullptr);
  }
  consumeToken();
  if (curToken.kind != TokenKind::integer_literal) {
    return (emitError("expected storage type width"), nullptr);
  }
  if (curToken.spelling.getAsInteger(10, storageTypeWidth) ||
      storageTypeWidth == 0 ||
      storageTypeWidth > QuantizedType::MaxStorageBits) {
    return (emitError("illegal storage type size: " + Twine(curToken.spelling)),
            nullptr);
  }
  consumeToken();

  if (storageTypePrefix == "i") {
    isSigned = true;
    return IntegerType::get(storageTypeWidth, context);
  } else if (storageTypePrefix == "u") {
    isSigned = false;
    return IntegerType::get(storageTypeWidth, context);
  } else {
    return (
        emitError("illegal storage type prefix: " + Twine(storageTypePrefix)),
        nullptr);
  }
}

bool TypeParser::parseStorageRange(IntegerType storageType, bool isSigned,
                                   int64_t &storageTypeMin,
                                   int64_t &storageTypeMax) {

  int64_t defaultIntegerMin = QuantizedType::getDefaultMinimumForInteger(
      isSigned, storageType.getWidth());
  int64_t defaultIntegerMax = QuantizedType::getDefaultMaximumForInteger(
      isSigned, storageType.getWidth());
  if (consumeIf(TokenKind::l_angle)) {
    // Explicit storage min and storage max.
    if (curToken.kind != TokenKind::integer_literal) {
      return (emitError("expected storage type minimum"), true);
    }
    if (curToken.spelling.getAsInteger(10, storageTypeMin) ||
        storageTypeMin < defaultIntegerMin) {
      return (emitError("illegal storage type minimum: " + curToken.spelling),
              true);
    }
    consumeToken(TokenKind::integer_literal);

    if (!consumeIf(TokenKind::colon)) {
      return (emitError("unrecognized token: " + curToken.spelling), true);
    }

    if (curToken.kind != TokenKind::integer_literal) {
      return (emitError("expected storage type maximum"), true);
    }
    if (curToken.spelling.getAsInteger(10, storageTypeMax) ||
        storageTypeMax > defaultIntegerMax) {
      return (emitError("illegal storage type maximum: " + curToken.spelling),
              true);
    }
    consumeToken(TokenKind::integer_literal);

    if (!consumeIf(TokenKind::r_angle)) {
      return (emitError("unrecognized token: " + curToken.spelling), true);
    }
  } else {
    storageTypeMin = defaultIntegerMin;
    storageTypeMax = defaultIntegerMax;
  }

  return false;
}

FloatType TypeParser::parseExpressedType() {
  // Expect an alpha_ident followed by integer literal that we concat back
  // together.
  StringRef prefix = curToken.spelling;
  if (!consumeIf(TokenKind::alpha_ident)) {
    return (emitError("expected expressed type"), nullptr);
  }
  StringRef suffix = curToken.spelling;
  if (!consumeIf(TokenKind::integer_literal)) {
    return (emitError("expected expressed type"), nullptr);
  }

  SmallVector<char, 4> holder;
  StringRef typeName = (Twine(prefix) + Twine(suffix)).toStringRef(holder);
  if (typeName == "f32")
    return FloatType::getF32(context);
  if (typeName == "f16")
    return FloatType::getF16(context);
  if (typeName == "bf16")
    return FloatType::getBF16(context);
  if (typeName == "f64")
    return FloatType::getF64(context);

  return (emitError("unrecognized expressed type: " + typeName), nullptr);
}

bool TypeParser::parseQuantParams(double &scale, int64_t &zeroPoint) {
  // scale[:zeroPoint]?
  // scale.
  StringRef scaleSpelling = curToken.spelling;
  if (!consumeIf(TokenKind::float_literal) ||
      scaleSpelling.getAsDouble(scale)) {
    return (
        emitError("expected valid uniform scale. got: " + Twine(scaleSpelling)),
        true);
  }

  // zero point.
  zeroPoint = 0;
  if (!consumeIf(TokenKind::colon)) {
    // Default zero point.
    return false;
  }
  StringRef zeroPointSpelling = curToken.spelling;
  if (!consumeIf(TokenKind::integer_literal) ||
      zeroPointSpelling.getAsInteger(10, zeroPoint)) {
    return (emitError("expected integer uniform zero point. got: " +
                      Twine(zeroPointSpelling)),
            true);
  }

  return false;
}

/// Parse a type registered to this dialect.
Type QuantizationDialect::parseType(StringRef spec, Location loc) const {
  TypeParser parser(spec, getContext(), loc);
  Type parsedType = parser.parseType();
  if (parsedType == nullptr) {
    // Error.
    // TODO(laurenzo): Do something?
    return parsedType;
  }

  return parsedType;
}

static void printStorageType(QuantizedType type, raw_ostream &out) {
  // storage type
  unsigned storageWidth = type.getStorageTypeIntegralWidth();
  bool isSigned = type.isSigned();
  if (isSigned) {
    out << "i" << storageWidth;
  } else {
    out << "u" << storageWidth;
  }

  // storageTypeMin and storageTypeMax if not default.
  int64_t defaultIntegerMin =
      QuantizedType::getDefaultMinimumForInteger(isSigned, storageWidth);
  int64_t defaultIntegerMax =
      QuantizedType::getDefaultMaximumForInteger(isSigned, storageWidth);
  if (defaultIntegerMin != type.getStorageTypeMin() ||
      defaultIntegerMax != type.getStorageTypeMax()) {
    out << "<" << type.getStorageTypeMin() << ":" << type.getStorageTypeMax()
        << ">";
  }
}

static void printExpressedType(QuantizedType type, raw_ostream &out) {
  // repr type
  Type expressedType = type.getExpressedType();
  if (expressedType.isF32()) {
    out << "f32";
  } else if (expressedType.isF64()) {
    out << "f64";
  } else if (expressedType.isF16()) {
    out << "f16";
  } else if (expressedType.isBF16()) {
    out << "bf16";
  } else {
    out << "unknown";
  }
}

static void printQuantParams(double scale, int64_t zeroPoint,
                             raw_ostream &out) {
  printStabilizedFloat(APFloat(scale), out);
  if (zeroPoint != 0) {
    out << ":" << zeroPoint;
  }
}

/// Helper that prints a UniformQuantizedType.
static void printAnyQuantizedType(AnyQuantizedType type, raw_ostream &out) {
  out << "any<";
  printStorageType(type, out);
  if (type.getExpressedType()) {
    out << ":";
    printExpressedType(type, out);
  }
  out << ">";
}

/// Helper that prints a UniformQuantizedType.
static void printUniformQuantizedType(UniformQuantizedType type,
                                      raw_ostream &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":";
  printExpressedType(type, out);
  out << ", ";

  // scheme specific parameters
  printQuantParams(type.getScale(), type.getZeroPoint(), out);
  out << ">";
}

/// Helper that prints a UniformQuantizedPerAxisType.
static void printUniformQuantizedPerAxisType(UniformQuantizedPerAxisType type,
                                             raw_ostream &out) {
  out << "uniform<";
  printStorageType(type, out);
  out << ":";
  printExpressedType(type, out);
  out << ":";
  out << type.getQuantizedDimension();
  out << ", ";

  // scheme specific parameters
  ArrayRef<double> scales = type.getScales();
  ArrayRef<int64_t> zeroPoints = type.getZeroPoints();
  out << "{";
  for (unsigned i = 0; i < scales.size(); ++i) {
    printQuantParams(scales[i], zeroPoints[i], out);
    if (i != scales.size() - 1) {
      out << ",";
    }
  }
  out << "}>";
}

/// Print a type registered to this dialect.
void QuantizationDialect::printType(Type type, raw_ostream &os) const {
  switch (type.getKind()) {
  default:
    llvm_unreachable("Unhandled quantized type");
  case QuantizationTypes::Any:
    printAnyQuantizedType(type.cast<AnyQuantizedType>(), os);
    break;
  case QuantizationTypes::UniformQuantized:
    printUniformQuantizedType(type.cast<UniformQuantizedType>(), os);
    break;
  case QuantizationTypes::UniformQuantizedPerAxis:
    printUniformQuantizedPerAxisType(type.cast<UniformQuantizedPerAxisType>(),
                                     os);
    break;
  }
}

} // namespace quant
} // namespace mlir
