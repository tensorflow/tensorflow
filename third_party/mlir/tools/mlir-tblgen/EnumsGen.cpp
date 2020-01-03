//===- EnumsGen.cpp - MLIR enum utility generator -------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EnumsGen generates common utility functions for enums.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::formatv;
using llvm::isDigit;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::StringRef;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::EnumAttrCase;

static std::string makeIdentifier(StringRef str) {
  if (!str.empty() && isDigit(static_cast<unsigned char>(str.front()))) {
    std::string newStr = std::string("_") + str.str();
    return newStr;
  }
  return str.str();
}

static void emitEnumClass(const Record &enumDef, StringRef enumName,
                          StringRef underlyingType, StringRef description,
                          const std::vector<EnumAttrCase> &enumerants,
                          raw_ostream &os) {
  os << "// " << description << "\n";
  os << "enum class " << enumName;

  if (!underlyingType.empty())
    os << " : " << underlyingType;
  os << " {\n";

  for (const auto &enumerant : enumerants) {
    auto symbol = makeIdentifier(enumerant.getSymbol());
    auto value = enumerant.getValue();
    if (value >= 0) {
      os << formatv("  {0} = {1},\n", symbol, value);
    } else {
      os << formatv("  {0},\n", symbol);
    }
  }
  os << "};\n\n";
}

static void emitDenseMapInfo(StringRef enumName, std::string underlyingType,
                             StringRef cppNamespace, raw_ostream &os) {
  std::string qualName = formatv("{0}::{1}", cppNamespace, enumName);
  if (underlyingType.empty())
    underlyingType = formatv("std::underlying_type<{0}>::type", qualName);

  const char *const mapInfo = R"(
namespace llvm {
template<> struct DenseMapInfo<{0}> {{
  using StorageInfo = llvm::DenseMapInfo<{1}>;

  static inline {0} getEmptyKey() {{
    return static_cast<{0}>(StorageInfo::getEmptyKey());
  }

  static inline {0} getTombstoneKey() {{
    return static_cast<{0}>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const {0} &val) {{
    return StorageInfo::getHashValue(static_cast<{1}>(val));
  }

  static bool isEqual(const {0} &lhs, const {0} &rhs) {{
    return lhs == rhs;
  }
};
})";
  os << formatv(mapInfo, qualName, underlyingType);
  os << "\n\n";
}

static void emitMaxValueFn(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef maxEnumValFnName = enumAttr.getMaxEnumValFnName();
  auto enumerants = enumAttr.getAllCases();

  unsigned maxEnumVal = 0;
  for (const auto &enumerant : enumerants) {
    int64_t value = enumerant.getValue();
    // Avoid generating the max value function if there is an enumerant without
    // explicit value.
    if (value < 0)
      return;

    maxEnumVal = std::max(maxEnumVal, static_cast<unsigned>(value));
  }

  // Emit the function to return the max enum value
  os << formatv("inline constexpr unsigned {0}() {{\n", maxEnumValFnName);
  os << formatv("  return {0};\n", maxEnumVal);
  os << "}\n\n";
}

// Returns the EnumAttrCase whose value is zero if exists; returns llvm::None
// otherwise.
static llvm::Optional<EnumAttrCase>
getAllBitsUnsetCase(llvm::ArrayRef<EnumAttrCase> cases) {
  for (auto attrCase : cases) {
    if (attrCase.getValue() == 0)
      return attrCase;
  }
  return llvm::None;
}

// Emits the following inline function for bit enums:
//
// inline <enum-type> operator|(<enum-type> a, <enum-type> b);
// inline <enum-type> operator&(<enum-type> a, <enum-type> b);
// inline <enum-type> bitEnumContains(<enum-type> a, <enum-type> b);
static void emitOperators(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::string underlyingType = enumAttr.getUnderlyingType();
  os << formatv("inline {0} operator|({0} lhs, {0} rhs) {{\n", enumName)
     << formatv("  return static_cast<{0}>("
                "static_cast<{1}>(lhs) | static_cast<{1}>(rhs));\n",
                enumName, underlyingType)
     << "}\n";
  os << formatv("inline {0} operator&({0} lhs, {0} rhs) {{\n", enumName)
     << formatv("  return static_cast<{0}>("
                "static_cast<{1}>(lhs) & static_cast<{1}>(rhs));\n",
                enumName, underlyingType)
     << "}\n";
  os << formatv(
            "inline bool bitEnumContains({0} bits, {0} bit) {{\n"
            "  return (static_cast<{1}>(bits) & static_cast<{1}>(bit)) != 0;\n",
            enumName, underlyingType)
     << "}\n";
}

static void emitSymToStrFnForIntEnum(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef symToStrFnName = enumAttr.getSymbolToStringFnName();
  StringRef symToStrFnRetType = enumAttr.getSymbolToStringFnRetType();
  auto enumerants = enumAttr.getAllCases();

  os << formatv("{2} {1}({0} val) {{\n", enumName, symToStrFnName,
                symToStrFnRetType);
  os << "  switch (val) {\n";
  for (const auto &enumerant : enumerants) {
    auto symbol = enumerant.getSymbol();
    os << formatv("    case {0}::{1}: return \"{2}\";\n", enumName,
                  makeIdentifier(symbol), symbol);
  }
  os << "  }\n";
  os << "  return \"\";\n";
  os << "}\n\n";
}

static void emitSymToStrFnForBitEnum(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef symToStrFnName = enumAttr.getSymbolToStringFnName();
  StringRef symToStrFnRetType = enumAttr.getSymbolToStringFnRetType();
  StringRef separator = enumDef.getValueAsString("separator");
  auto enumerants = enumAttr.getAllCases();
  auto allBitsUnsetCase = getAllBitsUnsetCase(enumerants);

  os << formatv("{2} {1}({0} symbol) {{\n", enumName, symToStrFnName,
                symToStrFnRetType);

  os << formatv("  auto val = static_cast<{0}>(symbol);\n",
                enumAttr.getUnderlyingType());
  if (allBitsUnsetCase) {
    os << "  // Special case for all bits unset.\n";
    os << formatv("  if (val == 0) return \"{0}\";\n\n",
                  allBitsUnsetCase->getSymbol());
  }
  os << "  llvm::SmallVector<llvm::StringRef, 2> strs;\n";
  for (const auto &enumerant : enumerants) {
    // Skip the special enumerant for None.
    if (auto val = enumerant.getValue())
      os << formatv("  if ({0}u & val) {{ strs.push_back(\"{1}\"); "
                    "val &= ~{0}u; }\n",
                    val, enumerant.getSymbol());
  }
  // If we have unknown bit set, return an empty string to signal errors.
  os << "\n  if (val) return \"\";\n";
  os << formatv("  return llvm::join(strs, \"{0}\");\n", separator);

  os << "}\n\n";
}

static void emitStrToSymFnForIntEnum(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef strToSymFnName = enumAttr.getStringToSymbolFnName();
  auto enumerants = enumAttr.getAllCases();

  os << formatv("llvm::Optional<{0}> {1}(llvm::StringRef str) {{\n", enumName,
                strToSymFnName);
  os << formatv("  return llvm::StringSwitch<llvm::Optional<{0}>>(str)\n",
                enumName);
  for (const auto &enumerant : enumerants) {
    auto symbol = enumerant.getSymbol();
    os << formatv("      .Case(\"{1}\", {0}::{2})\n", enumName, symbol,
                  makeIdentifier(symbol));
  }
  os << "      .Default(llvm::None);\n";
  os << "}\n";
}

static void emitStrToSymFnForBitEnum(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::string underlyingType = enumAttr.getUnderlyingType();
  StringRef strToSymFnName = enumAttr.getStringToSymbolFnName();
  StringRef separator = enumDef.getValueAsString("separator");
  auto enumerants = enumAttr.getAllCases();
  auto allBitsUnsetCase = getAllBitsUnsetCase(enumerants);

  os << formatv("llvm::Optional<{0}> {1}(llvm::StringRef str) {{\n", enumName,
                strToSymFnName);

  if (allBitsUnsetCase) {
    os << "  // Special case for all bits unset.\n";
    StringRef caseSymbol = allBitsUnsetCase->getSymbol();
    os << formatv("  if (str == \"{1}\") return {0}::{2};\n\n", enumName,
                  caseSymbol, makeIdentifier(caseSymbol));
  }

  // Split the string to get symbols for all the bits.
  os << "  llvm::SmallVector<llvm::StringRef, 2> symbols;\n";
  os << formatv("  str.split(symbols, \"{0}\");\n\n", separator);

  os << formatv("  {0} val = 0;\n", underlyingType);
  os << "  for (auto symbol : symbols) {\n";

  // Convert each symbol to the bit ordinal and set the corresponding bit.
  os << formatv(
      "    auto bit = llvm::StringSwitch<llvm::Optional<{0}>>(symbol)\n",
      underlyingType);
  for (const auto &enumerant : enumerants) {
    // Skip the special enumerant for None.
    if (auto val = enumerant.getValue())
      os.indent(6) << formatv(".Case(\"{0}\", {1})\n", enumerant.getSymbol(),
                              val);
  }
  os.indent(6) << ".Default(llvm::None);\n";

  os << "    if (bit) { val |= *bit; } else { return llvm::None; }\n";
  os << "  }\n";

  os << formatv("  return static_cast<{0}>(val);\n", enumName);
  os << "}\n\n";
}

static void emitUnderlyingToSymFnForIntEnum(const Record &enumDef,
                                            raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::string underlyingType = enumAttr.getUnderlyingType();
  StringRef underlyingToSymFnName = enumAttr.getUnderlyingToSymbolFnName();
  auto enumerants = enumAttr.getAllCases();

  // Avoid generating the underlying value to symbol conversion function if
  // there is an enumerant without explicit value.
  if (llvm::any_of(enumerants, [](EnumAttrCase enumerant) {
        return enumerant.getValue() < 0;
      }))
    return;

  os << formatv("llvm::Optional<{0}> {1}({2} value) {{\n", enumName,
                underlyingToSymFnName,
                underlyingType.empty() ? std::string("unsigned")
                                       : underlyingType)
     << "  switch (value) {\n";
  for (const auto &enumerant : enumerants) {
    auto symbol = enumerant.getSymbol();
    auto value = enumerant.getValue();
    os << formatv("  case {0}: return {1}::{2};\n", value, enumName,
                  makeIdentifier(symbol));
  }
  os << "  default: return llvm::None;\n"
     << "  }\n"
     << "}\n\n";
}

static void emitUnderlyingToSymFnForBitEnum(const Record &enumDef,
                                            raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  std::string underlyingType = enumAttr.getUnderlyingType();
  StringRef underlyingToSymFnName = enumAttr.getUnderlyingToSymbolFnName();
  auto enumerants = enumAttr.getAllCases();
  auto allBitsUnsetCase = getAllBitsUnsetCase(enumerants);

  os << formatv("llvm::Optional<{0}> {1}({2} value) {{\n", enumName,
                underlyingToSymFnName, underlyingType);
  if (allBitsUnsetCase) {
    os << "  // Special case for all bits unset.\n";
    os << formatv("  if (value == 0) return {0}::{1};\n\n", enumName,
                  makeIdentifier(allBitsUnsetCase->getSymbol()));
  }
  llvm::SmallVector<std::string, 8> values;
  for (const auto &enumerant : enumerants) {
    if (auto val = enumerant.getValue())
      values.push_back(formatv("{0}u", val));
  }
  os << formatv("  if (value & ~({0})) return llvm::None;\n",
                llvm::join(values, " | "));
  os << formatv("  return static_cast<{0}>(value);\n", enumName);
  os << "}\n";
}

static void emitEnumDecl(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef enumName = enumAttr.getEnumClassName();
  StringRef cppNamespace = enumAttr.getCppNamespace();
  std::string underlyingType = enumAttr.getUnderlyingType();
  StringRef description = enumAttr.getDescription();
  StringRef strToSymFnName = enumAttr.getStringToSymbolFnName();
  StringRef symToStrFnName = enumAttr.getSymbolToStringFnName();
  StringRef symToStrFnRetType = enumAttr.getSymbolToStringFnRetType();
  StringRef underlyingToSymFnName = enumAttr.getUnderlyingToSymbolFnName();
  auto enumerants = enumAttr.getAllCases();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  // Emit the enum class definition
  emitEnumClass(enumDef, enumName, underlyingType, description, enumerants, os);

  // Emit conversion function declarations
  if (llvm::all_of(enumerants, [](EnumAttrCase enumerant) {
        return enumerant.getValue() >= 0;
      })) {
    os << formatv(
        "llvm::Optional<{0}> {1}({2});\n", enumName, underlyingToSymFnName,
        underlyingType.empty() ? std::string("unsigned") : underlyingType);
  }
  os << formatv("{2} {1}({0});\n", enumName, symToStrFnName, symToStrFnRetType);
  os << formatv("llvm::Optional<{0}> {1}(llvm::StringRef);\n", enumName,
                strToSymFnName);

  if (enumAttr.isBitEnum()) {
    emitOperators(enumDef, os);
  } else {
    emitMaxValueFn(enumDef, os);
  }

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";

  // Emit DenseMapInfo for this enum class
  emitDenseMapInfo(enumName, underlyingType, cppNamespace, os);
}

static bool emitEnumDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Enum Utility Declarations", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    emitEnumDecl(*def, os);

  return false;
}

static void emitEnumDef(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef cppNamespace = enumAttr.getCppNamespace();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  if (enumAttr.isBitEnum()) {
    emitSymToStrFnForBitEnum(enumDef, os);
    emitStrToSymFnForBitEnum(enumDef, os);
    emitUnderlyingToSymFnForBitEnum(enumDef, os);
  } else {
    emitSymToStrFnForIntEnum(enumDef, os);
    emitStrToSymFnForIntEnum(enumDef, os);
    emitUnderlyingToSymFnForIntEnum(enumDef, os);
  }

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  os << "\n";
}

static bool emitEnumDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Enum Utility Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    emitEnumDef(*def, os);

  return false;
}

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDecls("gen-enum-decls", "Generate enum utility declarations",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   return emitEnumDecls(records, os);
                 });

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genEnumDefs("gen-enum-defs", "Generate enum utility definitions",
                [](const RecordKeeper &records, raw_ostream &os) {
                  return emitEnumDefs(records, os);
                });
