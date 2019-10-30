//===- EnumsGen.cpp - MLIR enum utility generator -------------------------===//
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
// EnumsGen generates common utility functions for enums.
//
//===----------------------------------------------------------------------===//

#include "EnumsGen.h"
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

static void emitSymToStrFn(const Record &enumDef, raw_ostream &os) {
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

static void emitStrToSymFn(const Record &enumDef, raw_ostream &os) {
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

static void emitUnderlyingToSymFn(const Record &enumDef, raw_ostream &os) {
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

void mlir::tblgen::emitEnumDecl(const Record &enumDef,
                                ExtraFnEmitter emitExtraFns, raw_ostream &os) {
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

  emitExtraFns(enumDef, os);

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";

  // Emit DenseMapInfo for this enum class
  emitDenseMapInfo(enumName, underlyingType, cppNamespace, os);
}

static bool emitEnumDecls(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Enum Utility Declarations", os);

  auto extraFnEmitter = [](const Record &enumDef, raw_ostream &os) {
    emitMaxValueFn(enumDef, os);
  };

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    if (!EnumAttr(def).skipAutoGen())
      mlir::tblgen::emitEnumDecl(*def, extraFnEmitter, os);

  return false;
}

static void emitEnumDef(const Record &enumDef, raw_ostream &os) {
  EnumAttr enumAttr(enumDef);
  StringRef cppNamespace = enumAttr.getCppNamespace();

  llvm::SmallVector<StringRef, 2> namespaces;
  llvm::SplitString(cppNamespace, namespaces, "::");

  for (auto ns : namespaces)
    os << "namespace " << ns << " {\n";

  emitSymToStrFn(enumDef, os);
  emitStrToSymFn(enumDef, os);
  emitUnderlyingToSymFn(enumDef, os);

  for (auto ns : llvm::reverse(namespaces))
    os << "} // namespace " << ns << "\n";
  os << "\n";
}

static bool emitEnumDefs(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("Enum Utility Definitions", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (const auto *def : defs)
    if (!EnumAttr(def).skipAutoGen())
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
