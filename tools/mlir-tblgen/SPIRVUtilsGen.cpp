//===- SPIRVSerializationGen.cpp - SPIR-V serialization utility generator -===//
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
// SPIRVSerializationGen generates common utility functions for SPIR-V
// serialization.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::formatv;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::Operator;

// Writes the following function to `os`:
//   inline uint32_t getOpcode(<op-class-name>) { return <opcode>; }
static void emitGetOpcodeFunction(const llvm::Record &record,
                                  Operator const &op, raw_ostream &os) {
  if (record.getValueAsInt("hasOpcode")) {
    os << formatv("template <> constexpr inline uint32_t getOpcode<{0}>()",
                  op.getQualCppClassName())
       << " {\n  return static_cast<uint32_t>("
       << formatv("Opcode::Op{0});\n}\n", record.getValueAsString("opName"));
  }
}

static bool emitSerializationUtils(const RecordKeeper &recordKeeper,
                                   raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Serialization Utilities", os);

  /// Define the function to get the opcode
  os << "template <typename OpClass> inline constexpr uint32_t getOpcode();\n";
  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_Op");
  for (const auto *def : defs) {
    Operator op(def);
    emitGetOpcodeFunction(*def, op, os);
  }

  return false;
}

static void emitEnumGetAttrNameFnDecl(raw_ostream &os) {
  os << formatv("template <typename EnumClass> inline constexpr StringRef "
                "attributeName();\n");
}

static void emitEnumGetSymbolizeFnDecl(raw_ostream &os) {
  os << "template <typename EnumClass> using SymbolizeFnTy = "
        "llvm::Optional<EnumClass> (*)(StringRef);\n";
  os << "template <typename EnumClass> inline constexpr "
        "SymbolizeFnTy<EnumClass> symbolizeEnum();\n";
}

std::string convertSnakeCase(llvm::StringRef inputString) {
  std::string snakeCase;
  for (auto c : inputString) {
    if (c >= 'A' && c <= 'Z') {
      if (!snakeCase.empty()) {
        snakeCase.push_back('_');
      }
      snakeCase.push_back((c - 'A') + 'a');
    } else {
      snakeCase.push_back(c);
    }
  }
  return snakeCase;
}

static void emitEnumGetAttrNameFnDefn(const EnumAttr &enumAttr,
                                      raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  os << formatv("template <> inline StringRef attributeName<{0}>()", enumName)
     << " {\n";
  os << "  "
     << formatv("static constexpr const char attrName[] = \"{0}\";\n",
                convertSnakeCase(enumName));
  os << "  return attrName;\n";
  os << "}\n";
}

static void emitEnumGetSymbolizeFnDefn(const EnumAttr &enumAttr,
                                       raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  auto strToSymFnName = enumAttr.getStringToSymbolFnName();
  os << formatv("template <> inline SymbolizeFnTy<{0}> symbolizeEnum<{0}>()",
                enumName)
     << " {\n";
  os << "  return " << strToSymFnName << ";\n";
  os << "}\n";
}

static bool emitOpUtils(const RecordKeeper &recordKeeper, raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Op Utilites", os);

  auto defs = recordKeeper.getAllDerivedDefinitions("I32EnumAttr");
  emitEnumGetAttrNameFnDecl(os);
  emitEnumGetSymbolizeFnDecl(os);
  for (const auto *def : defs) {
    EnumAttr enumAttr(*def);
    emitEnumGetAttrNameFnDefn(enumAttr, os);
    emitEnumGetSymbolizeFnDefn(enumAttr, os);
  }
  return false;
}

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration
    genSerializationDefs("gen-spirv-serial",
                         "Generate SPIR-V serialization utility definitions",
                         [](const RecordKeeper &records, raw_ostream &os) {
                           return emitSerializationUtils(records, os);
                         });

static mlir::GenRegistration
    genOpUtils("gen-spirv-op-utils",
               "Generate SPIR-V operation utility definitions",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpUtils(records, os);
               });
