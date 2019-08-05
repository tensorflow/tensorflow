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

#include "mlir/Support/StringExtras.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::ArrayRef;
using llvm::formatv;
using llvm::raw_ostream;
using llvm::raw_string_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using llvm::SMLoc;
using llvm::StringRef;
using llvm::Twine;
using mlir::tblgen::Attribute;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::NamedTypeConstraint;
using mlir::tblgen::Operator;

// Writes the following function to `os`:
//   inline uint32_t getOpcode(<op-class-name>) { return <opcode>; }
static void emitGetOpcodeFunction(const Record *record, Operator const &op,
                                  raw_ostream &os) {
  os << formatv("template <> constexpr inline ::mlir::spirv::Opcode "
                "getOpcode<{0}>()",
                op.getQualCppClassName())
     << " {\n  "
     << formatv("return ::mlir::spirv::Opcode::{0};\n}\n",
                record->getValueAsString("spirvOpName"));
}

static void declareOpcodeFn(raw_ostream &os) {
  os << "template <typename OpClass> inline constexpr ::mlir::spirv::Opcode "
        "getOpcode();\n";
}

static void emitAttributeSerialization(const Attribute &attr,
                                       ArrayRef<SMLoc> loc, llvm::StringRef op,
                                       llvm::StringRef operandList,
                                       llvm::StringRef attrName,
                                       raw_ostream &os) {
  os << "    auto attr = " << op << ".getAttr(\"" << attrName << "\");\n";
  os << "    if (attr) {\n";
  if (attr.getAttrDefName() == "I32ArrayAttr") {
    // Serialize all the elements of the array
    os << "      for (auto attrElem : attr.cast<ArrayAttr>()) {\n";
    os << "        " << operandList
       << ".push_back(static_cast<uint32_t>(attrElem.cast<IntegerAttr>()."
          "getValue().getZExtValue()));\n";
    os << "      }\n";
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "I32Attr") {
    os << "      " << operandList
       << ".push_back(static_cast<uint32_t>(attr.cast<IntegerAttr>().getValue()"
          ".getZExtValue()));\n";
  } else {
    PrintFatalError(
        loc,
        llvm::Twine(
            "unhandled attribute type in SPIR-V serialization generation : '") +
            attr.getAttrDefName() + llvm::Twine("'"));
  }
  os << "    }\n";
}

static void emitSerializationFunction(const Record *attrClass,
                                      const Record *record, const Operator &op,
                                      raw_ostream &os) {
  // If the record has 'autogenSerialization' set to 0, nothing to do
  if (!record->getValueAsBit("autogenSerialization")) {
    return;
  }
  os << formatv("template <> LogicalResult\nSerializer::processOp<{0}>(\n"
                "  {0} op)",
                op.getQualCppClassName())
     << " {\n";
  os << "  SmallVector<uint32_t, 4> operands;\n";
  os << "  SmallVector<StringRef, 2> elidedAttrs;\n";

  // Serialize result information
  if (op.getNumResults() == 1) {
    os << "  uint32_t resultTypeID = 0;\n";
    os << "  if (failed(processType(op.getLoc(), op.getType(), resultTypeID))) "
          "{\n";
    os << "    return failure();\n";
    os << "  }\n";
    os << "  operands.push_back(resultTypeID);\n";
    // Create an SSA result <id> for the op
    os << "  auto resultID = getNextID();\n";
    os << "  valueIDMap[op.getResult()] = resultID;\n";
    os << "  operands.push_back(resultID);\n";
  } else if (op.getNumResults() != 0) {
    PrintFatalError(record->getLoc(), "SPIR-V ops can only zero or one result");
  }

  // Process arguments
  auto operandNum = 0;
  for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    os << "  {\n";
    if (argument.is<NamedTypeConstraint *>()) {
      os << "    for (auto arg : op.getODSOperands(" << operandNum << ")) {\n";
      os << "      auto argID = findValueID(arg);\n";
      os << "      if (!argID) {\n";
      os << "        emitError(op.getLoc(), \"operand " << operandNum
         << " has a use before def\");\n";
      os << "      }\n";
      os << "      operands.push_back(argID);\n";
      os << "    }\n";
      operandNum++;
    } else {
      auto attr = argument.get<NamedAttribute *>();
      emitAttributeSerialization(
          (attr->attr.isOptional() ? attr->attr.getBaseAttr() : attr->attr),
          record->getLoc(), "op", "operands", attr->name, os);
      os << "    elidedAttrs.push_back(\"" << attr->name << "\");\n";
    }
    os << "  }\n";
  }

  os << formatv("  encodeInstructionInto("
                "functions, spirv::getOpcode<{0}>(), operands);\n",
                op.getQualCppClassName());

  if (op.getNumResults() == 1) {
    // All non-argument attributes translated into OpDecorate instruction
    os << "  for (auto attr : op.getAttrs()) {\n";
    os << "    if (llvm::any_of(elidedAttrs, [&](StringRef elided) { return "
          "attr.first.is(elided); })) {\n";
    os << "      continue;\n";
    os << "    }\n";
    os << "    if (failed(processDecoration(op.getLoc(), resultID, attr))) {\n";
    os << "      return failure();";
    os << "    }\n";
    os << "  }\n";
  }

  os << "  return success();\n";
  os << "}\n\n";
}

static void initDispatchSerializationFn(raw_ostream &os) {
  os << "LogicalResult Serializer::dispatchToAutogenSerialization(Operation "
        "*op) {\n ";
}

static void emitSerializationDispatch(const Operator &op, raw_ostream &os) {
  os << formatv(" if (isa<{0}>(op)) ", op.getQualCppClassName()) << "{\n";
  os << "    ";
  os << formatv("return processOp<{0}>(cast<{0}>(op));\n",
                op.getQualCppClassName());
  os << "  } else";
}

static void finalizeDispatchSerializationFn(raw_ostream &os) {
  os << " {\n";
  os << "    return op->emitError(\"unhandled operation serialization\");\n";
  os << "  }\n";
  os << "  return success();\n";
  os << "}\n\n";
}

static void emitAttributeDeserialization(
    const Attribute &attr, ArrayRef<SMLoc> loc, llvm::StringRef attrList,
    llvm::StringRef attrName, llvm::StringRef operandsList,
    llvm::StringRef wordIndex, llvm::StringRef wordCount, raw_ostream &os) {
  if (attr.getAttrDefName() == "I32ArrayAttr") {
    os << "    SmallVector<Attribute, 4> attrListElems;\n";
    os << "    while (" << wordIndex << " < " << wordCount << ") {\n";
    os << "      attrListElems.push_back(opBuilder.getI32IntegerAttr("
       << operandsList << "[" << wordIndex << "++]));\n";
    os << "    }\n";
    os << "    " << attrList << ".push_back(opBuilder.getNamedAttr(\""
       << attrName << "\", opBuilder.getArrayAttr(attrListElems)));\n";
  } else if (attr.isEnumAttr() || attr.getAttrDefName() == "I32Attr") {
    os << "    " << attrList << ".push_back(opBuilder.getNamedAttr(\""
       << attrName << "\", opBuilder.getI32IntegerAttr(" << operandsList << "["
       << wordIndex << "++])));\n";
  } else {
    PrintFatalError(
        loc, llvm::Twine(
                 "unhandled attribute type in deserialization generation : '") +
                 attr.getAttrDefName() + llvm::Twine("'"));
  }
}

static void emitDeserializationFunction(const Record *attrClass,
                                        const Record *record,
                                        const Operator &op, raw_ostream &os) {
  // If the record has 'autogenSerialization' set to 0, nothing to do
  if (!record->getValueAsBit("autogenSerialization")) {
    return;
  }
  os << formatv("template <> "
                "LogicalResult\nDeserializer::processOp<{0}>(ArrayRef<"
                "uint32_t> words)",
                op.getQualCppClassName());
  os << " {\n";
  os << "  SmallVector<Type, 1> resultTypes;\n";
  os << "  size_t wordIndex = 0; (void)wordIndex;\n";

  // Deserialize result information if it exists
  bool hasResult = false;
  if (op.getNumResults() == 1) {
    os << "  {\n";
    os << "    if (wordIndex >= words.size()) {\n";
    os << "      "
       << formatv("return emitError(unknownLoc, \"expected result type <id> "
                  "while deserializing {0}\");\n",
                  op.getQualCppClassName());
    os << "    }\n";
    os << "    auto ty = getType(words[wordIndex]);\n";
    os << "    if (!ty) {\n";
    os << "      return emitError(unknownLoc, \"unknown type result <id> : "
          "\") << words[wordIndex];\n";
    os << "    }\n";
    os << "    resultTypes.push_back(ty);\n";
    os << "    wordIndex++;\n";
    os << "  }\n";
    os << "  if (wordIndex >= words.size()) {\n";
    os << "    "
       << formatv("return emitError(unknownLoc, \"expected result <id> while "
                  "deserializing {0}\");\n",
                  op.getQualCppClassName());
    os << "  }\n";
    os << "  uint32_t valueID = words[wordIndex++];\n";
    hasResult = true;
  } else if (op.getNumResults() != 0) {
    PrintFatalError(record->getLoc(),
                    "SPIR-V ops can have only zero or one result");
  }

  // Process operands/attributes
  os << "  SmallVector<Value *, 4> operands;\n";
  os << "  SmallVector<NamedAttribute, 4> attributes;\n";
  unsigned operandNum = 0;
  for (unsigned i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (auto valueArg = argument.dyn_cast<NamedTypeConstraint *>()) {
      if (valueArg->isVariadic()) {
        if (i != e - 1) {
          PrintFatalError(record->getLoc(),
                          "SPIR-V ops can have Variadic<..> argument only if "
                          "it's the last argument");
        }
        os << "  for (; wordIndex < words.size(); ++wordIndex)";
      } else {
        os << "  if (wordIndex < words.size())";
      }
      os << " {\n";
      os << "    auto arg = getValue(words[wordIndex]);\n";
      os << "    if (!arg) {\n";
      os << "      return emitError(unknownLoc, \"unknown result <id> : \") << "
            "words[wordIndex];\n";
      os << "    }\n";
      os << "    operands.push_back(arg);\n";
      if (!valueArg->isVariadic()) {
        os << "    wordIndex++;\n";
      }
      operandNum++;
      os << "  }\n";
    } else {
      os << "  if (wordIndex < words.size()) {\n";
      auto attr = argument.get<NamedAttribute *>();
      emitAttributeDeserialization(
          (attr->attr.isOptional() ? attr->attr.getBaseAttr() : attr->attr),
          record->getLoc(), "attributes", attr->name, "words", "wordIndex",
          "words.size()", os);
      os << "  }\n";
    }
  }

  os << "  if (wordIndex != words.size()) {\n";
  os << "    return emitError(unknownLoc, \"found more operands than expected "
        "when deserializing "
     << op.getQualCppClassName()
     << ", only \") << wordIndex << \" of \" << words.size() << \" "
        "processed\";\n";
  os << "  }\n";
  os << formatv("  auto op = opBuilder.create<{0}>(unknownLoc, resultTypes, "
                "operands, attributes); (void)op;\n",
                op.getQualCppClassName());
  if (hasResult) {
    os << "  valueMap[valueID] = op.getResult();\n\n";
  }

  // Import decorations parsed
  if (op.getNumResults() == 1) {
    os << "  if (decorations.count(valueID)) {\n";
    os << "    auto decorationAttrs = decorations[valueID];\n";
    os << "    for (auto attr : decorationAttrs.getAttrs()) {\n";
    os << "      op.setAttr(attr.first, attr.second);\n";
    os << "    }\n";
    os << "  }\n";
  }

  os << "  return success();\n";
  os << "}\n\n";
}

static void initDispatchDeserializationFn(raw_ostream &os) {
  os << "LogicalResult "
        "Deserializer::dispatchToAutogenDeserialization(spirv::Opcode "
        "opcode, ArrayRef<uint32_t> words) {\n";
  os << "  switch (opcode) {\n";
}

static void emitDeserializationDispatch(const Operator &op, const Record *def,
                                        raw_ostream &os) {
  os << formatv("  case spirv::Opcode::{0}:\n",
                def->getValueAsString("spirvOpName"));
  os << formatv("    return processOp<{0}>(words);\n",
                op.getQualCppClassName());
}

static void finalizeDispatchDeserializationFn(raw_ostream &os) {
  os << "  default:\n";
  os << "    ;\n";
  os << "  }\n";
  os << "  return emitError(unknownLoc, \"unhandled deserialization of \") << "
        "spirv::stringifyOpcode(opcode);\n";
  os << "}\n";
}

static bool emitSerializationFns(const RecordKeeper &recordKeeper,
                                 raw_ostream &os) {
  llvm::emitSourceFileHeader("SPIR-V Serialization Utilities/Functions", os);

  std::string dSerFnString, dDesFnString, serFnString, deserFnString,
      utilsString;
  raw_string_ostream dSerFn(dSerFnString), dDesFn(dDesFnString),
      serFn(serFnString), deserFn(deserFnString), utils(utilsString);
  auto attrClass = recordKeeper.getClass("Attr");

  declareOpcodeFn(utils);
  initDispatchSerializationFn(dSerFn);
  initDispatchDeserializationFn(dDesFn);
  auto defs = recordKeeper.getAllDerivedDefinitions("SPV_Op");
  for (const auto *def : defs) {
    if (!def->getValueAsBit("hasOpcode")) {
      continue;
    }
    Operator op(def);
    emitGetOpcodeFunction(def, op, utils);
    emitSerializationFunction(attrClass, def, op, serFn);
    emitSerializationDispatch(op, dSerFn);
    emitDeserializationFunction(attrClass, def, op, deserFn);
    emitDeserializationDispatch(op, def, dDesFn);
  }
  finalizeDispatchSerializationFn(dSerFn);
  finalizeDispatchDeserializationFn(dDesFn);

  os << "#ifdef GET_SPIRV_SERIALIZATION_UTILS\n";
  os << utils.str();
  os << "#endif // GET_SPIRV_SERIALIZATION_UTILS\n\n";

  os << "#ifdef GET_SERIALIZATION_FNS\n\n";
  os << serFn.str();
  os << dSerFn.str();
  os << "#endif // GET_SERIALIZATION_FNS\n\n";

  os << "#ifdef GET_DESERIALIZATION_FNS\n\n";
  os << deserFn.str();
  os << dDesFn.str();
  os << "#endif // GET_DESERIALIZATION_FNS\n\n";

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

static void emitEnumGetAttrNameFnDefn(const EnumAttr &enumAttr,
                                      raw_ostream &os) {
  auto enumName = enumAttr.getEnumClassName();
  os << formatv("template <> inline StringRef attributeName<{0}>()", enumName)
     << " {\n";
  os << "  "
     << formatv("static constexpr const char attrName[] = \"{0}\";\n",
                mlir::convertToSnakeCase(enumName));
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
  os << "#ifndef SPIRV_OP_UTILS_H_\n";
  os << "#define SPIRV_OP_UTILS_H_\n";
  emitEnumGetAttrNameFnDecl(os);
  emitEnumGetSymbolizeFnDecl(os);
  for (const auto *def : defs) {
    EnumAttr enumAttr(*def);
    emitEnumGetAttrNameFnDefn(enumAttr, os);
    emitEnumGetSymbolizeFnDefn(enumAttr, os);
  }
  os << "#endif // SPIRV_OP_UTILS_H\n";
  return false;
}

// Registers the enum utility generator to mlir-tblgen.
static mlir::GenRegistration genSerialization(
    "gen-spirv-serialization",
    "Generate SPIR-V (de)serialization utilities and functions",
    [](const RecordKeeper &records, raw_ostream &os) {
      return emitSerializationFns(records, os);
    });

static mlir::GenRegistration
    genOpUtils("gen-spirv-op-utils",
               "Generate SPIR-V operation utility definitions",
               [](const RecordKeeper &records, raw_ostream &os) {
                 return emitOpUtils(records, os);
               });
