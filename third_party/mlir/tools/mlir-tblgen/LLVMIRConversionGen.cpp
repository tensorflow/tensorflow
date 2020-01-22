//===- LLVMIRConversionGen.cpp - MLIR LLVM IR builder generator -----------===//
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
// This file uses tablegen definitions of the LLVM IR Dialect operations to
// generate the code building the LLVM IR from it.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

static bool emitError(const Twine &message) {
  llvm::errs() << message << "\n";
  return false;
}

namespace {
// Helper structure to return a position of the substring in a string.
struct StringLoc {
  size_t pos;
  size_t length;

  // Take a substring identified by this location in the given string.
  StringRef in(StringRef str) const { return str.substr(pos, length); }

  // A location is invalid if its position is outside the string.
  explicit operator bool() { return pos != std::string::npos; }
};
} // namespace

// Find the next TableGen variable in the given pattern.  These variables start
// with a `$` character and can contain alphanumeric characters or underscores.
// Return the position of the variable in the pattern and its length, including
// the `$` character.  The escape syntax `$$` is also detected and returned.
static StringLoc findNextVariable(StringRef str) {
  size_t startPos = str.find('$');
  if (startPos == std::string::npos)
    return {startPos, 0};

  // If we see "$$", return immediately.
  if (startPos != str.size() - 1 && str[startPos + 1] == '$')
    return {startPos, 2};

  // Otherwise, the symbol spans until the first character that is not
  // alphanumeric or '_'.
  size_t endPos = str.find_if_not([](char c) { return isAlnum(c) || c == '_'; },
                                  startPos + 1);
  if (endPos == std::string::npos)
    endPos = str.size();

  return {startPos, endPos - startPos};
}

// Check if `name` is the name of the variadic operand of `op`.  The variadic
// operand can only appear at the last position in the list of operands.
static bool isVariadicOperandName(const tblgen::Operator &op, StringRef name) {
  unsigned numOperands = op.getNumOperands();
  if (numOperands == 0)
    return false;
  const auto &operand = op.getOperand(numOperands - 1);
  return operand.isVariadic() && operand.name == name;
}

// Check if `result` is a known name of a result of `op`.
static bool isResultName(const tblgen::Operator &op, StringRef name) {
  for (int i = 0, e = op.getNumResults(); i < e; ++i)
    if (op.getResultName(i) == name)
      return true;
  return false;
}

// Check if `name` is a known name of an attribute of `op`.
static bool isAttributeName(const tblgen::Operator &op, StringRef name) {
  return llvm::any_of(
      op.getAttributes(),
      [name](const tblgen::NamedAttribute &attr) { return attr.name == name; });
}

// Check if `name` is a known name of an operand of `op`.
static bool isOperandName(const tblgen::Operator &op, StringRef name) {
  for (int i = 0, e = op.getNumOperands(); i < e; ++i)
    if (op.getOperand(i).name == name)
      return true;
  return false;
}

// Emit to `os` the operator-name driven check and the call to LLVM IRBuilder
// for one definition of a LLVM IR Dialect operation.  Return true on success.
static bool emitOneBuilder(const Record &record, raw_ostream &os) {
  auto op = tblgen::Operator(record);

  if (!record.getValue("llvmBuilder"))
    return emitError("no 'llvmBuilder' field for op " + op.getOperationName());

  // Return early if there is no builder specified.
  auto builderStrRef = record.getValueAsString("llvmBuilder");
  if (builderStrRef.empty())
    return true;

  // Progressively create the builder string by replacing $-variables with
  // value lookups.  Keep only the not-yet-traversed part of the builder pattern
  // to avoid re-traversing the string multiple times.
  std::string builder;
  llvm::raw_string_ostream bs(builder);
  while (auto loc = findNextVariable(builderStrRef)) {
    auto name = loc.in(builderStrRef).drop_front();
    // First, insert the non-matched part as is.
    bs << builderStrRef.substr(0, loc.pos);
    // Then, rewrite the name based on its kind.
    bool isVariadicOperand = isVariadicOperandName(op, name);
    if (isOperandName(op, name)) {
      auto result = isVariadicOperand
                        ? formatv("lookupValues(op.{0}())", name)
                        : formatv("valueMapping.lookup(op.{0}())", name);
      bs << result;
    } else if (isAttributeName(op, name)) {
      bs << formatv("op.{0}()", name);
    } else if (isResultName(op, name)) {
      bs << formatv("valueMapping[op.{0}()]", name);
    } else if (name == "_resultType") {
      bs << "op.getResult()->getType().cast<LLVM::LLVMType>()."
            "getUnderlyingType()";
    } else if (name == "_hasResult") {
      bs << "opInst.getNumResults() == 1";
    } else if (name == "_location") {
      bs << "opInst.getLoc()";
    } else if (name == "_numOperands") {
      bs << "opInst.getNumOperands()";
    } else if (name == "$") {
      bs << '$';
    } else {
      return emitError(name + " is neither an argument nor a result of " +
                       op.getOperationName());
    }
    // Finally, only keep the untraversed part of the string.
    builderStrRef = builderStrRef.substr(loc.pos + loc.length);
  }

  // Output the check and the rewritten builder string.
  os << "if (auto op = dyn_cast<" << op.getQualCppClassName()
     << ">(opInst)) {\n";
  os << bs.str() << builderStrRef << "\n";
  os << "  return success();\n";
  os << "}\n";

  return true;
}

// Emit all builders.  Returns false on success because of the generator
// registration requirements.
static bool emitBuilders(const RecordKeeper &recordKeeper, raw_ostream &os) {
  for (const auto *def : recordKeeper.getAllDerivedDefinitions("LLVM_OpBase")) {
    if (!emitOneBuilder(*def, os))
      return true;
  }
  return false;
}

static mlir::GenRegistration
    genLLVMIRConversions("gen-llvmir-conversions",
                         "Generate LLVM IR conversions", emitBuilders);
