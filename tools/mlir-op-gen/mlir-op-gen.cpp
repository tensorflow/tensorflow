//===- mlir-op-gen.cpp - MLIR op generator --------------------------------===//
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
// This is a command line utility that generates C++ definitions for ops
// declared in a op database.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

enum ActionType { PrintRecords, GenDefFile, GenOpDefinitions };

static cl::opt<ActionType> action(
    cl::desc("Action to perform:"),
    cl::values(clEnumValN(PrintRecords, "print-records",
                          "Print all records to stdout (default)"),
               clEnumValN(GenDefFile, "gen-def-file", "Generate def file"),
               clEnumValN(GenOpDefinitions, "gen-op-definitions",
                          "Generate op definitions")));
static cl::opt<std::string> opcodeClass("opcode-enum",
                                        cl::desc("The opcode enum to use"));

// TODO(jpienaar): The builder body should probably be separate from the header.

using AttrPair = std::pair<const RecordVal *, const Record *>;
using AttrVector = SmallVectorImpl<AttrPair>;

namespace {
// Simple RAII helper for defining ifdef-undef-endif scopes.
class IfDefScope {
public:
  IfDefScope(StringRef name, raw_ostream &os) : name(name), os(os) {
    os << "#ifdef " << name << "\n"
       << "#undef " << name << "\n";
  }

  ~IfDefScope() { os << "\n#endif  // " << name << "\n\n"; }

private:
  StringRef name;
  raw_ostream &os;
};
} // end anonymous namespace

namespace {
// Helper class to emit a record into the given output stream.
class OpEmitter {
public:
  static void emit(const Record &def, raw_ostream &os);

  // Emit getters for the attributes of the operation.
  void emitAttrGetters();

  // Emit builder method for the operation.
  void emitBuilder();

  // Emit the parser for the operation.
  void emitParser();

  // Emit the printer for the operation.
  void emitPrinter();

  // Emit verify method for the operation.
  void emitVerifier();

  // Emit the traits used by the object.
  void emitTraits();

  // Populate the attributes of the op from its definition.
  void getAttributes();

private:
  OpEmitter(const Record &def, raw_ostream &os) : def(def), os(os){};

  SmallVector<AttrPair, 4> attrs;
  const Record &def;
  raw_ostream &os;
};
} // end anonymous namespace

void OpEmitter::emit(const Record &def, raw_ostream &os) {
  OpEmitter emitter(def, os);

  // Query the returned type and operands types of the op.
  emitter.getAttributes();

  // Query the C++ class from which this op should derived.
  StringRef baseClass = def.getValueAsString("baseClass");
  os << "\nclass " << def.getName() << " : public " << baseClass << "<"
     << def.getName();
  emitter.emitTraits();
  os << "> {\npublic:\n";

  // Build operation name.
  os << "  static StringRef getOperationName() { return \""
     << def.getValueAsString("name") << "\"; };\n";

  emitter.emitBuilder();
  emitter.emitParser();
  emitter.emitPrinter();
  emitter.emitVerifier();
  emitter.emitAttrGetters();

  os << "private:\n  friend class ::mlir::Operation;\n";
  os << "  explicit " << def.getName()
     << "(const Operation* state) : " << baseClass << "(state) {}\n";
  os << "};\n";
}

void OpEmitter::getAttributes() {
  const auto &recordKeeper = def.getRecords();
  const auto attrType = recordKeeper.getClass("Attr");
  for (const auto &val : def.getValues()) {
    if (DefInit *defInit = dyn_cast<DefInit>(val.getValue())) {
      auto attr = defInit->getDef();
      if (attr->isSubClassOf(attrType))
        attrs.emplace_back(&val, attr);
    }
  }
}

// TODO(jpienaar): Improve Attr specification to make adding them in the tblgen
// file better.
void OpEmitter::emitAttrGetters() {
  for (const auto &pair : attrs) {
    auto &val = *pair.first;
    auto &attr = *pair.second;
    auto name = attr.getValueAsString("name");
    os << "  " << attr.getValueAsString("PrimitiveType").trim() << " get"
       << val.getName() << "() const {\n";
    os << "    return this->getAttrOfType<"
       << attr.getValueAsString("AttrType").trim() << ">(\"" << name
       << "\")->getValue();\n  }\n";
  }
}

void OpEmitter::emitBuilder() {
  // If a custom builder is given then print that out instead.
  auto builder = def.getValueAsString("builder");
  if (!builder.empty()) {
    os << builder << '\n';
    return;
  }

  os << "  static void build(Builder *builder, OperationState *result";
  const std::vector<Record *> &operandTypes =
      def.getValueAsListOfDefs("operandTypes");

  // Label the operands as simply arg_i.
  for (int i = 0, e = operandTypes.size(); i != e; ++i)
    os << ", SSAValue *arg_" << i;

  // Add a parameter for every attribute.
  for (const auto &attr : attrs) {
    os << ", " << attr.second->getValueAsString("PrimitiveType") << " "
       << attr.second->getValueAsString("name");
  }
  os << ") {\n";

  // Build the OperationState.

  // Add the operands.
  if (!operandTypes.empty()) {
    os << "    result->addOperands({";
    for (int i = 0, e = operandTypes.size(); i != e; ++i) {
      if (i)
        os << ", ";
      os << "arg_" << i;
    }
    os << "});\n";
  }

  // Set the return type.
  // TODO(jpienaar): Perform type propagation here.
  if (isa<DefInit>(def.getValueInit("returnType"))) {
    os << "    result->types.push_back(" << def.getName()
       << "::ReturnType(*builder));\n";
  }

  // Add any attributes.
  for (const auto &attr : attrs) {
    os << "    result->addAttribute(\"" << attr.second->getValueAsString("name")
       << "\", builder->get" << attr.second->getValueAsString("AttrType") << "("
       << attr.second->getValueAsString("name") << "));\n";
  }

  os << "  }\n";
}

void OpEmitter::emitParser() {
  auto valueInit = def.getValueInit("parser");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto parser = codeInit->getValue();
  os << "  static bool parse(OpAsmParser *parser, OperationState *result) {"
     << "\n    " << parser << "\n  }\n";
}

void OpEmitter::emitPrinter() {
  auto valueInit = def.getValueInit("printer");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto printer = codeInit->getValue();
  os << "  void print(OpAsmPrinter *p) const {\n"
     << "    " << printer << "\n  }\n";
}

void OpEmitter::emitVerifier() {
  auto valueInit = def.getValueInit("verifier");
  CodeInit *codeInit = dyn_cast<CodeInit>(valueInit);
  if (!codeInit)
    return;

  auto verifier = codeInit->getValue();
  os << "  bool verify() const {\n";

  // Verify the attributes have the correct type.
  for (const auto attr : attrs) {
    auto name = attr.second->getValueAsString("name");
    os << "     if (!dyn_cast_or_null<"
       << attr.second->getValueAsString("AttrType") << ">(this->getAttr(\""
       << name << "\"))) return emitOpError(\"requires "
       << attr.second->getValueAsString("PrimitiveType") << " attribute '"
       << name << "'\");\n";
  }

  if (verifier.empty())
    os << "    return false;\n";
  else
    os << "    " << verifier << "\n";
  os << "  }\n";
}

void OpEmitter::emitTraits() {
  std::vector<Record *> returnTypes = def.getValueAsListOfDefs("returnTypes");
  std::vector<Record *> operandTypes = def.getValueAsListOfDefs("operandTypes");

  // Add return size trait.
  switch (returnTypes.size()) {
  case 0:
    os << ", OpTrait::ZeroResult";
    break;
  case 1:
    os << ", OpTrait::OneResult";
    break;
  default:
    os << ", OpTrait::NResults<" << returnTypes.size() << ">::Impl";
    break;
  }

  // Add operand size trait.
  switch (operandTypes.size()) {
  case 0:
    os << ", OpTrait::ZeroOperands";
    break;
  case 1:
    os << ", OpTrait::OneOperand";
    break;
  default:
    os << ", OpTrait::NOperands<" << operandTypes.size() << ">::Impl";
    break;
  }

  // Add op property traits.
  if (def.getValueAsBit("isCommutative"))
    os << ", OpTrait::IsCommutative";
  if (def.getValueAsBit("hasNoSideEffect"))
    os << ", OpTrait::HasNoSideEffect";

  // Add explicitly added traits.
  // TODO(jpienaar): Improve Trait specification to make adding them in the
  // tblgen file better.
  const auto traitType = def.getRecords().getClass("Trait");
  for (const auto &val : def.getValues()) {
    if (DefInit *defInit = dyn_cast<DefInit>(val.getValue())) {
      auto attr = defInit->getDef();
      if (attr->isSubClassOf(traitType)) {
        os << ", OpTrait::" << attr->getValueAsString("trait").trim();
      }
    }
  }
}

// Emits the opcode enum and op classes.
static void emitOpClasses(const RecordKeeper &recordKeeper,
                          const std::vector<Record *> &defs, raw_ostream &os) {
  IfDefScope scope("GET_OP_CLASSES", os);

  // Enumeration of all the ops defined.
  os << "enum " << opcodeClass << " {\n";
  for (int i = 0, e = defs.size(); i != e; ++i) {
    auto &def = defs[i];
    os << (i != 0 ? "," : "") << "k" << def->getName();
  }
  os << "\n};\n";

  for (auto *def : defs)
    OpEmitter::emit(*def, os);
}

// Emits a comma-separated list of the ops.
static void emitOpList(const std::vector<Record *> &defs, raw_ostream &os) {
  IfDefScope scope("GET_OP_LIST", os);
  bool first = true;

  for (auto &def : defs) {
    if (!first)
      os << ",";
    os << def->getName();
    first = false;
  }
}

static void emitOpDefinitions(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  emitSourceFileHeader("List of ops", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpList(defs, os);
  emitOpClasses(recordKeeper, defs, os);
}

static void emitOpDefFile(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Op def file", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  os << "#ifndef ALL_OPS\n#define ALL_OPS(OP, NAME)\n#endif\n";
  for (const auto *def : defs) {
    os << "ALL_OPS(" << def->getName() << ", \""
       << def->getValueAsString("name") << "\")\n";
  }
  os << "#undef ALL_OPS";
}

static bool MlirOpTableGenMain(raw_ostream &os, RecordKeeper &records) {
  switch (action) {
  case PrintRecords:
    os << records;
    return false;
  case GenOpDefinitions:
    emitOpDefinitions(records, os);
    return false;
  case GenDefFile:
    emitOpDefFile(records, os);
    return false;
  }
}

int main(int argc, char **argv) {
  sys::PrintStackTraceOnErrorSignal(argv[0]);
  PrettyStackTraceProgram X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  llvm_shutdown_obj Y;
  return TableGenMain(argv[0], &MlirOpTableGenMain);
}
