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

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
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

// Variation of method in FormatVariadic.h which takes a StringRef as input
// instead.
template <typename... Ts>
inline auto formatv(StringRef fmt, Ts &&... vals) -> formatv_object<decltype(
    std::make_tuple(detail::build_format_adapter(std::forward<Ts>(vals))...))> {
  using ParamTuple = decltype(
      std::make_tuple(detail::build_format_adapter(std::forward<Ts>(vals))...));
  return llvm::formatv_object<ParamTuple>(
      fmt,
      std::make_tuple(detail::build_format_adapter(std::forward<Ts>(vals))...));
}

// Returns whether the record has a value of the given name that can be returned
// via getValueAsString.
static inline bool hasStringAttribute(const Record &record,
                                      StringRef fieldName) {
  auto valueInit = record.getValueInit(fieldName);
  return isa<CodeInit>(valueInit) || isa<StringInit>(valueInit);
}

// Returns `fieldName`'s value queried from `record` if `fieldName` is set as
// an string in record; otherwise, returns `defaultVal`.
static inline StringRef getAsStringOrDefault(const Record &record,
                                             StringRef fieldName,
                                             StringRef defaultVal) {
  return hasStringAttribute(record, fieldName)
             ? record.getValueAsString(fieldName)
             : defaultVal;
}

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

  // Emit method declaration for the getCanonicalizationPatterns() interface.
  void emitCanonicalizationPatterns();

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

  SmallVector<StringRef, 2> split;
  SplitString(def.getName(), split, "_");
  for (auto it = split.begin(), e = std::prev(split.end()); it != e; ++it) {
    os << "\nnamespace " << *it << "{\n";
  }
  StringRef className = split.back();
  os << "class " << className << " : public Op<" << className;
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
  emitter.emitCanonicalizationPatterns();

  os << "private:\n  friend class ::mlir::Operation;\n";
  os << "  explicit " << className
     << "(const Operation* state) : Op(state) {}\n";
  os << "};\n";
  for (auto it = split.begin(), e = std::prev(split.end()); it != e; ++it) {
    os << "} // end namespace " << *it << "\n";
  }
}

void OpEmitter::getAttributes() {
  const auto &recordKeeper = def.getRecords();
  const auto attrType = recordKeeper.getClass("Attr");
  for (const auto &val : def.getValues()) {
    if (auto *record = dyn_cast<RecordRecTy>(val.getType())) {
      if (record->isSubClassOf(attrType)) {
        if (record->getClasses().size() != 1) {
          PrintFatalError(
              def.getLoc(),
              "unsupported attribute modelling, only single class expected");
        }
        attrs.emplace_back(&val, *record->getClasses().begin());
      }
    }
  }
}

void OpEmitter::emitAttrGetters() {
  const auto &recordKeeper = def.getRecords();
  const auto *derivedAttrType = recordKeeper.getClass("DerivedAttr")->getType();
  for (const auto &pair : attrs) {
    auto &val = *pair.first;
    auto &attr = *pair.second;

    // Emit the derived attribute body.
    if (auto defInit = dyn_cast<DefInit>(val.getValue())) {
      if (defInit->getType()->typeIsA(derivedAttrType)) {
        auto *def = defInit->getDef();
        os << "  " << def->getValueAsString("returnType").trim() << ' '
           << val.getName() << "() const {" << def->getValueAsString("body")
           << " }\n";
        continue;
      }
    }

    // Emit normal emitter.
    if (!hasStringAttribute(attr, "storageType")) {
      // Handle the base case where there is no storage type specified.
      os << "  Attribute " << val.getName()
         << "() const {\n    return getAttr(\"" << val.getName()
         << "\");\n  }\n";
      continue;
    }

    os << "  " << attr.getValueAsString("returnType").trim() << ' '
       << val.getName() << "() const {\n";

    // Return the queried attribute with the correct return type.
    const auto &attrVal = Twine("this->getAttrOfType<") +
                          attr.getValueAsString("storageType").trim() + ">(\"" +
                          val.getName() + "\").getValue()";
    os << "    return "
       << formatv(attr.getValueAsString("convertFromStorage"), attrVal.str())
       << ";\n  }\n";
  }
}

void OpEmitter::emitBuilder() {
  if (hasStringAttribute(def, "builder")) {
    // If a custom builder is given then print that out instead.
    auto builder = def.getValueAsString("builder");
    if (!builder.empty())
      os << builder << '\n';
  }

  // Generate default builders that requires all result type, operands, and
  // attributes as parameters.

  // We generate two builders here, one having a stand-alone parameter for
  // each result type / operand / attribute, the other having an aggregated
  // parameter for all result types / operands / attributes, to facilitate
  // different call patterns.

  // 1. Stand-alone parameters

  std::vector<Record *> returnTypes = def.getValueAsListOfDefs("returnTypes");
  DagInit *operands = def.getValueAsDag("operands");

  os << "  static void build(Builder* builder, OperationState* result";

  // Emit parameters for all return types
  for (unsigned i = 0, e = returnTypes.size(); i != e; ++i)
    os << ", Type returnType" << i;

  // Emit parameters for all operands
  for (unsigned i = 0, e = operands->getNumArgs(); i != e; ++i)
    os << ", SSAValue* arg" << i;

  // Emit parameters for all attributes
  // TODO(antiagainst): Support default initializer for attributes
  for (const auto &pair : attrs) {
    const Record &attr = *pair.second;
    os << ", " << getAsStringOrDefault(attr, "storageType", "Attribute").trim()
       << ' ' << pair.first->getName();
  }

  os << ") {\n";

  // Push all result types to the result
  if (!returnTypes.empty()) {
    os << "    result->addTypes({returnType0";
    for (unsigned i = 1, e = returnTypes.size(); i != e; ++i)
      os << ", returnType" << i;
    os << "});\n\n";
  }

  // Push all operands to the result
  if (operands->getNumArgs() != 0) {
    os << "    result->addOperands({arg0";
    for (unsigned i = 1, e = operands->getNumArgs(); i != e; ++i)
      os << ", arg" << i;
    os << "});\n";
  }

  // Push all attributes to the result
  for (const auto &pair : attrs) {
    StringRef name = pair.first->getName();
    os << "    result->addAttribute(\"" << name << "\", " << name << ");\n";
  }

  os << "  }\n";

  // 2. Aggregated parameters

  // Signature
  os << "  static void build(Builder* builder, OperationState* result, "
     << "ArrayRef<Type> resultTypes, ArrayRef<SSAValue*> args, "
        "ArrayRef<NamedAttribute> attributes) {\n";

  // Result types
  os << "    assert(resultTypes.size() == " << returnTypes.size()
     << "u && \"mismatched number of return types\");\n"
     << "    result->addTypes(resultTypes);\n";

  // Operands
  os << "    assert(args.size() == " << operands->getNumArgs()
     << "u && \"mismatched number of parameters\");\n"
     << "    result->addOperands(args);\n\n";

  // Attributes
  os << "    assert(attributes.size() >= " << attrs.size()
     << "u && \"not enough attributes\");\n"
     << "    for (const auto& pair : attributes)\n"
     << "      result->addAttribute(pair.first, pair.second);\n"
     << "  }\n";
}

void OpEmitter::emitCanonicalizationPatterns() {
  if (!def.getValueAsBit("hasCanonicalizationPatterns"))
    return;
  os << "  static void getCanonicalizationPatterns("
     << "OwningRewritePatternList &results, MLIRContext* context);\n";
}

void OpEmitter::emitParser() {
  if (!hasStringAttribute(def, "parser"))
    return;
  os << "  static bool parse(OpAsmParser *parser, OperationState *result) {"
     << "\n    " << def.getValueAsString("parser") << "\n  }\n";
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
  bool hasCustomVerify = codeInit && !codeInit->getValue().empty();
  if (!hasCustomVerify && attrs.empty())
    return;

  const auto &recordKeeper = def.getRecords();
  const auto *derivedAttrType = recordKeeper.getClass("DerivedAttr")->getType();

  os << "  bool verify() const {\n";
  // Verify the attributes have the correct type.
  for (const auto attr : attrs) {
    // Skip verification for derived attributes.
    if (auto defInit = dyn_cast<DefInit>(attr.first->getValue()))
      if (defInit->getType()->typeIsA(derivedAttrType))
        continue;

    auto name = attr.first->getName();
    if (!hasStringAttribute(*attr.second, "storageType")) {
      os << "    if (!this->getAttr(\"" << name
         << "\")) return emitOpError(\"requires attribute '" << name
         << "'\");\n";
      continue;
    }

    os << "    if (!this->getAttr(\"" << name << "\").dyn_cast_or_null<"
       << attr.second->getValueAsString("storageType").trim()
       << ">()) return emitOpError(\"requires "
       << attr.second->getValueAsString("returnType").trim() << " attribute '"
       << name << "'\");\n";
  }

  if (hasCustomVerify)
    os << "    " << codeInit->getValue() << "\n";
  else
    os << "    return false;\n";
  os << "  }\n";
}

void OpEmitter::emitTraits() {
  std::vector<Record *> returnTypes = def.getValueAsListOfDefs("returnTypes");
  auto operands = def.getValueAsDag("operands");

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

  // Add explicitly added traits. Note that some traits might implicitly defines
  // the number of operands.
  // TODO(jpienaar): Improve Trait specification to make adding them in the
  // tblgen file better.
  bool hasVariadicOperands = false;
  bool hasAtLeastNOperands = false;
  auto *recordVal = def.getValue("traits");
  if (recordVal && recordVal->getValue()) {
    auto traitList = dyn_cast<ListInit>(recordVal->getValue())->getValues();
    for (Init *trait : traitList) {
      auto traitStr = StringRef(trait->getAsUnquotedString()).trim();
      hasVariadicOperands = traitStr.contains("VariadicOperands");
      hasAtLeastNOperands = traitStr.contains("AtLeastNOperands");
      os << ", OpTrait::" << traitStr.trim();
    }
  }

  if ((hasVariadicOperands || hasAtLeastNOperands) &&
      operands->getNumArgs() != 0) {
    PrintFatalError(def.getLoc(),
                    "Operands number definition is not consistent.");
  }

  // Add operand size trait if defined explicitly.
  switch (operands->getNumArgs()) {
  case 0:
    if (!hasVariadicOperands && !hasAtLeastNOperands)
      os << ", OpTrait::ZeroOperands";
    break;
  case 1:
    os << ", OpTrait::OneOperand";
    break;
  default:
    os << ", OpTrait::NOperands<" << operands->getNumArgs() << ">::Impl";
    break;
  }

  // Add op property traits. These match the propoerties specified in the table
  // with the OperationProperty specified in OperationSupport.h.
  for (Record *property : def.getValueAsListOfDefs("properties")) {
    if (property->getName() == "Commutative") {
      os << ", OpTrait::IsCommutative";
    } else if (property->getName() == "NoSideEffect") {
      os << ", OpTrait::HasNoSideEffect";
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

    SmallVector<StringRef, 2> split;
    SplitString(def->getName(), split, "_");
    os << join(split, "::");
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
