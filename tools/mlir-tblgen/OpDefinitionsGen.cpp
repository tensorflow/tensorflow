//===- OpDefinitionsGen.cpp - MLIR op definitions generator ---------------===//
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
// OpDefinitionsGen uses the description of operations to generate C++
// definitions for ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;

static const char *const generatedArgName = "_arg";

// TODO(jpienaar): The builder body should probably be separate from the header.

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

  // Emit query methods for the named operands.
  void emitNamedOperands();

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

private:
  OpEmitter(const Record &def, raw_ostream &os);

  // Populates the operands and attributes.
  void getOperandsAndAttributes();

  // Returns the class name of the op.
  StringRef cppClassName() const;

  // Invokes the given function over all the namespaces of the class.
  void mapOverClassNamespaces(function_ref<void(StringRef)> fn) const;

  // Returns the operation name.
  StringRef getOperationName() const;

  // The record corresponding to the op.
  const Record &def;

  const RecordKeeper &recordKeeper;

  // Record of Attr class.
  Record *attrClass;

  // Type of DerivedAttr.
  const RecordRecTy *derivedAttrType;

  // The name of the op split around '_'.
  SmallVector<StringRef, 2> splittedDefName;

  // The operands of the op.
  SmallVector<std::pair<std::string, const DefInit *>, 4> operands;

  // The attributes of the op.
  SmallVector<std::pair<std::string, const DefInit *>, 4> attrs;
  SmallVector<std::pair<const RecordVal *, const Record *>, 4> derivedAttrs;

  raw_ostream &os;
};
} // end anonymous namespace

OpEmitter::OpEmitter(const Record &def, raw_ostream &os)
    : def(def), recordKeeper(def.getRecords()),
      attrClass(recordKeeper.getClass("Attr")),
      derivedAttrType(recordKeeper.getClass("DerivedAttr")->getType()), os(os) {
  SplitString(def.getName(), splittedDefName, "_");
  getOperandsAndAttributes();
}

StringRef OpEmitter::cppClassName() const { return splittedDefName.back(); }

StringRef OpEmitter::getOperationName() const {
  return def.getValueAsString("opName");
}

void OpEmitter::mapOverClassNamespaces(function_ref<void(StringRef)> fn) const {
  for (auto it = splittedDefName.begin(), e = std::prev(splittedDefName.end());
       it != e; ++it)
    fn(*it);
}

void OpEmitter::getOperandsAndAttributes() {
  DagInit *argumentValues = def.getValueAsDag("arguments");
  for (unsigned i = 0, e = argumentValues->getNumArgs(); i != e; ++i) {
    auto arg = argumentValues->getArg(i);
    auto givenName = argumentValues->getArgName(i);
    DefInit *argDef = dyn_cast<DefInit>(arg);
    if (!argDef)
      PrintFatalError(def.getLoc(),
                      "unexpected type for " + Twine(i) + "th argument");

    // Handle attribute.
    if (argDef->getDef()->isSubClassOf(attrClass)) {
      if (!givenName)
        PrintFatalError(argDef->getDef()->getLoc(), "attributes must be named");
      attrs.emplace_back(givenName->getValue(), argDef);
      continue;
    }

    // Handle operands.
    std::string name;
    if (givenName)
      name = givenName->getValue();
    else
      name = formatv("{0}_{1}", generatedArgName, i);
    operands.emplace_back(name, argDef);
  }

  // Derived attributes.
  for (const auto &val : def.getValues()) {
    if (auto *record = dyn_cast<RecordRecTy>(val.getType())) {
      if (record->typeIsA(derivedAttrType)) {
        if (record->getClasses().size() != 1) {
          PrintFatalError(
              def.getLoc(),
              "unsupported attribute modelling, only single class expected");
        }
        derivedAttrs.emplace_back(&val, *record->getClasses().begin());
        continue;
      }
      if (record->isSubClassOf(attrClass))
        PrintFatalError(def.getLoc(),
                        "unexpected Attr where only DerivedAttr is allowed");
    }
  }
}

void OpEmitter::emit(const Record &def, raw_ostream &os) {
  OpEmitter emitter(def, os);

  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "\nnamespace " << ns << "{\n"; });
  os << "class " << emitter.cppClassName() << " : public Op<"
     << emitter.cppClassName();
  emitter.emitTraits();
  os << "> {\npublic:\n";

  // Build operation name.
  os << "  static StringRef getOperationName() { return \""
     << emitter.getOperationName() << "\"; };\n";

  emitter.emitNamedOperands();
  emitter.emitBuilder();
  emitter.emitParser();
  emitter.emitPrinter();
  emitter.emitVerifier();
  emitter.emitAttrGetters();
  emitter.emitCanonicalizationPatterns();

  os << "private:\n  friend class ::mlir::OperationInst;\n";
  os << "  explicit " << emitter.cppClassName()
     << "(const OperationInst* state) : Op(state) {}\n";
  os << "};\n";
  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "} // end namespace " << ns << "\n"; });
}

void OpEmitter::emitAttrGetters() {
  for (const auto &pair : derivedAttrs) {
    auto &val = *pair.first;

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
  }

  for (const auto &pair : attrs) {
    auto &name = pair.first;
    auto &attr = *pair.second->getDef();
    // Emit normal emitter.
    if (!hasStringAttribute(attr, "storageType")) {
      // Handle the base case where there is no storage type specified.
      os << "  Attribute " << name << "() const {\n    return getAttr(\""
         << name << "\");\n  }\n";
      continue;
    }

    os << "  " << attr.getValueAsString("returnType").trim() << ' ' << name
       << "() const {\n";

    // Return the queried attribute with the correct return type.
    std::string attrVal =
        formatv("this->getAttrOfType<{0}>(\"{1}\")",
                attr.getValueAsString("storageType").trim(), name);
    os << "    return "
       << formatv(attr.getValueAsString("convertFromStorage"), attrVal)
       << ";\n  }\n";
  }
}

void OpEmitter::emitNamedOperands() {
  const auto operandMethods = R"(  Value *{0}() {
    return this->getInstruction()->getOperand({1});
  }
  const Value *{0}() const {
    return this->getInstruction()->getOperand({1});
  }
)";
  for (int i = 0, e = operands.size(); i != e; ++i) {
    const auto &op = operands[i];
    if (!StringRef(op.first).startswith(generatedArgName))
      os << formatv(operandMethods, op.first, i);
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
  os << "  static void build(Builder* builder, OperationState* result";

  // Emit parameters for all return types
  for (unsigned i = 0, e = returnTypes.size(); i != e; ++i)
    os << ", Type returnType" << i;

  // Emit parameters for all operands
  for (const auto &pair : operands)
    os << ", Value* " << pair.first;

  // Emit parameters for all attributes
  // TODO(antiagainst): Support default initializer for attributes
  for (const auto &pair : attrs) {
    const Record &attr = *pair.second->getDef();
    os << ", " << getAsStringOrDefault(attr, "storageType", "Attribute").trim()
       << ' ' << pair.first;
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
  if (!operands.empty()) {
    os << "    result->addOperands({" << operands.front().first;
    for (auto it = operands.begin() + 1, e = operands.end(); it != e; ++it)
      os << ", " << it->first;
    os << "});\n";
  }

  // Push all attributes to the result
  for (const auto &pair : attrs) {
    StringRef name = pair.first;
    os << "    result->addAttribute(\"" << name << "\", " << name << ");\n";
  }

  os << "  }\n";

  // 2. Aggregated parameters

  // Signature
  os << "  static void build(Builder* builder, OperationState* result, "
     << "ArrayRef<Type> resultTypes, ArrayRef<Value*> args, "
        "ArrayRef<NamedAttribute> attributes) {\n";

  // Result types
  os << "    assert(resultTypes.size() == " << returnTypes.size()
     << "u && \"mismatched number of return types\");\n"
     << "    result->addTypes(resultTypes);\n";

  // Operands
  os << "    assert(args.size() == " << operands.size()
     << "u && \"mismatched number of parameters\");\n"
     << "    result->addOperands(args);\n\n";

  // Attributes
  if (attrs.empty()) {
    os << "    assert(!attributes.size() && \"no attributes expected\");\n"
       << "  }\n";
  } else {
    os << "    assert(attributes.size() >= " << attrs.size()
       << "u && \"not enough attributes\");\n"
       << "    for (const auto& pair : attributes)\n"
       << "      result->addAttribute(pair.first, pair.second);\n"
       << "  }\n";
  }
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

  os << "  bool verify() const {\n";
  // Verify the attributes have the correct type.
  for (const auto attr : attrs) {
    auto name = attr.first;
    if (!hasStringAttribute(*attr.second->getDef(), "storageType")) {
      os << "    if (!this->getAttr(\"" << name
         << "\")) return emitOpError(\"requires attribute '" << name
         << "'\");\n";
      continue;
    }

    os << "    if (!this->getAttr(\"" << name << "\").dyn_cast_or_null<"
       << attr.second->getDef()->getValueAsString("storageType").trim()
       << ">()) return emitOpError(\"requires "
       << attr.second->getDef()->getValueAsString("returnType").trim()
       << " attribute '" << name << "'\");\n";
  }

  if (hasCustomVerify)
    os << "    " << codeInit->getValue() << "\n";
  else
    os << "    return false;\n";
  os << "  }\n";
}

void OpEmitter::emitTraits() {
  std::vector<Record *> returnTypes = def.getValueAsListOfDefs("returnTypes");

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
      std::string traitStr = trait->getAsUnquotedString();
      auto ref = StringRef(traitStr).trim();
      hasVariadicOperands = ref == "VariadicOperands";
      hasAtLeastNOperands = ref == "AtLeastNOperands";
      os << ", OpTrait::" << ref;
    }
  }

  if ((hasVariadicOperands || hasAtLeastNOperands) && !operands.empty()) {
    PrintFatalError(def.getLoc(),
                    "Operands number definition is not consistent.");
  }

  // Add operand size trait if defined explicitly.
  switch (operands.size()) {
  case 0:
    if (!hasVariadicOperands && !hasAtLeastNOperands)
      os << ", OpTrait::ZeroOperands";
    break;
  case 1:
    os << ", OpTrait::OneOperand";
    break;
  default:
    os << ", OpTrait::NOperands<" << operands.size() << ">::Impl";
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
  os << "enum class OpCode {\n";
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

    SmallVector<StringRef, 2> splittedDefName;
    SplitString(def->getName(), splittedDefName, "_");
    os << join(splittedDefName, "::");
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
       << def->getValueAsString("opName") << "\")\n";
  }
  os << "#undef ALL_OPS";
}

mlir::GenRegistration
    genOpDefinitions("gen-op-definitions", "Generate op definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       emitOpDefinitions(records, os);
                       return false;
                     });
