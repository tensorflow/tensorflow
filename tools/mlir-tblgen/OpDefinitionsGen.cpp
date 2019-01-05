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
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

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

static std::string getAttributeName(const Operator::Attribute &attr) {
  return attr.name->getAsUnquotedString();
}

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (operand.name)
    return operand.name->getAsUnquotedString();
  else
    return formatv("{0}_{1}", generatedArgName, index);
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

  // Emit the constant folder method for the operation.
  void emitConstantFolder();

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

  // Invokes the given function over all the namespaces of the class.
  void mapOverClassNamespaces(function_ref<void(StringRef)> fn);

  // The record corresponding to the op.
  const Record &def;

  // The operator being emitted.
  Operator op;

  raw_ostream &os;
};
} // end anonymous namespace

OpEmitter::OpEmitter(const Record &def, raw_ostream &os)
    : def(def), op(def), os(os) {}

void OpEmitter::mapOverClassNamespaces(function_ref<void(StringRef)> fn) {
  auto &splittedDefName = op.getSplitDefName();
  for (auto it = splittedDefName.begin(), e = std::prev(splittedDefName.end());
       it != e; ++it)
    fn(*it);
}

void OpEmitter::emit(const Record &def, raw_ostream &os) {
  OpEmitter emitter(def, os);

  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "\nnamespace " << ns << "{\n"; });
  os << formatv("class {0} : public Op<{0}", emitter.op.cppClassName());
  emitter.emitTraits();
  os << "> {\npublic:\n";

  // Build operation name.
  os << "  static StringRef getOperationName() { return \""
     << emitter.op.getOperationName() << "\"; };\n";

  emitter.emitNamedOperands();
  emitter.emitBuilder();
  emitter.emitParser();
  emitter.emitPrinter();
  emitter.emitVerifier();
  emitter.emitAttrGetters();
  emitter.emitCanonicalizationPatterns();
  emitter.emitConstantFolder();

  os << "private:\n  friend class ::mlir::OperationInst;\n";
  os << "  explicit " << emitter.op.cppClassName()
     << "(const OperationInst* state) : Op(state) {}\n};\n";
  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "} // end namespace " << ns << "\n"; });
}

void OpEmitter::emitAttrGetters() {
  for (auto &attr : op.getAttributes()) {
    auto name = getAttributeName(attr);
    auto *def = attr.record;

    // Emit the derived attribute body.
    if (attr.isDerived) {
      os << "  " << def->getValueAsString("returnType").trim() << ' ' << name
         << "() const {" << def->getValueAsString("body") << " }\n";
      continue;
    }

    // Emit normal emitter.
    os << "  " << def->getValueAsString("returnType").trim() << ' ' << name
       << "() const {\n";

    // Return the queried attribute with the correct return type.
    std::string attrVal =
        formatv("this->getAttrOfType<{0}>(\"{1}\")",
                def->getValueAsString("storageType").trim(), name);
    os << "    return "
       << formatv(def->getValueAsString("convertFromStorage"), attrVal)
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
  for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
    const auto &operand = op.getOperand(i);
    if (operand.name)
      os << formatv(operandMethods, operand.name->getAsUnquotedString(), i);
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
  for (int i = 0, e = op.getNumOperands(); i != e; ++i)
    os << ", Value* " << getArgumentName(op, i);

  // Emit parameters for all attributes
  // TODO(antiagainst): Support default initializer for attributes
  for (const auto &attr : op.getAttributes()) {
    if (attr.isDerived)
      break;
    const Record &def = *attr.record;
    os << ", " << getAsStringOrDefault(def, "storageType", "Attribute").trim()
       << ' ' << getAttributeName(attr);
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
  if (op.getNumOperands() > 0) {
    os << "    result->addOperands({" << getArgumentName(op, 0);
    for (int i = 1, e = op.getNumOperands(); i != e; ++i)
      os << ", " << getArgumentName(op, i);
    os << "});\n";
  }

  // Push all attributes to the result
  for (const auto &attr : op.getAttributes())
    if (!attr.isDerived)
      os.indent(4) << formatv("result->addAttribute(\"{0}\", {0});\n",
                              getAttributeName(attr));
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
  os << "    assert(args.size() == " << op.getNumOperands()
     << "u && \"mismatched number of parameters\");\n"
     << "    result->addOperands(args);\n\n";

  // Attributes
  if (op.getNumAttributes() > 0) {
    os << "    assert(!attributes.size() && \"no attributes expected\");\n"
       << "  }\n";
  } else {
    os << "    assert(attributes.size() >= " << op.getNumAttributes()
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

void OpEmitter::emitConstantFolder() {
  if (!def.getValueAsBit("hasConstantFolder"))
    return;
  if (def.getValueAsListOfDefs("returnTypes").size() == 1) {
    os << "  Attribute constantFold(ArrayRef<Attribute> operands,\n"
          "                         MLIRContext *context) const;\n";
  } else {
    os << "  bool constantFold(ArrayRef<Attribute> operands,\n"
       << "                    SmallVectorImpl<Attribute> &results,\n"
       << "                    MLIRContext *context) const;\n";
  }
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
  if (!hasCustomVerify && op.getNumArgs() == 0)
    return;

  os << "  bool verify() const {\n";
  // Verify the attributes have the correct type.
  for (const auto &attr : op.getAttributes()) {
    if (attr.isDerived)
      continue;

    auto name = getAttributeName(attr);
    if (!hasStringAttribute(*attr.record, "storageType")) {
      os << "    if (!this->getAttr(\"" << name
         << "\")) return emitOpError(\"requires attribute '" << name
         << "'\");\n";
      continue;
    }

    os << "    if (!this->getAttr(\"" << name << "\").dyn_cast_or_null<"
       << attr.record->getValueAsString("storageType").trim()
       << ">()) return emitOpError(\"requires "
       << attr.record->getValueAsString("returnType").trim() << " attribute '"
       << name << "'\");\n";
  }

  // TODO: Handle variadic.
  int opIndex = 0;
  for (const auto &operand : op.getOperands()) {
    // TODO: Commonality between matchers could be extracted to have a more
    // concise code.
    if (operand.hasMatcher()) {
      auto pred =
          "if (!(" + operand.createTypeMatcherTemplate() + ")) return false;\n";
      os.indent(4) << formatv(pred, "this->getInstruction()->getOperand(" +
                                        Twine(opIndex) + ")->getType()");
    }
    ++opIndex;
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

  if ((hasVariadicOperands || hasAtLeastNOperands) && op.getNumOperands() > 0) {
    PrintFatalError(def.getLoc(),
                    "Operands number definition is not consistent.");
  }

  // Add operand size trait if defined explicitly.
  switch (op.getNumOperands()) {
  case 0:
    if (!hasVariadicOperands && !hasAtLeastNOperands)
      os << ", OpTrait::ZeroOperands";
    break;
  case 1:
    os << ", OpTrait::OneOperand";
    break;
  default:
    os << ", OpTrait::NOperands<" << op.getNumOperands() << ">::Impl";
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
static void emitOpClasses(const std::vector<Record *> &defs, raw_ostream &os) {
  IfDefScope scope("GET_OP_CLASSES", os);
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
    os << Operator(def).qualifiedCppClassName();
    first = false;
  }
}

static void emitOpDefinitions(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  emitSourceFileHeader("List of ops", os);

  const auto &defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpList(defs, os);
  emitOpClasses(defs, os);
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
