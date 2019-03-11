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
#include "mlir/TableGen/OpTrait.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

using mlir::tblgen::Operator;

static const char *const generatedArgName = "_arg";

// Helper macro that returns indented os.
#define OUT(X) os.indent((X))

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

static std::string getArgumentName(const Operator &op, int index) {
  const auto &operand = op.getOperand(index);
  if (!operand.name.empty())
    return operand.name;
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

  // Emit query methods for the named results.
  void emitNamedResults();

  // Emit builder method for the operation.
  void emitBuilder();

  // Emit method declaration for the getCanonicalizationPatterns() interface.
  void emitCanonicalizationPatterns();

  // Emit the folder methods for the operation.
  void emitFolders();

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

  // Emits the build() method that takes each result-type/operand/attribute as
  // a stand-alone parameter. Using the first operand's type as all result
  // types if `isAllSameType` is true.
  void emitStandaloneParamBuilder(bool isAllSameType);

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
  // We only care about namespaces, so drop the class name here
  auto splittedDefName = op.getSplitDefName().drop_back();
  for (auto ns : splittedDefName)
    fn(ns);
}

void OpEmitter::emit(const Record &def, raw_ostream &os) {
  OpEmitter emitter(def, os);

  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "\nnamespace " << ns << "{\n"; });
  os << formatv("class {0} : public Op<{0}", emitter.op.getCppClassName());
  emitter.emitTraits();
  os << "> {\npublic:\n";

  // Build operation name.
  OUT(2) << "static StringRef getOperationName() { return \""
         << emitter.op.getOperationName() << "\"; };\n";

  emitter.emitNamedOperands();
  emitter.emitNamedResults();
  emitter.emitBuilder();
  emitter.emitParser();
  emitter.emitPrinter();
  emitter.emitVerifier();
  emitter.emitAttrGetters();
  emitter.emitCanonicalizationPatterns();
  emitter.emitFolders();

  os << "private:\n  friend class ::mlir::Instruction;\n"
     << "  explicit " << emitter.op.getCppClassName()
     << "(const Instruction* state) : Op(state) {}\n};\n";
  emitter.mapOverClassNamespaces(
      [&os](StringRef ns) { os << "} // end namespace " << ns << "\n"; });
}

void OpEmitter::emitAttrGetters() {
  for (auto &namedAttr : op.getAttributes()) {
    auto name = namedAttr.getName();
    const auto &attr = namedAttr.attr;

    // Determine the name of the attribute getter. The name matches the
    // attribute name excluding dialect prefix.
    StringRef getter = name;
    auto it = getter.split('.');
    if (!it.second.empty())
      getter = it.second;

    // Emit the derived attribute body.
    if (attr.isDerivedAttr()) {
      OUT(2) << attr.getReturnType() << ' ' << getter << "() const {"
             << attr.getDerivedCodeBody() << " }\n";
      continue;
    }

    // Emit normal emitter.
    OUT(2) << attr.getReturnType() << ' ' << getter << "() const {\n";

    // Return the queried attribute with the correct return type.
    std::string attrVal =
        formatv("this->getAttr(\"{1}\").dyn_cast_or_null<{0}>()",
                attr.getStorageType(), name);
    OUT(4) << "auto attr = " << attrVal << ";\n";
    if (attr.hasDefaultValue()) {
      // Returns the default value if not set.
      // TODO: this is inefficient, we are recreating the attribute for every
      // call. This should be set instead.
      OUT(4) << "if (!attr)\n";
      OUT(6) << "return "
             << formatv(
                    attr.getConvertFromStorageCall(),
                    formatv(
                        attr.getDefaultValueTemplate(),
                        "mlir::Builder(this->getInstruction()->getContext())"))
             << ";\n";
    }
    OUT(4) << "return " << formatv(attr.getConvertFromStorageCall(), "attr")
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

  const auto variadicOperandMethods = R"(  SmallVector<Value *, 4> {0}() {
    assert(getInstruction()->getNumOperands() >= {1});
    SmallVector<Value *, 4> operands(
        std::next(getInstruction()->operand_begin(), {1}),
        getInstruction()->operand_end());
    return operands;
  }
  SmallVector<const Value *, 4> {0}() const {
    assert(getInstruction()->getNumOperands() >= {1});
    SmallVector<const Value *, 4> operands(
        std::next(getInstruction()->operand_begin(), {1}),
        getInstruction()->operand_end());
    return operands;
  }
)";

  for (int i = 0, e = op.getNumOperands(); i != e; ++i) {
    const auto &operand = op.getOperand(i);
    if (!operand.name.empty()) {
      if (operand.type.isVariadic()) {
        assert(i == e - 1 && "only the last operand can be variadic");
        os << formatv(variadicOperandMethods, operand.name, i);
      } else {
        os << formatv(operandMethods, operand.name, i);
      }
    }
  }
}

void OpEmitter::emitNamedResults() {
  const auto resultMethods = R"(  Value *{0}() {
    return this->getInstruction()->getResult({1});
  }
  const Value *{0}() const {
    return this->getInstruction()->getResult({1});
  }
)";
  for (int i = 0, e = op.getNumResults(); i != e; ++i) {
    const auto &result = op.getResult(i);
    if (!result.type.isVariadic() && !result.name.empty())
      os << formatv(resultMethods, result.name, i);
  }
}

void OpEmitter::emitStandaloneParamBuilder(bool isAllSameType) {
  OUT(2) << "static void build(Builder *builder, OperationState *result";

  auto numResults = op.getNumResults();

  llvm::SmallVector<std::string, 4> resultNames;
  resultNames.reserve(numResults);

  // Emit parameters for all return types
  if (!isAllSameType) {
    for (unsigned i = 0; i != numResults; ++i) {
      std::string resultName = op.getResultName(i);
      if (resultName.empty())
        resultName = formatv("resultType{0}", i);

      os << (op.getResultType(i).isVariadic() ? ", ArrayRef<Type> " : ", Type ")
         << resultName;

      resultNames.emplace_back(std::move(resultName));
    }
  }

  // Emit parameters for all arguments (operands and attributes).
  int numOperands = 0;
  int numAttrs = 0;
  for (int i = 0, e = op.getNumArgs(); i < e; ++i) {
    auto argument = op.getArg(i);
    if (argument.is<tblgen::Value *>()) {
      auto &operand = op.getOperand(numOperands);
      os << (operand.type.isVariadic() ? ", ArrayRef<Value *> " : ", Value *")
         << getArgumentName(op, numOperands);
      ++numOperands;
    } else {
      // TODO(antiagainst): Support default initializer for attributes
      const auto &namedAttr = op.getAttribute(numAttrs);
      const auto &attr = namedAttr.attr;
      os << ", ";
      if (attr.isOptional())
        os << "/*optional*/";
      os << attr.getStorageType() << ' ' << namedAttr.name;
      ++numAttrs;
    }
  }
  if (numOperands + numAttrs != op.getNumArgs())
    return PrintFatalError(
        "op arguments must be either operands or attributes");

  os << ") {\n";

  // Push all result types to the result
  if (numResults > 0) {
    if (!isAllSameType) {
      bool hasVariadicResult = op.hasVariadicResult();
      int numNonVariadicResults =
          numResults - static_cast<int>(hasVariadicResult);

      if (numNonVariadicResults > 0) {
        OUT(4) << "result->addTypes({" << resultNames.front();
        for (int i = 1; i < numNonVariadicResults; ++i) {
          os << ", " << resultNames[i];
        }
        os << "});\n";
      }

      if (hasVariadicResult) {
        OUT(4) << "result->addTypes(" << resultNames.back() << ");\n";
      }
    } else {
      OUT(4) << "result->addTypes({";
      auto resultType = formatv("{0}->getType()", getArgumentName(op, 0)).str();
      os << resultType;
      for (unsigned i = 1; i != numResults; ++i)
        os << resultType;
      os << "});\n\n";
    }
  }

  // Push all operands to the result
  bool hasVariadicOperand = op.hasVariadicOperand();
  int numNonVariadicOperands =
      numOperands - static_cast<int>(hasVariadicOperand);
  if (numNonVariadicOperands > 0) {
    OUT(4) << "result->addOperands({" << getArgumentName(op, 0);
    for (int i = 1; i < numNonVariadicOperands; ++i) {
      os << ", " << getArgumentName(op, i);
    }
    os << "});\n";
  }
  if (hasVariadicOperand) {
    OUT(4) << "result->addOperands(" << getArgumentName(op, numOperands - 1)
           << ");\n";
  }

  // Push all attributes to the result
  for (const auto &namedAttr : op.getAttributes()) {
    if (!namedAttr.attr.isDerivedAttr()) {
      bool emitNotNullCheck = namedAttr.attr.isOptional();
      if (emitNotNullCheck) {
        OUT(4) << formatv("if ({0}) ", namedAttr.name) << "{\n";
      }
      OUT(4) << formatv("result->addAttribute(\"{0}\", {1});\n",
                        namedAttr.getName(), namedAttr.name);
      if (emitNotNullCheck) {
        OUT(4) << "}\n";
      }
    }
  }
  OUT(2) << "}\n";
}

void OpEmitter::emitBuilder() {
  if (hasStringAttribute(def, "builder")) {
    // If a custom builder is given then print that out.
    auto builder = def.getValueAsString("builder");
    if (!builder.empty())
      os << builder << '\n';
  }

  auto numResults = op.getNumResults();
  bool hasVariadicResult = op.hasVariadicResult();
  int numNonVariadicResults = numResults - int(hasVariadicResult);

  auto numOperands = op.getNumOperands();
  bool hasVariadicOperand = op.hasVariadicOperand();
  int numNonVariadicOperands = numOperands - int(hasVariadicOperand);

  // Generate default builders that requires all result type, operands, and
  // attributes as parameters.

  // We generate three builders here:
  // 1. one having a stand-alone parameter for each result type / operand /
  //    attribute, and
  // 2. one having an aggregated parameter for all result types / operands /
  //    attributes, and
  // 3. one having a stand-alone prameter for each operand and attribute,
  //    use the first operand's type as all result types
  // to facilitate different call patterns.

  // 1. Stand-alone parameters

  emitStandaloneParamBuilder(/*isAllSameType=*/false);

  // 2. Aggregated parameters

  // Signature
  OUT(2) << "static void build(Builder* builder, OperationState* result, "
         << "ArrayRef<Type> resultTypes, ArrayRef<Value*> args, "
            "ArrayRef<NamedAttribute> attributes) {\n";

  // Result types
  OUT(4) << "assert(resultTypes.size()" << (hasVariadicResult ? " >= " : " == ")
         << numNonVariadicResults
         << "u && \"mismatched number of return types\");\n"
         << "    result->addTypes(resultTypes);\n";

  // Operands
  OUT(4) << "assert(args.size()" << (hasVariadicOperand ? " >= " : " == ")
         << numNonVariadicOperands
         << "u && \"mismatched number of parameters\");\n"
         << "    result->addOperands(args);\n\n";

  // Attributes
  if (op.getNumAttributes() > 0) {
    OUT(4) << "assert(!attributes.size() && \"no attributes expected\");\n"
           << "  }\n";
  } else {
    OUT(4) << "assert(attributes.size() >= " << op.getNumAttributes()
           << "u && \"not enough attributes\");\n"
           << "    for (const auto& pair : attributes)\n"
           << "      result->addAttribute(pair.first, pair.second);\n"
           << "  }\n";
  }

  // 3. Deduced result types

  if (!op.hasVariadicResult() && op.hasTrait("SameOperandsAndResultType"))
    emitStandaloneParamBuilder(/*isAllSameType=*/true);
}

void OpEmitter::emitCanonicalizationPatterns() {
  if (!def.getValueAsBit("hasCanonicalizer"))
    return;
  OUT(2) << "static void getCanonicalizationPatterns("
         << "OwningRewritePatternList &results, MLIRContext* context);\n";
}

void OpEmitter::emitFolders() {
  bool hasSingleResult = op.getNumResults() == 1;
  if (def.getValueAsBit("hasConstantFolder")) {
    if (hasSingleResult) {
      os << "  Attribute constantFold(ArrayRef<Attribute> operands,\n"
            "                         MLIRContext *context) const;\n";
    } else {
      os << "  LogicalResult constantFold(ArrayRef<Attribute> operands,\n"
         << "                             SmallVectorImpl<Attribute> &results,"
         << "\n                             MLIRContext *context) const;\n";
    }
  }

  if (def.getValueAsBit("hasFolder")) {
    if (hasSingleResult) {
      os << "  Value *fold();\n";
    } else {
      os << "  bool fold(SmallVectorImpl<Value *> &results);\n";
    }
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
  if (!hasCustomVerify && op.getNumArgs() == 0 && op.getNumResults() == 0)
    return;

  OUT(2) << "bool verify() const {\n";
  // Verify the attributes have the correct type.
  for (const auto &namedAttr : op.getAttributes()) {
    const auto &attr = namedAttr.attr;

    if (attr.isDerivedAttr())
      continue;

    auto name = namedAttr.getName();
    if (!attr.hasStorageType() && !attr.hasDefaultValue()) {
      // TODO: Some verification can be done even without storage type.
      OUT(4) << "if (!this->getAttr(\"" << name
             << "\")) return emitOpError(\"requires attribute '" << name
             << "'\");\n";
      continue;
    }

    bool allowMissingAttr = attr.hasDefaultValue() || attr.isOptional();
    if (allowMissingAttr) {
      // If the attribute has a default value, then only verify the predicate if
      // set. This does effectively assume that the default value is valid.
      // TODO: verify the debug value is valid (perhaps in debug mode only).
      OUT(4) << "if (this->getAttr(\"" << name << "\")) {\n";
    }

    OUT(6) << "if (!this->getAttr(\"" << name << "\").dyn_cast_or_null<"
           << attr.getStorageType() << ">()) return emitOpError(\"requires "
           << attr.getDescription() << " attribute '" << name << "'\");\n";

    auto attrPred = attr.getPredicate();
    if (!attrPred.isNull()) {
      OUT(6) << formatv("if (!({0})) return emitOpError(\"attribute '{1}' "
                        "failed to satisfy {2} attribute constraints\");\n",
                        formatv(attrPred.getCondition(),
                                formatv("this->getAttr(\"{0}\")", name)),
                        name, attr.getDescription());
    }

    if (allowMissingAttr)
      OUT(4) << "}\n";
  }

  // Emits verification code for an operand or result.
  auto verifyValue = [this](const tblgen::Value &value, int index,
                            bool isOperand) -> void {
    // TODO: Handle variadic operand/result verification.
    if (value.type.isVariadic())
      return;

    // TODO: Commonality between matchers could be extracted to have a more
    // concise code.
    if (value.hasPredicate()) {
      auto description = value.type.getDescription();
      OUT(4) << "if (!("
             << formatv(value.type.getConditionTemplate(),
                        "this->getInstruction()->get" +
                            Twine(isOperand ? "Operand" : "Result") + "(" +
                            Twine(index) + ")->getType()")
             << ")) {\n";
      OUT(6) << "return emitOpError(\"" << (isOperand ? "operand" : "result")
             << " #" << index
             << (description.empty() ? " type precondition failed"
                                     : " must be " + Twine(description))
             << "\");";
      OUT(4) << "}\n";
    }
  };

  for (unsigned i = 0, e = op.getNumOperands(); i < e; ++i) {
    verifyValue(op.getOperand(i), i, /*isOperand=*/true);
  }

  for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
    verifyValue(op.getResult(i), i, /*isOperand=*/false);
  }

  for (auto &trait : op.getTraits()) {
    if (auto t = dyn_cast<tblgen::PredOpTrait>(&trait)) {
      OUT(4) << "if (!"
             << formatv(t->getPredTemplate().c_str(),
                        "(*this->getInstruction())")
             << ")\n";
      OUT(6) << "return emitOpError(\"failed to verify that "
             << t->getDescription() << "\");\n";
    }
  }

  if (hasCustomVerify)
    OUT(4) << codeInit->getValue() << "\n";
  else
    OUT(4) << "return false;\n";
  OUT(2) << "}\n";
}

void OpEmitter::emitTraits() {
  auto numResults = op.getNumResults();
  bool hasVariadicResult = op.hasVariadicResult();

  // Add return size trait.
  os << ", OpTrait::";
  if (hasVariadicResult) {
    if (numResults == 1)
      os << "VariadicResults";
    else
      os << "AtLeastNResults<" << (numResults - 1) << ">::Impl";
  } else {
    switch (numResults) {
    case 0:
      os << "ZeroResult";
      break;
    case 1:
      os << "OneResult";
      break;
    default:
      os << "NResults<" << numResults << ">::Impl";
      break;
    }
  }

  for (const auto &trait : op.getTraits()) {
    if (auto opTrait = dyn_cast<tblgen::NativeOpTrait>(&trait))
      os << ", OpTrait::" << opTrait->getTrait();
  }

  // Add variadic size trait and normal op traits.
  auto numOperands = op.getNumOperands();
  bool hasVariadicOperand = op.hasVariadicOperand();

  // Add operand size trait.
  os << ", OpTrait::";
  if (hasVariadicOperand) {
    if (numOperands == 1)
      os << "VariadicOperands";
    else
      os << "AtLeastNOperands<" << (numOperands - 1) << ">::Impl";
  } else {
    switch (numOperands) {
    case 0:
      os << "ZeroOperands";
      break;
    case 1:
      os << "OneOperand";
      break;
    default:
      os << "NOperands<" << numOperands << ">::Impl";
      break;
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
    os << Operator(def).getQualCppClassName();
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

static mlir::GenRegistration
    genOpDefinitions("gen-op-definitions", "Generate op definitions",
                     [](const RecordKeeper &records, raw_ostream &os) {
                       emitOpDefinitions(records, os);
                       return false;
                     });
