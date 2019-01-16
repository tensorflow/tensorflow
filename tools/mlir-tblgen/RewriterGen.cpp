//===- RewriterGen.cpp - MLIR pattern rewriter generator ------------===//
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
// RewriterGen uses pattern rewrite definitions to generate rewriter matchers.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;

using mlir::tblgen::Attribute;
using mlir::tblgen::Operator;
using mlir::tblgen::Type;

namespace {

// Wrapper around DAG argument.
struct DagArg {
  DagArg(mlir::tblgen::Operator::Argument arg, Init *constraintInit)
      : arg(arg), constraintInit(constraintInit) {}
  bool isAttr();

  mlir::tblgen::Operator::Argument arg;
  Init *constraintInit;
};

} // end namespace

bool DagArg::isAttr() { return arg.is<Operator::NamedAttribute *>(); }

namespace {
class Pattern {
public:
  static void emit(StringRef rewriteName, Record *p, raw_ostream &os);

private:
  Pattern(Record *pattern, raw_ostream &os) : pattern(pattern), os(os) {}

  // Emit the rewrite pattern named `rewriteName`.
  void emit(StringRef rewriteName);

  // Emit the matcher.
  void emitMatcher(DagInit *tree);

  // Emits the value of constant attribute to `os`.
  void emitAttributeValue(Record *constAttr);

  // Collect bound arguments.
  void collectBoundArguments(DagInit *tree);

  // Helper function to match patterns.
  void matchOp(DagInit *tree, int depth);

  // Returns the Operator stored for the given record.
  Operator &getOperator(const llvm::Record *record);

  // Map from bound argument name to DagArg.
  StringMap<DagArg> boundArguments;

  // Map from Record* to Operator.
  DenseMap<const llvm::Record *, Operator> opMap;

  // Number of the operations in the input pattern.
  int numberOfOpsMatched = 0;

  Record *pattern;
  raw_ostream &os;
};
} // end namespace

// Returns the Operator stored for the given record.
auto Pattern::getOperator(const llvm::Record *record) -> Operator & {
  return opMap.try_emplace(record, record).first->second;
}

void Pattern::emitAttributeValue(Record *constAttr) {
  Attribute attr(constAttr->getValueAsDef("attr"));
  auto value = constAttr->getValue("value");

  if (!attr.isConstBuildable())
    PrintFatalError(pattern->getLoc(),
                    "Attribute " + attr.getTableGenDefName() +
                        " does not have the 'constBuilderCall' field");

  // TODO(jpienaar): Verify the constants here
  os << formatv(attr.getConstBuilderTemplate().str().c_str(), "rewriter",
                value->getValue()->getAsUnquotedString());
}

void Pattern::collectBoundArguments(DagInit *tree) {
  ++numberOfOpsMatched;
  Operator &op = getOperator(cast<DefInit>(tree->getOperator())->getDef());
  // TODO(jpienaar): Expand to multiple matches.
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto arg = tree->getArg(i);
    if (auto argTree = dyn_cast<DagInit>(arg)) {
      collectBoundArguments(argTree);
      continue;
    }
    auto name = tree->getArgNameStr(i);
    if (name.empty())
      continue;
    boundArguments.try_emplace(name, op.getArg(i), arg);
  }
}

// Helper function to match patterns.
void Pattern::matchOp(DagInit *tree, int depth) {
  Operator &op = getOperator(cast<DefInit>(tree->getOperator())->getDef());
  int indent = 4 + 2 * depth;
  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0) {
    // Skip if there is no defining instruction (e.g., arguments to function).
    os.indent(indent) << formatv("if (!op{0}) return matchFailure();\n", depth);
    os.indent(indent) << formatv(
        "if (!op{0}->isa<{1}>()) return matchFailure();\n", depth,
        op.qualifiedCppClassName());
  }
  if (tree->getNumArgs() != op.getNumArgs())
    PrintFatalError(pattern->getLoc(),
                    Twine("mismatch in number of arguments to op '") +
                        op.getOperationName() +
                        "' in pattern and op's definition");
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto arg = tree->getArg(i);
    auto opArg = op.getArg(i);

    if (auto argTree = dyn_cast<DagInit>(arg)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2) << formatv(
          "auto op{0} = op{1}->getOperand({2})->getDefiningInst();\n",
          depth + 1, depth, i);
      matchOp(argTree, depth + 1);
      os.indent(indent) << "}\n";
      continue;
    }

    // Verify arguments.
    if (auto defInit = dyn_cast<DefInit>(arg)) {
      // Verify operands.
      if (auto *operand = opArg.dyn_cast<Operator::Operand *>()) {
        // Skip verification where not needed due to definition of op.
        if (operand->defInit == defInit)
          goto StateCapture;

        if (!defInit->getDef()->isSubClassOf("Type"))
          PrintFatalError(pattern->getLoc(),
                          "type argument required for operand");

        auto constraint = tblgen::TypeConstraint(*defInit);
        os.indent(indent)
            << "if (!("
            << formatv(constraint.getConditionTemplate().str().c_str(),
                       formatv("op{0}->getOperand({1})->getType()", depth, i))
            << ")) return matchFailure();\n";
      }

      // TODO(jpienaar): Verify attributes.
      if (auto *namedAttr = opArg.dyn_cast<Operator::NamedAttribute *>()) {
        // TODO(jpienaar): move to helper class.
        if (defInit->getDef()->isSubClassOf("mAttr")) {
          auto pred =
              tblgen::Pred(defInit->getDef()->getValueInit("predicate"));
          os.indent(indent)
              << "if (!("
              << formatv(pred.getCondition().str().c_str(),
                         formatv("op{0}->getAttrOfType<{1}>(\"{2}\")", depth,
                                 namedAttr->attr.getStorageType(),
                                 namedAttr->getName()))
              << ")) return matchFailure();\n";
        }
      }
    }

  StateCapture:
    auto name = tree->getArgNameStr(i);
    if (name.empty())
      continue;
    if (opArg.is<Operator::Operand *>())
      os.indent(indent) << "state->" << name << " = op" << depth
                        << "->getOperand(" << i << ");\n";
    if (auto namedAttr = opArg.dyn_cast<Operator::NamedAttribute *>()) {
      os.indent(indent) << "state->" << name << " = op" << depth
                        << "->getAttrOfType<"
                        << namedAttr->attr.getStorageType() << ">(\""
                        << namedAttr->getName() << "\");\n";
    }
  }
}

void Pattern::emitMatcher(DagInit *tree) {
  // Emit the heading.
  os << R"(
  PatternMatchResult match(OperationInst *op0) const override {
    // TODO: This just handle 1 result
    if (op0->getNumResults() != 1) return matchFailure();
    auto state = std::make_unique<MatchedState>();)"
     << "\n";
  matchOp(tree, 0);
  os.indent(4) << "return matchSuccess(std::move(state));\n  }\n";
}

void Pattern::emit(StringRef rewriteName) {
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");
  // Collect bound arguments and compute number of ops matched.
  // TODO(jpienaar): the benefit metric is simply number of ops matched at the
  // moment, revise.
  collectBoundArguments(tree);

  // Emit RewritePattern for Pattern.
  DefInit *root = cast<DefInit>(tree->getOperator());
  auto *rootName = cast<StringInit>(root->getDef()->getValueInit("opName"));
  os << formatv(R"(struct {0} : public RewritePattern {
  {0}(MLIRContext *context) : RewritePattern({1}, {2}, context) {{})",
                rewriteName, rootName->getAsString(), numberOfOpsMatched)
     << "\n";

  // Emit matched state.
  os << "  struct MatchedState : public PatternState {\n";
  for (auto &arg : boundArguments) {
    if (auto namedAttr =
            arg.second.arg.dyn_cast<Operator::NamedAttribute *>()) {
      os.indent(4) << namedAttr->attr.getStorageType() << " " << arg.first()
                   << ";\n";
    } else {
      os.indent(4) << "Value* " << arg.first() << ";\n";
    }
  }
  os << "  };\n";

  emitMatcher(tree);
  ListInit *resultOps = pattern->getValueAsListInit("ResultOps");
  if (resultOps->size() != 1)
    PrintFatalError("only single result rules supported");
  DagInit *resultTree = cast<DagInit>(resultOps->getElement(0));

  // TODO(jpienaar): Expand to multiple results.
  for (auto result : resultTree->getArgs()) {
    if (isa<DagInit>(result))
      PrintFatalError(pattern->getLoc(), "only single op result supported");
  }

  DefInit *resultRoot = cast<DefInit>(resultTree->getOperator());
  Operator &resultOp = getOperator(resultRoot->getDef());
  auto resultOperands = resultRoot->getDef()->getValueAsDag("arguments");

  os << formatv(R"(
  void rewrite(OperationInst *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    auto& s = *static_cast<MatchedState *>(state.get());
    rewriter.replaceOpWithNewOp<{0}>(op, op->getResult(0)->getType())",
                resultOp.cppClassName());
  if (resultOperands->getNumArgs() != resultTree->getNumArgs()) {
    PrintFatalError(pattern->getLoc(),
                    Twine("mismatch between arguments of resultant op (") +
                        Twine(resultOperands->getNumArgs()) +
                        ") and arguments provided for rewrite (" +
                        Twine(resultTree->getNumArgs()) + Twine(')'));
  }

  // Create the builder call for the result.
  // Add operands.
  int i = 0;
  for (auto operand : resultOp.getOperands()) {
    // Start each operand on its own line.
    (os << ",\n").indent(6);

    auto name = resultTree->getArgNameStr(i);
    if (boundArguments.find(name) == boundArguments.end())
      PrintFatalError(pattern->getLoc(),
                      Twine("referencing unbound variable '") + name + "'");
    if (operand.name)
      os << "/*" << operand.name->getAsUnquotedString() << "=*/";
    os << "s." << name;
    // TODO(jpienaar): verify types
    ++i;
  }

  // Add attributes.
  for (int e = resultTree->getNumArgs(); i != e; ++i) {
    // Start each attribute on its own line.
    (os << ",\n").indent(6);

    // The argument in the result DAG pattern.
    auto name = resultTree->getArgNameStr(i);
    auto opName = resultOp.getArgName(i);
    auto defInit = dyn_cast<DefInit>(resultTree->getArg(i));
    auto *value = defInit ? defInit->getDef()->getValue("value") : nullptr;
    if (!value) {
      if (boundArguments.find(name) == boundArguments.end())
        PrintFatalError(pattern->getLoc(),
                        Twine("referencing unbound variable '") + name + "'");
      auto result = "s." + name;
      os << "/*" << opName << "=*/";
      if (defInit) {
        auto transform = defInit->getDef();
        if (transform->isSubClassOf("tAttr")) {
          // TODO(jpienaar): move to helper class.
          os << formatv(
              transform->getValueAsString("attrTransform").str().c_str(),
              result);
          continue;
        }
      }
      os << result;
      continue;
    }

    // TODO(jpienaar): Refactor out into map to avoid recomputing these.
    auto argument = resultOp.getArg(i);
    if (!argument.is<Operator::NamedAttribute *>())
      PrintFatalError(pattern->getLoc(),
                      Twine("expected attribute ") + Twine(i));

    if (!name.empty())
      os << "/*" << name << "=*/";
    emitAttributeValue(defInit->getDef());
    // TODO(jpienaar): verify types
  }
  os << "\n    );\n  }\n};\n";
}

void Pattern::emit(StringRef rewriteName, Record *p, raw_ostream &os) {
  Pattern pattern(p, os);
  pattern.emit(rewriteName);
}

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");

  // Ensure unique patterns simply by appending unique suffix.
  std::string baseRewriteName = "GeneratedConvert";
  int rewritePatternCount = 0;
  for (Record *p : patterns) {
    Pattern::emit(baseRewriteName + llvm::utostr(rewritePatternCount++), p, os);
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void populateWithGenerated(MLIRContext *context, "
     << "OwningRewritePatternList *patterns) {\n";
  for (unsigned i = 0; i != rewritePatternCount; ++i) {
    os.indent(2) << "patterns->push_back(std::make_unique<" << baseRewriteName
                 << i << ">(context));\n";
  }
  os << "}\n";
}

mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
