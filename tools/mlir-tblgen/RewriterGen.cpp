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

#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
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

namespace {

// Wrapper around dag argument.
struct DagArg {
  DagArg(Init *init) : init(init){};
  bool isAttr();

  Init *init;
};

} // end namespace

bool DagArg::isAttr() {
  if (auto defInit = dyn_cast<DefInit>(init))
    return defInit->getDef()->isSubClassOf("Attr");
  return false;
}

namespace {
class Pattern {
public:
  static void emit(StringRef rewriteName, Record *p, raw_ostream &os);

private:
  Pattern(Record *pattern, raw_ostream &os) : pattern(pattern), os(os){};

  // Emit the rewrite pattern named `rewriteName`.
  void emit(StringRef rewriteName);

  // Emit the matcher.
  void emitMatcher(DagInit *tree);

  // Emits the value of constant attribute to `os`.
  void emitAttributeValue(Record *constAttr);

  // Collect bound arguments.
  void collectBoundArguments(DagInit *tree);

  // Map from bound argument name to DagArg.
  StringMap<DagArg> boundArguments;

  // Number of the operations in the input pattern.
  int numberOfOpsMatched = 0;

  Record *pattern;
  raw_ostream &os;
};
} // end namespace

void Pattern::emitAttributeValue(Record *constAttr) {
  Record *type = constAttr->getValueAsDef("type");
  auto value = constAttr->getValue("value");

  // Construct the attribute based on `type`.
  // TODO(jpienaar): Generalize this to avoid hardcoding here.
  if (type->isSubClassOf("F")) {
    string mlirType;
    switch (type->getValueAsInt("bitwidth")) {
    case 32:
      mlirType = "Type::getF32(context)";
      break;
    default:
      PrintFatalError("unsupported floating point width");
    }
    // TODO(jpienaar): Verify the floating point constant here.
    os << formatv("FloatAttr::get({0}, {1})", mlirType,
                  value->getValue()->getAsUnquotedString());
    return;
  }

  // Fallback to the type of value.
  switch (value->getType()->getRecTyKind()) {
  case RecTy::IntRecTyKind:
    // TODO(jpienaar): This is using 64-bits for all the bitwidth of the
    // type could instead be queried. These are expected to be mostly used
    // for enums or constant indices and so no arithmetic operations are
    // expected on these.
    os << formatv("IntegerAttr::get(Type::getInteger(64, context), {0})",
                  value->getValue()->getAsString());
    break;
  case RecTy::StringRecTyKind:
    os << formatv("StringAttr::get({0}, context)",
                  value->getValue()->getAsString());
    break;
  default:
    PrintFatalError(pattern->getLoc(),
                    Twine("unsupported/unimplemented value type for ") +
                        value->getName());
  }
}

void Pattern::collectBoundArguments(DagInit *tree) {
  ++numberOfOpsMatched;
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
    boundArguments.try_emplace(name, arg);
  }
}

// Helper function to match patterns.
static void matchOp(DagInit *tree, int depth, raw_ostream &os) {
  Operator op(cast<DefInit>(tree->getOperator())->getDef());
  int indent = 4 + 2 * depth;
  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0) {
    // Skip if there is no defining instruction (e.g., arguments to function).
    os.indent(indent) << formatv("if (!op{0}) return matchFailure();\n", depth);
    // TODO(jpienaar): This is bad, we should not be checking strings here, we
    // should be matching using mOp (and helpers). Currently doing this to allow
    // for TF ops that aren't registed. Fix it.
    os.indent(indent) << formatv(
                             "if (op{0}->getName().getStringRef() != \"{1}\")",
                             depth, op.getOperationName())
                      << "\n";
    os.indent(indent + 2) << "return matchFailure();\n";
  }
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto arg = tree->getArg(i);
    if (auto argTree = dyn_cast<DagInit>(arg)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2) << formatv(
          "auto op{0} = op{1}->getOperand({2})->getDefiningInst();\n",
          depth + 1, depth, i);
      matchOp(argTree, depth + 1, os);
      os.indent(indent) << "}\n";
      continue;
    }
    auto name = tree->getArgNameStr(i);
    if (name.empty())
      continue;
    os.indent(indent) << "state->" << name << " = op" << depth
                      << "->getOperand(" << i << ");\n";
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
  matchOp(tree, 0, os);
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
    if (arg.second.isAttr()) {
      DefInit *defInit = cast<DefInit>(arg.second.init);
      os.indent(4) << defInit->getDef()->getValueAsString("storageType").trim()
                   << " " << arg.first() << ";\n";
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
  Operator resultOp(*resultRoot->getDef());
  auto resultOperands = resultRoot->getDef()->getValueAsDag("arguments");

  os << formatv(R"(
  void rewrite(OperationInst *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    auto* context = op->getContext(); (void)context;
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
    std::string name = resultTree->getArgNameStr(i);
    auto defInit = dyn_cast<DefInit>(resultTree->getArg(i));
    auto *value = defInit ? defInit->getDef()->getValue("value") : nullptr;
    if (!value)
      PrintFatalError(pattern->getLoc(),
                      Twine("attribute '") + name +
                          "' needs to be constant initialized");

    // TODO(jpienaar): Refactor out into map to avoid recomputing these.
    auto argument = resultOp.getArg(i);
    if (!argument.is<mlir::Operator::Attribute *>())
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
