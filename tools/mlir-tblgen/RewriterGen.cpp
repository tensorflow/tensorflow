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
class Pattern {
public:
  Pattern(Record *pattern) : pattern(pattern){};

  // Emit the rewrite pattern named `rewriteName` to `os`.
  void emit(StringRef rewriteName, raw_ostream &os);

  // Emits the value of constant attribute to `os`.
  void emitAttributeValue(RecordVal *value, raw_ostream &os);

private:
  Record *pattern;
};
} // end namespace

void Pattern::emitAttributeValue(RecordVal *value, raw_ostream &os) {
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

void Pattern::emit(StringRef rewriteName, raw_ostream &os) {
  DagInit *tree = pattern->getValueAsDag("PatternToMatch");

  // TODO(jpienaar): Expand to multiple matches.
  for (auto arg : tree->getArgs()) {
    if (isa<DagInit>(arg))
      PrintFatalError(pattern->getLoc(),
                      "only single pattern inputs supported");
  }

  // Emit RewritePattern for Pattern.
  DefInit *root = cast<DefInit>(tree->getOperator());
  auto *rootName = cast<StringInit>(root->getDef()->getValueInit("opName"));
  os << "struct " << rewriteName << " : public RewritePattern {\n"
     << "  " << rewriteName << "(MLIRContext *context) : RewritePattern("
     << rootName->getAsString() << ", 1, context) {}\n";

  os << "  struct MatchedState : public PatternState {\n";
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto arg = tree->getArgNameStr(i);
    if (!arg.empty())
      os.indent(6) << "Value* " << arg << ";\n";
  }
  os << "  };\n";

  StringSet<> boundArguments;
  os << R"(
  PatternMatchResult match(OperationInst *op) const override {
    // TODO: This just handle 1 result
    if (op->getNumResults() != 1) return matchFailure();
    auto state = std::make_unique<MatchedState>();)"
     << "\n";
  for (int i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto arg = tree->getArgNameStr(i);
    if (!arg.empty())
      os.indent(4) << "state->" << arg << " = op->getOperand(" << i << ");\n";
    boundArguments.insert(arg);
  }
  os.indent(4) << "return matchSuccess(std::move(state));\n  }\n";

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

    // TODO: verify that it is an arg.
    // TODO(jpienaar): Refactor out into map to avoid recomputing these.
    auto argument = resultOp.getArg(i);
    if (!argument.is<mlir::Operator::Attribute *>())
      PrintFatalError(pattern->getLoc(),
                      Twine("expected attribute ") + Twine(i));

    if (!name.empty())
      os << "/*" << name << "=*/";
    emitAttributeValue(value, os);
    // TODO(jpienaar): verify types
  }
  os << "\n    );\n  }\n};\n";
}

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");

  // Ensure unique patterns simply by appending unique suffix.
  std::string baseRewriteName = "GeneratedConvert";
  int rewritePatternCount = 0;
  for (Record *p : patterns) {
    Pattern pattern(p);
    pattern.emit(baseRewriteName + llvm::utostr(rewritePatternCount++), os);
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
