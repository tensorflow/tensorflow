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
#include "mlir/TableGen/Pattern.h"
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

using mlir::tblgen::Argument;
using mlir::tblgen::Attribute;
using mlir::tblgen::DagNode;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::Operand;
using mlir::tblgen::Operator;
using mlir::tblgen::Pattern;
using mlir::tblgen::RecordOperatorMap;
using mlir::tblgen::Type;

namespace {

// Wrapper around DAG argument.
struct DagArg {
  DagArg(Argument arg, Init *constraintInit)
      : arg(arg), constraintInit(constraintInit) {}
  bool isAttr();

  Argument arg;
  Init *constraintInit;
};

} // end namespace

bool DagArg::isAttr() { return arg.is<NamedAttribute *>(); }

namespace {
class PatternEmitter {
public:
  static void emit(StringRef rewriteName, Record *p, RecordOperatorMap *mapper,
                   raw_ostream &os);

private:
  PatternEmitter(Record *pat, RecordOperatorMap *mapper, raw_ostream &os)
      : loc(pat->getLoc()), opMap(mapper), pattern(pat, mapper), os(os) {}

  // Emits the mlir::RewritePattern struct named `rewriteName`.
  void emit(StringRef rewriteName);

  // Emits the match() method.
  void emitMatchMethod(DagNode tree);

  // Emits the rewrite() method.
  void emitRewriteMethod();

  // Emits the C++ statement to replace the matched DAG with an existing value.
  void emitReplaceWithExistingValue(DagNode resultTree);
  // Emits the C++ statement to replace the matched DAG with a new op.
  void emitReplaceOpWithNewOp(DagNode resultTree);

  // Emits the value of constant attribute to `os`.
  void emitAttributeValue(Record *constAttr);

  // Emits C++ statements for matching the op constrained by the given DAG
  // `tree`.
  void emitOpMatch(DagNode tree, int depth);

private:
  // Pattern instantiation location followed by the location of multiclass
  // prototypes used. This is intended to be used as a whole to
  // PrintFatalError() on errors.
  ArrayRef<llvm::SMLoc> loc;
  // Op's TableGen Record to wrapper object
  RecordOperatorMap *opMap;
  // Handy wrapper for pattern being emitted
  Pattern pattern;
  raw_ostream &os;
};
} // end namespace

void PatternEmitter::emitAttributeValue(Record *constAttr) {
  Attribute attr(constAttr->getValueAsDef("attr"));
  auto value = constAttr->getValue("value");

  if (!attr.isConstBuildable())
    PrintFatalError(loc, "Attribute " + attr.getTableGenDefName() +
                             " does not have the 'constBuilderCall' field");

  // TODO(jpienaar): Verify the constants here
  os << formatv(attr.getConstBuilderTemplate().str().c_str(), "rewriter",
                value->getValue()->getAsUnquotedString());
}

// Helper function to match patterns.
void PatternEmitter::emitOpMatch(DagNode tree, int depth) {
  Operator &op = tree.getDialectOp(opMap);
  int indent = 4 + 2 * depth;
  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0) {
    // Skip if there is no defining instruction (e.g., arguments to function).
    os.indent(indent) << formatv("if (!op{0}) return matchFailure();\n", depth);
    os.indent(indent) << formatv(
        "if (!op{0}->isa<{1}>()) return matchFailure();\n", depth,
        op.qualifiedCppClassName());
  }
  if (tree.getNumArgs() != op.getNumArgs())
    PrintFatalError(loc, Twine("mismatch in number of arguments to op '") +
                             op.getOperationName() +
                             "' in pattern and op's definition");
  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    auto opArg = op.getArg(i);

    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2) << formatv(
          "auto op{0} = op{1}->getOperand({2})->getDefiningInst();\n",
          depth + 1, depth, i);
      emitOpMatch(argTree, depth + 1);
      os.indent(indent) << "}\n";
      continue;
    }

    // Verify arguments.
    if (auto defInit = tree.getArgAsDefInit(i)) {
      // Verify operands.
      if (auto *operand = opArg.dyn_cast<Operand *>()) {
        // Skip verification where not needed due to definition of op.
        if (operand->defInit == defInit)
          goto StateCapture;

        if (!defInit->getDef()->isSubClassOf("Type"))
          PrintFatalError(loc, "type argument required for operand");

        auto constraint = tblgen::TypeConstraint(*defInit);
        os.indent(indent)
            << "if (!("
            << formatv(constraint.getConditionTemplate().c_str(),
                       formatv("op{0}->getOperand({1})->getType()", depth, i))
            << ")) return matchFailure();\n";
      }

      // TODO(jpienaar): Verify attributes.
      if (auto *namedAttr = opArg.dyn_cast<NamedAttribute *>()) {
        auto constraint = tblgen::AttrConstraint(defInit);
        std::string condition = formatv(
            constraint.getConditionTemplate().c_str(),
            formatv("op{0}->getAttrOfType<{1}>(\"{2}\")", depth,
                    namedAttr->attr.getStorageType(), namedAttr->getName()));
        os.indent(indent) << "if (!(" << condition
                          << ")) return matchFailure();\n";
      }
    }

  StateCapture:
    auto name = tree.getArgName(i);
    if (name.empty())
      continue;
    if (opArg.is<Operand *>())
      os.indent(indent) << "state->" << name << " = op" << depth
                        << "->getOperand(" << i << ");\n";
    if (auto namedAttr = opArg.dyn_cast<NamedAttribute *>()) {
      os.indent(indent) << "state->" << name << " = op" << depth
                        << "->getAttrOfType<"
                        << namedAttr->attr.getStorageType() << ">(\""
                        << namedAttr->getName() << "\");\n";
    }
  }
}

void PatternEmitter::emitMatchMethod(DagNode tree) {
  // Emit the heading.
  os << R"(
  PatternMatchResult match(OperationInst *op0) const override {
    // TODO: This just handle 1 result
    if (op0->getNumResults() != 1) return matchFailure();
    auto state = std::make_unique<MatchedState>();)"
     << "\n";
  emitOpMatch(tree, 0);
  os.indent(4) << "return matchSuccess(std::move(state));\n  }\n";
}

void PatternEmitter::emit(StringRef rewriteName) {
  // Get the DAG tree for the source pattern
  DagNode tree = pattern.getSourcePattern();

  // TODO(jpienaar): the benefit metric is simply number of ops matched at the
  // moment, revise.
  unsigned benefit = tree.getNumOps();

  const Operator &rootOp = pattern.getSourceRootOp();
  auto rootName = rootOp.getOperationName();

  // Emit RewritePattern for Pattern.
  os << formatv(R"(struct {0} : public RewritePattern {
  {0}(MLIRContext *context) : RewritePattern("{1}", {2}, context) {{})",
                rewriteName, rootName, benefit)
     << "\n";

  // Emit matched state.
  os << "  struct MatchedState : public PatternState {\n";
  for (const auto &arg : pattern.getSourcePatternBoundArgs()) {
    if (auto namedAttr = arg.second.arg.dyn_cast<NamedAttribute *>()) {
      os.indent(4) << namedAttr->attr.getStorageType() << " " << arg.first()
                   << ";\n";
    } else {
      os.indent(4) << "Value* " << arg.first() << ";\n";
    }
  }
  os << "  };\n";

  emitMatchMethod(tree);
  emitRewriteMethod();

  os << "};\n";
}

void PatternEmitter::emitRewriteMethod() {
  if (pattern.getNumResults() != 1)
    PrintFatalError("only single result rules supported");

  DagNode resultTree = pattern.getResultPattern(0);

  // TODO(jpienaar): Expand to multiple results.
  for (unsigned i = 0, e = resultTree.getNumArgs(); i != e; ++i)
    if (resultTree.getArgAsNestedDag(i))
      PrintFatalError(loc, "only single op result supported");

  os << R"(
  void rewrite(OperationInst *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    auto& s = *static_cast<MatchedState *>(state.get());
)";

  if (resultTree.isReplaceWithValue())
    emitReplaceWithExistingValue(resultTree);
  else
    emitReplaceOpWithNewOp(resultTree);

  os << "  }\n";
}

void PatternEmitter::emitReplaceWithExistingValue(DagNode resultTree) {
  if (resultTree.getNumArgs() != 1) {
    PrintFatalError(loc, "exactly one argument needed in the result pattern");
  }

  auto name = resultTree.getArgName(0);
  pattern.ensureArgBoundInSourcePattern(name);
  os.indent(4) << "rewriter.replaceOp(op, {s." << name << "});\n";
}

void PatternEmitter::emitReplaceOpWithNewOp(DagNode resultTree) {
  Operator &resultOp = resultTree.getDialectOp(opMap);
  auto numOpArgs =
      resultOp.getNumOperands() + resultOp.getNumNativeAttributes();

  os << formatv(R"(
    rewriter.replaceOpWithNewOp<{0}>(op, op->getResult(0)->getType())",
                resultOp.cppClassName());
  if (numOpArgs != resultTree.getNumArgs()) {
    PrintFatalError(loc, Twine("mismatch between arguments of resultant op (") +
                             Twine(numOpArgs) +
                             ") and arguments provided for rewrite (" +
                             Twine(resultTree.getNumArgs()) + Twine(')'));
  }

  // Create the builder call for the result.
  // Add operands.
  int i = 0;
  for (auto operand : resultOp.getOperands()) {
    // Start each operand on its own line.
    (os << ",\n").indent(6);

    auto name = resultTree.getArgName(i);
    pattern.ensureArgBoundInSourcePattern(name);
    if (operand.name)
      os << "/*" << operand.name->getAsUnquotedString() << "=*/";
    os << "s." << name;
    // TODO(jpienaar): verify types
    ++i;
  }

  // Add attributes.
  for (int e = resultTree.getNumArgs(); i != e; ++i) {
    // Start each attribute on its own line.
    (os << ",\n").indent(6);

    // The argument in the result DAG pattern.
    auto argName = resultTree.getArgName(i);
    auto opName = resultOp.getArgName(i);
    auto *defInit = resultTree.getArgAsDefInit(i);
    auto *value = defInit ? defInit->getDef()->getValue("value") : nullptr;
    if (!value) {
      pattern.ensureArgBoundInSourcePattern(argName);
      auto result = "s." + argName;
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
    if (!argument.is<NamedAttribute *>())
      PrintFatalError(loc, Twine("expected attribute ") + Twine(i));

    if (!argName.empty())
      os << "/*" << argName << "=*/";
    emitAttributeValue(defInit->getDef());
    // TODO(jpienaar): verify types
  }
  os << "\n    );\n";
}

void PatternEmitter::emit(StringRef rewriteName, Record *p,
                          RecordOperatorMap *mapper, raw_ostream &os) {
  PatternEmitter(p, mapper, os).emit(rewriteName);
}

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);
  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");

  // We put the map here because it can be shared among multiple patterns.
  RecordOperatorMap recordOpMap;

  // Ensure unique patterns simply by appending unique suffix.
  std::string baseRewriteName = "GeneratedConvert";
  int rewritePatternCount = 0;
  for (Record *p : patterns) {
    PatternEmitter::emit(baseRewriteName + llvm::utostr(rewritePatternCount++),
                         p, &recordOpMap, os);
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

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
