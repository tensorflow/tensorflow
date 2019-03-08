//===- RewriterGen.cpp - MLIR pattern rewriter generator ------------------===//
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

#include "mlir/Support/STLExtras.h"
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

using mlir::tblgen::DagNode;
using mlir::tblgen::NamedAttribute;
using mlir::tblgen::Operator;
using mlir::tblgen::RecordOperatorMap;
using mlir::tblgen::Value;

namespace {
class PatternEmitter {
public:
  static void emit(StringRef rewriteName, Record *p, RecordOperatorMap *mapper,
                   raw_ostream &os);

private:
  PatternEmitter(Record *pat, RecordOperatorMap *mapper, raw_ostream &os);

  // Emits the mlir::RewritePattern struct named `rewriteName`.
  void emit(StringRef rewriteName);

  // Emits the match() method.
  void emitMatchMethod(DagNode tree);

  // Emits the rewrite() method.
  void emitRewriteMethod();

  // Emits the value of constant attribute to `os`.
  void emitConstantAttr(tblgen::ConstantAttr constAttr);

  // Emits C++ statements for matching the op constrained by the given DAG
  // `tree`.
  void emitOpMatch(DagNode tree, int depth);

  // Emits C++ statements for matching the `index`-th argument of the given DAG
  // `tree` as an operand.
  void emitOperandMatch(DagNode tree, int index, int depth, int indent);
  // Emits C++ statements for matching the `index`-th argument of the given DAG
  // `tree` as an attribute.
  void emitAttributeMatch(DagNode tree, int index, int depth, int indent);

  // Returns a unique name for an value of the given `op`.
  std::string getUniqueValueName(const Operator *op);

  // Entry point for handling a rewrite pattern rooted at `resultTree` and
  // dispatches to concrete handlers. The given tree is the `resultIndex`-th
  // argument of the enclosing DAG.
  std::string handleRewritePattern(DagNode resultTree, int resultIndex,
                                   int depth, llvm::StringRef treeName = "");

  // Emits the C++ statement to replace the matched DAG with a native C++ built
  // value.
  std::string emitReplaceWithNativeBuilder(DagNode resultTree);

  // Returns the C++ expression referencing the old value serving as the
  // replacement.
  std::string handleReplaceWithValue(DagNode tree);

  // Handles the `verifyUnusedValue` directive: emitting C++ statements to check
  // the `index`-th result of the source op is not used.
  void handleVerifyUnusedValue(DagNode tree, int index);

  // Emits the C++ statement to build a new op out of the given DAG `tree` and
  // returns the variable name that this op is assigned to. If `treeName` is not
  // empty, the created op will be assigned to a variable of the given
  // `treeName`. Otherwise, a unique name will be used as the result value name.
  std::string emitOpCreate(DagNode tree, int resultIndex, int depth,
                           llvm::StringRef treeName = "");

private:
  // Pattern instantiation location followed by the location of multiclass
  // prototypes used. This is intended to be used as a whole to
  // PrintFatalError() on errors.
  ArrayRef<llvm::SMLoc> loc;
  // Op's TableGen Record to wrapper object
  RecordOperatorMap *opMap;
  // Handy wrapper for pattern being emitted
  tblgen::Pattern pattern;
  // The next unused ID for newly created values
  unsigned nextValueId;
  raw_ostream &os;
};
} // end namespace

PatternEmitter::PatternEmitter(Record *pat, RecordOperatorMap *mapper,
                               raw_ostream &os)
    : loc(pat->getLoc()), opMap(mapper), pattern(pat, mapper), nextValueId(0),
      os(os) {}

void PatternEmitter::emitConstantAttr(tblgen::ConstantAttr constAttr) {
  auto attr = constAttr.getAttribute();

  if (!attr.isConstBuildable())
    PrintFatalError(loc, "Attribute " + attr.getTableGenDefName() +
                             " does not have the 'constBuilderCall' field");

  // TODO(jpienaar): Verify the constants here
  os << formatv(attr.getConstBuilderTemplate().str().c_str(), "rewriter",
                constAttr.getConstantValue());
}

static Twine resultName(const StringRef &name) { return Twine("res_") + name; }

static Twine boundArgNameInMatch(const StringRef &name) {
  // Bound value in the source pattern are grouped into a transient struct. That
  // struct is hold in a local variable named as "state" in the match() method.
  return Twine("state->") + name;
}

static Twine boundArgNameInRewrite(const StringRef &name) {
  // Bound value in the source pattern are grouped into a transient struct. That
  // struct is passed into the rewrite() method as a parameter with name `s`.
  return Twine("s.") + name;
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
        op.getQualCppClassName());
  }
  if (tree.getNumArgs() != op.getNumArgs()) {
    PrintFatalError(loc, formatv("op '{0}' argument number mismatch: {1} in "
                                 "pattern vs. {2} in definition",
                                 op.getOperationName(), tree.getNumArgs(),
                                 op.getNumArgs()));
  }

  // If the operand's name is set, set to that variable.
  auto name = tree.getOpName();
  if (!name.empty())
    os.indent(indent) << formatv("{0} = op{1};\n", resultName(name), depth);

  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    auto opArg = op.getArg(i);

    // Handle nested DAG construct first
    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2) << formatv(
          "auto op{0} = op{1}->getOperand({2})->getDefiningInst();\n",
          depth + 1, depth, i);
      emitOpMatch(argTree, depth + 1);
      os.indent(indent) << "}\n";
      continue;
    }

    // Next handle DAG leaf: operand or attribute
    if (auto *operand = opArg.dyn_cast<Value *>()) {
      emitOperandMatch(tree, i, depth, indent);
    } else if (auto *namedAttr = opArg.dyn_cast<NamedAttribute *>()) {
      emitAttributeMatch(tree, i, depth, indent);
    } else {
      PrintFatalError(loc, "unhandled case when matching op");
    }
  }
}

void PatternEmitter::emitOperandMatch(DagNode tree, int index, int depth,
                                      int indent) {
  Operator &op = tree.getDialectOp(opMap);
  auto *operand = op.getArg(index).get<Value *>();
  auto matcher = tree.getArgAsLeaf(index);

  // If a constraint is specified, we need to generate C++ statements to
  // check the constraint.
  if (!matcher.isUnspecified()) {
    if (!matcher.isOperandMatcher()) {
      PrintFatalError(
          loc, formatv("the {1}-th argument of op '{0}' should be an operand",
                       op.getOperationName(), index + 1));
    }

    // Only need to verify if the matcher's type is different from the one
    // of op definition.
    if (static_cast<tblgen::TypeConstraint>(operand->type) !=
        matcher.getAsTypeConstraint()) {
      os.indent(indent) << "if (!("
                        << formatv(matcher.getConditionTemplate().c_str(),
                                   formatv("op{0}->getOperand({1})->getType()",
                                           depth, index))
                        << ")) return matchFailure();\n";
    }
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << "state->" << name << " = op" << depth
                      << "->getOperand(" << index << ");\n";
  }
}

void PatternEmitter::emitAttributeMatch(DagNode tree, int index, int depth,
                                        int indent) {
  Operator &op = tree.getDialectOp(opMap);
  auto *namedAttr = op.getArg(index).get<NamedAttribute *>();
  auto matcher = tree.getArgAsLeaf(index);

  if (!matcher.isUnspecified()) {
    if (!matcher.isAttrMatcher()) {
      PrintFatalError(
          loc, formatv("the {1}-th argument of op '{0}' should be an attribute",
                       op.getOperationName(), index + 1));
    }

    // If a constraint is specified, we need to generate C++ statements to
    // check the constraint.
    std::string condition = formatv(
        matcher.getConditionTemplate().c_str(),
        formatv("op{0}->getAttrOfType<{1}>(\"{2}\")", depth,
                namedAttr->attr.getStorageType(), namedAttr->getName()));
    os.indent(indent) << "if (!(" << condition << ")) return matchFailure();\n";
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << "state->" << name << " = op" << depth
                      << "->getAttrOfType<" << namedAttr->attr.getStorageType()
                      << ">(\"" << namedAttr->getName() << "\");\n";
  }
}

void PatternEmitter::emitMatchMethod(DagNode tree) {
  // Emit the heading.
  os << R"(
  PatternMatchResult match(Instruction *op0) const override {
    auto ctx = op0->getContext(); (void)ctx;
    auto state = std::make_unique<MatchedState>();)"
     << "\n";

  // The rewrite pattern may specify that certain outputs should be unused in
  // the source IR. Check it here.
  for (int i = 0, e = pattern.getNumResults(); i < e; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    if (resultTree.isVerifyUnusedValue()) {
      handleVerifyUnusedValue(resultTree, i);
    }
  }

  for (auto &res : pattern.getSourcePatternBoundResults())
    os.indent(4) << formatv("mlir::Instruction* {0}; (void){0};\n",
                            resultName(res.first()));

  emitOpMatch(tree, 0);

  auto deduceName = [&](const std::string &name) -> std::string {
    if (pattern.isArgBoundInSourcePattern(name)) {
      return boundArgNameInMatch(name).str();
    }
    if (pattern.isResultBoundInSourcePattern(name)) {
      return resultName(name).str();
    }
    PrintFatalError(loc, formatv("referencing unbound variable '{0}'", name));
  };

  for (auto constraint : pattern.getConstraints()) {
    if (constraint.isTypeConstraint()) {
      auto cmd = "if (!{0}) return matchFailure();\n";
      // TODO(jpienaar): Use the op definition here to simplify this.
      auto condition = constraint.getAsTypeConstraint().getConditionTemplate();
      // TODO(jpienaar): Verify op only has one result.
      os.indent(4) << formatv(
          cmd, formatv(condition.c_str(),
                       "(*" + deduceName(*constraint.name_begin()) +
                           "->result_type_begin())"));
    } else if (constraint.isNativeConstraint()) {
      os.indent(4) << "if (!" << constraint.getNativeConstraintFunction()
                   << "(";
      interleave(constraint.name_begin(), constraint.name_end(),
                 [&](const std::string &name) { os << deduceName(name); },
                 [&]() { os << ", "; });
      os << ")) return matchFailure();\n";
    } else {
      llvm_unreachable(
          "Pattern constraints have to be either a type or native constraint");
    }
  }

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
    auto fieldName = arg.first();
    if (auto namedAttr = arg.second.dyn_cast<NamedAttribute *>()) {
      os.indent(4) << namedAttr->attr.getStorageType() << " " << fieldName
                   << ";\n";
    } else {
      os.indent(4) << "Value* " << fieldName << ";\n";
    }
  }
  os << "  };\n";

  emitMatchMethod(tree);
  emitRewriteMethod();

  os << "};\n";
}

void PatternEmitter::emitRewriteMethod() {
  unsigned numResults = pattern.getNumResults();
  if (numResults == 0)
    PrintFatalError(loc, "must provide at least one result pattern");

  os << R"(
  void rewrite(Instruction *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    auto& s = *static_cast<MatchedState *>(state.get());
    auto loc = op->getLoc(); (void)loc;
)";

  // Collect the replacement value for each result
  llvm::SmallVector<std::string, 2> resultValues;
  for (unsigned i = 0; i < numResults; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    resultValues.push_back(handleRewritePattern(resultTree, i, 0));
  }

  // Emit the final replaceOp() statement
  os.indent(4) << "rewriter.replaceOp(op, {";
  interleave(
      resultValues, [&](const std::string &name) { os << name; },
      [&]() { os << ", "; });
  os << "});\n  }\n";
}

std::string PatternEmitter::getUniqueValueName(const Operator *op) {
  return formatv("v{0}{1}", op->getCppClassName(), nextValueId++);
}

std::string PatternEmitter::handleRewritePattern(DagNode resultTree,
                                                 int resultIndex, int depth,
                                                 llvm::StringRef treeName) {
  if (resultTree.isNativeCodeBuilder())
    return emitReplaceWithNativeBuilder(resultTree);

  if (resultTree.isVerifyUnusedValue()) {
    if (depth > 0) {
      // TODO: Revisit this when we have use cases of matching an intermediate
      // multi-result op with no uses of its certain results.
      PrintFatalError(loc, "verifyUnusedValue directive can only be used to "
                           "verify top-level result");
    }
    // The C++ statements to check that this result value is unused are already
    // emitted in the match() method. So returning a nullptr here directly
    // should be safe because the C++ RewritePattern harness will use it to
    // replace nothing.
    return "nullptr";
  }

  if (resultTree.isReplaceWithValue())
    return handleReplaceWithValue(resultTree);

  return emitOpCreate(resultTree, resultIndex, depth, treeName);
}

std::string PatternEmitter::handleReplaceWithValue(DagNode tree) {
  assert(tree.isReplaceWithValue());

  if (tree.getNumArgs() != 1) {
    PrintFatalError(
        loc, "replaceWithValue directive must take exactly one argument");
  }

  auto name = tree.getArgName(0);
  pattern.ensureArgBoundInSourcePattern(name);

  return boundArgNameInRewrite(name).str();
}

void PatternEmitter::handleVerifyUnusedValue(DagNode tree, int index) {
  assert(tree.isVerifyUnusedValue());

  os.indent(4) << "if (!op0->getResult(" << index
               << ")->use_empty()) return matchFailure();\n";
}

std::string PatternEmitter::emitOpCreate(DagNode tree, int resultIndex,
                                         int depth, StringRef treeName) {
  Operator &resultOp = tree.getDialectOp(opMap);
  auto numOpArgs = resultOp.getNumArgs();

  if (numOpArgs != tree.getNumArgs()) {
    PrintFatalError(loc, formatv("resultant op '{0}' argument number mismatch: "
                                 "{1} in pattern vs. {2} in definition",
                                 resultOp.getOperationName(), tree.getNumArgs(),
                                 numOpArgs));
  }

  if (resultOp.getNumResults() > 1) {
    PrintFatalError(
        loc, formatv("generating multiple-result op '{0}' is unsupported now",
                     resultOp.getOperationName()));
  }

  // A map to collect all nested DAG child nodes' names, with operand index as
  // the key.
  llvm::DenseMap<unsigned, std::string> childNodeNames;

  // First go through all the child nodes who are nested DAG constructs to
  // create ops for them, so that we can use the results in the current node.
  // This happens in a recursive manner.
  for (unsigned i = 0, e = resultOp.getNumOperands(); i != e; ++i) {
    if (auto child = tree.getArgAsNestedDag(i)) {
      childNodeNames[i] =
          handleRewritePattern(child, i, depth + 1, tree.getArgName(i));
    }
  }

  std::string resultValue =
      treeName.empty() ? getUniqueValueName(&resultOp) : treeName.str();

  // Then we build the new op corresponding to this DAG node.

  // Returns the name we should use for the `index`-th argument of this
  // DAG node. This is needed because the we can reference an argument
  // 1) generated from a nested DAG node and implicitly named,
  // 2) bound in the source pattern and explicitly named,
  // 3) bound in the result pattern and explicitly named.
  auto deduceArgName = [&](unsigned index) -> std::string {
    if (tree.isNestedDagArg(index)) {
      // Implicitly named
      return childNodeNames[index];
    }

    auto name = tree.getArgName(index);
    if (this->pattern.isArgBoundInSourcePattern(name)) {
      // Bound in source pattern, explicitly named
      return boundArgNameInRewrite(name).str();
    }

    // Bound in result pattern, explicitly named
    return name.str();
  };

  // TODO: this is a hack to support various constant ops. We are assuming
  // all of them have no operands and one attribute here. Figure out a better
  // way to do this.
  bool isConstOp =
      resultOp.getNumOperands() == 0 && resultOp.getNumNativeAttributes() == 1;

  bool isSameValueType = resultOp.hasTrait("SameOperandsAndResultType");
  bool isBroadcastable = resultOp.hasTrait("BroadcastableTwoOperandsOneResult");

  if (isConstOp || isSameValueType || isBroadcastable) {
    os.indent(4) << formatv("auto {0} = rewriter.create<{1}>(loc", resultValue,
                            resultOp.getQualCppClassName());
  } else {
    std::string resultType = formatv("op->getResult({0})", resultIndex).str();

    os.indent(4) << formatv(
        "auto {0} = rewriter.create<{1}>(loc, {2}->getType()", resultValue,
        resultOp.getQualCppClassName(), resultType);
  }

  // Create the builder call for the result.
  // Add operands.
  int i = 0;
  for (auto operand : resultOp.getOperands()) {
    // Start each operand on its own line.
    (os << ",\n").indent(6);

    if (!operand.name.empty())
      os << "/*" << operand.name << "=*/";

    os << deduceArgName(i);

    // TODO(jpienaar): verify types
    ++i;
  }

  // Add attributes.
  for (int e = tree.getNumArgs(); i != e; ++i) {
    // Start each attribute on its own line.
    (os << ",\n").indent(6);

    auto leaf = tree.getArgAsLeaf(i);
    // The argument in the result DAG pattern.
    auto patArgName = tree.getArgName(i);
    // The argument in the op definition.
    auto opArgName = resultOp.getArgName(i);

    if (leaf.isUnspecified() || leaf.isOperandMatcher()) {
      pattern.ensureArgBoundInSourcePattern(patArgName);
      std::string result = boundArgNameInRewrite(patArgName).str();
      os << formatv("/*{0}=*/{1}", opArgName, result);
    } else if (leaf.isAttrTransformer()) {
      pattern.ensureArgBoundInSourcePattern(patArgName);
      std::string result = boundArgNameInRewrite(patArgName).str();
      result = formatv(leaf.getTransformationTemplate().c_str(), result);
      os << formatv("/*{0}=*/{1}", opArgName, result);
    } else if (leaf.isConstantAttr()) {
      // TODO(jpienaar): Refactor out into map to avoid recomputing these.
      auto argument = resultOp.getArg(i);
      if (!argument.is<NamedAttribute *>())
        PrintFatalError(loc, Twine("expected attribute ") + Twine(i));

      if (!patArgName.empty())
        os << "/*" << patArgName << "=*/";
      emitConstantAttr(leaf.getAsConstantAttr());
      // TODO(jpienaar): verify types
    } else {
      PrintFatalError(loc, "unhandled case when rewriting op");
    }
  }
  os << "\n    );\n";

  return resultValue;
}

std::string PatternEmitter::emitReplaceWithNativeBuilder(DagNode resultTree) {
  // The variable's name for holding the result of this native builder call
  std::string value = formatv("v{0}", nextValueId++).str();

  os.indent(4) << "auto " << value << " = " << resultTree.getNativeCodeBuilder()
               << "(op, {";
  const auto &boundedValues = pattern.getSourcePatternBoundArgs();
  bool first = true;
  bool printingAttr = false;
  for (int i = 0, e = resultTree.getNumArgs(); i != e; ++i) {
    auto name = resultTree.getArgName(i);
    pattern.ensureArgBoundInSourcePattern(name);
    const auto &val = boundedValues.find(name);
    if (val->second.dyn_cast<NamedAttribute *>() && !printingAttr) {
      os << "}, {";
      first = true;
      printingAttr = true;
    }
    if (!first)
      os << ",";
    os << boundArgNameInRewrite(name);
    first = false;
  }
  if (!printingAttr)
    os << "},{";
  os << "}, rewriter);\n";

  return value;
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
