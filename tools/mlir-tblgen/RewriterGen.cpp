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
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/TableGen/Pattern.h"
#include "mlir/TableGen/Predicate.h"
#include "mlir/TableGen/Type.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

// Returns the bound symbol for the given op argument or op named `symbol`.
//
// Arguments and ops bound in the source pattern are grouped into a
// transient `PatternState` struct. This struct can be accessed in both
// `match()` and `rewrite()` via the local variable named as `s`.
static Twine getBoundSymbol(const StringRef &symbol) {
  return Twine("s.") + symbol;
}

//===----------------------------------------------------------------------===//
// PatternSymbolResolver
//===----------------------------------------------------------------------===//

namespace {
// A class for resolving symbols bound in patterns.
//
// Symbols can be bound to op arguments/results in the source pattern and op
// results in result patterns. For example, in
//
// ```
// def : Pattern<(SrcOp:$op1 $arg0, %arg1),
//               [(ResOp1:$op2), (ResOp2 $op2 (ResOp3))]>;
// ```
//
// `$argN` is bound to the `SrcOp`'s N-th argument. `$op1` is bound to `SrcOp`.
// `$op2` is bound to `ResOp1`.
//
// This class keeps track of such symbols and translates them into their bound
// values.
//
// Note that we also generate local variables for unnamed DAG nodes, like
// `(ResOp3)` in the above. Since we don't bind a symbol to the result, the
// generated local variable will be implicitly named. Those implicit names are
// not tracked in this class.
class PatternSymbolResolver {
public:
  PatternSymbolResolver(const StringMap<Argument> &srcArgs,
                        const StringSet<> &srcOperations);

  // Marks the given `symbol` as bound.  Returns false if the `symbol` is
  // already bound.
  bool add(StringRef symbol);

  // Queries the substitution for the given `symbol`.
  std::string query(StringRef symbol) const;

private:
  // Symbols bound to arguments in source pattern.
  const StringMap<Argument> &sourceArguments;
  // Symbols bound to ops (for their results) in source pattern.
  const StringSet<> &sourceOps;
  // Symbols bound to ops (for their results) in result patterns.
  StringSet<> resultOps;
};
} // end anonymous namespace

PatternSymbolResolver::PatternSymbolResolver(const StringMap<Argument> &srcArgs,
                                             const StringSet<> &srcOperations)
    : sourceArguments(srcArgs), sourceOps(srcOperations) {}

bool PatternSymbolResolver::add(StringRef symbol) {
  return resultOps.insert(symbol).second;
}

std::string PatternSymbolResolver::query(StringRef symbol) const {
  {
    auto it = resultOps.find(symbol);
    if (it != resultOps.end())
      return it->getKey();
  }
  {
    auto it = sourceArguments.find(symbol);
    if (it != sourceArguments.end())
      return getBoundSymbol(symbol).str();
  }
  {
    auto it = sourceOps.find(symbol);
    if (it != sourceOps.end())
      return getBoundSymbol(symbol).str();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// PatternEmitter
//===----------------------------------------------------------------------===//

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
                                   int depth);

  // Emits the C++ statement to replace the matched DAG with a value built via
  // calling native C++ code.
  std::string emitReplaceWithNativeCodeCall(DagNode resultTree);

  // Returns the C++ expression referencing the old value serving as the
  // replacement.
  std::string handleReplaceWithValue(DagNode tree);

  // Handles the `verifyUnusedValue` directive: emitting C++ statements to check
  // the `index`-th result of the source op is not used.
  void handleVerifyUnusedValue(DagNode tree, int index);

  // Emits the C++ statement to build a new op out of the given DAG `tree` and
  // returns the variable name that this op is assigned to. If the root op in
  // DAG `tree` has a specified name, the created op will be assigned to a
  // variable of the given name. Otherwise, a unique name will be used as the
  // result value name.
  std::string emitOpCreate(DagNode tree, int resultIndex, int depth);

  // Returns the C++ expression to construct a constant attribute of the given
  // `value` for the given attribute kind `attr`.
  std::string handleConstantAttr(Attribute attr, StringRef value);

  // Returns the C++ expression to build an argument from the given DAG `leaf`.
  // `patArgName` is used to bound the argument to the source pattern.
  std::string handleOpArgument(DagLeaf leaf, llvm::StringRef patArgName);

  // Marks the symbol attached to DagNode `node` as bound. Aborts if the symbol
  // is already bound.
  void addSymbol(DagNode node);

  // Gets the substitution for `symbol`. Aborts if `symbol` is not bound.
  std::string resolveSymbol(StringRef symbol);

private:
  // Pattern instantiation location followed by the location of multiclass
  // prototypes used. This is intended to be used as a whole to
  // PrintFatalError() on errors.
  ArrayRef<llvm::SMLoc> loc;
  // Op's TableGen Record to wrapper object
  RecordOperatorMap *opMap;
  // Handy wrapper for pattern being emitted
  Pattern pattern;
  PatternSymbolResolver symbolResolver;
  // The next unused ID for newly created values
  unsigned nextValueId;
  raw_ostream &os;

  // Format contexts containing placeholder substitutations for match().
  FmtContext matchCtx;
  // Format contexts containing placeholder substitutations for rewrite().
  FmtContext rewriteCtx;
};
} // end anonymous namespace

PatternEmitter::PatternEmitter(Record *pat, RecordOperatorMap *mapper,
                               raw_ostream &os)
    : loc(pat->getLoc()), opMap(mapper), pattern(pat, mapper),
      symbolResolver(pattern.getSourcePatternBoundArgs(),
                     pattern.getSourcePatternBoundOps()),
      nextValueId(0), os(os) {
  matchCtx.withBuilder("mlir::Builder(ctx)");
  rewriteCtx.withBuilder("rewriter");
}

std::string PatternEmitter::handleConstantAttr(Attribute attr,
                                               StringRef value) {
  if (!attr.isConstBuildable())
    PrintFatalError(loc, "Attribute " + attr.getTableGenDefName() +
                             " does not have the 'constBuilderCall' field");

  // TODO(jpienaar): Verify the constants here
  return tgfmt(attr.getConstBuilderTemplate(),
               &rewriteCtx.withBuilder("rewriter"), value);
}

// Helper function to match patterns.
void PatternEmitter::emitOpMatch(DagNode tree, int depth) {
  Operator &op = tree.getDialectOp(opMap);
  int indent = 4 + 2 * depth;
  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0) {
    // Skip if there is no defining operation (e.g., arguments to function).
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
    os.indent(indent) << formatv("{0} = op{1};\n", getBoundSymbol(name), depth);

  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    auto opArg = op.getArg(i);

    // Handle nested DAG construct first
    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2)
          << formatv("auto op{0} = op{1}->getOperand({2})->getDefiningOp();\n",
                     depth + 1, depth, i);
      emitOpMatch(argTree, depth + 1);
      os.indent(indent) << "}\n";
      continue;
    }

    // Next handle DAG leaf: operand or attribute
    if (auto *operand = opArg.dyn_cast<NamedTypeConstraint *>()) {
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
  auto *operand = op.getArg(index).get<NamedTypeConstraint *>();
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
    if (operand->constraint != matcher.getAsConstraint()) {
      auto self = formatv("op{0}->getOperand({1})->getType()", depth, index);
      os.indent(indent) << "if (!("
                        << tgfmt(matcher.getConditionTemplate(),
                                 &matchCtx.withSelf(self))
                        << ")) return matchFailure();\n";
    }
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << getBoundSymbol(name) << " = op" << depth
                      << "->getOperand(" << index << ");\n";
  }
}

void PatternEmitter::emitAttributeMatch(DagNode tree, int index, int depth,
                                        int indent) {
  Operator &op = tree.getDialectOp(opMap);
  auto *namedAttr = op.getArg(index).get<NamedAttribute *>();
  const auto &attr = namedAttr->attr;

  os.indent(indent) << "{\n";
  indent += 2;
  os.indent(indent) << formatv(
      "auto attr = op{0}->getAttrOfType<{1}>(\"{2}\");\n", depth,
      attr.getStorageType(), namedAttr->name);

  // TODO(antiagainst): This should use getter method to avoid duplication.
  if (attr.hasDefaultValueInitializer()) {
    os.indent(indent) << "if (!attr) attr = "
                      << tgfmt(attr.getConstBuilderTemplate(), &matchCtx,
                               attr.getDefaultValueInitializer())
                      << ";\n";
  } else if (attr.isOptional()) {
    // For a missing attribut that is optional according to definition, we
    // should just capature a mlir::Attribute() to signal the missing state.
    // That is precisely what getAttr() returns on missing attributes.
  } else {
    os.indent(indent) << "if (!attr) return matchFailure();\n";
  }

  auto matcher = tree.getArgAsLeaf(index);
  if (!matcher.isUnspecified()) {
    if (!matcher.isAttrMatcher()) {
      PrintFatalError(
          loc, formatv("the {1}-th argument of op '{0}' should be an attribute",
                       op.getOperationName(), index + 1));
    }

    // If a constraint is specified, we need to generate C++ statements to
    // check the constraint.
    os.indent(indent) << "if (!("
                      << tgfmt(matcher.getConditionTemplate(),
                               &matchCtx.withSelf("attr"))
                      << ")) return matchFailure();\n";
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << getBoundSymbol(name) << " = attr;\n";
  }

  indent -= 2;
  os.indent(indent) << "}\n";
}

void PatternEmitter::emitMatchMethod(DagNode tree) {
  // Emit the heading.
  os << R"(
  PatternMatchResult match(Operation *op0) const override {
    auto ctx = op0->getContext(); (void)ctx;
    auto state = llvm::make_unique<MatchedState>();
    auto &s = *state;
)";

  // The rewrite pattern may specify that certain outputs should be unused in
  // the source IR. Check it here.
  for (int i = 0, e = pattern.getNumResults(); i < e; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    if (resultTree.isVerifyUnusedValue()) {
      handleVerifyUnusedValue(resultTree, i);
    }
  }

  emitOpMatch(tree, 0);

  for (auto &appliedConstraint : pattern.getConstraints()) {
    auto &constraint = appliedConstraint.constraint;
    auto &entities = appliedConstraint.entities;

    auto condition = constraint.getConditionTemplate();
    auto cmd = "if (!{0}) return matchFailure();\n";

    if (isa<TypeConstraint>(constraint)) {
      auto self = formatv("(*{0}->result_type_begin())",
                          resolveSymbol(entities.front()));
      // TODO(jpienaar): Verify op only has one result.
      os.indent(4) << formatv(cmd,
                              tgfmt(condition, &matchCtx.withSelf(self.str())));
    } else if (isa<AttrConstraint>(constraint)) {
      PrintFatalError(
          loc, "cannot use AttrConstraint in Pattern multi-entity constraints");
    } else {
      // TODO(fengliuai): replace formatv arguments with the exact specified
      // args.
      if (entities.size() > 4) {
        PrintFatalError(loc, "only support up to 4-entity constraints now");
      }
      SmallVector<std::string, 4> names;
      unsigned i = 0;
      for (unsigned e = entities.size(); i < e; ++i)
        names.push_back(resolveSymbol(entities[i]));
      for (; i < 4; ++i)
        names.push_back("<unused>");
      os.indent(4) << formatv(cmd, tgfmt(condition, &matchCtx, names[0],
                                         names[1], names[2], names[3]));
    }
  }

  os.indent(4) << "return matchSuccess(std::move(state));\n  }\n";
}

void PatternEmitter::emit(StringRef rewriteName) {
  // Get the DAG tree for the source pattern
  DagNode tree = pattern.getSourcePattern();

  const Operator &rootOp = pattern.getSourceRootOp();
  auto rootName = rootOp.getOperationName();

  if (rootOp.hasVariadicResult())
    PrintFatalError(
        loc, "replacing op with variadic results not supported right now");

  // Emit RewritePattern for Pattern.
  os << formatv(R"(struct {0} : public RewritePattern {
  {0}(MLIRContext *context) : RewritePattern("{1}", {2}, context) {{})",
                rewriteName, rootName, pattern.getBenefit())
     << "\n";

  // Emit matched state.
  os << "  struct MatchedState : public PatternState {\n";
  for (const auto &arg : pattern.getSourcePatternBoundArgs()) {
    auto fieldName = arg.first();
    if (auto namedAttr = arg.second.dyn_cast<NamedAttribute *>()) {
      os.indent(4) << namedAttr->attr.getStorageType() << " " << fieldName
                   << ";\n";
    } else {
      os.indent(4) << "Value *" << fieldName << ";\n";
    }
  }
  for (const auto &result : pattern.getSourcePatternBoundOps()) {
    os.indent(4) << "Operation *" << result.getKey() << ";\n";
  }
  os << "  };\n";

  emitMatchMethod(tree);
  emitRewriteMethod();

  os << "};\n";
}

void PatternEmitter::emitRewriteMethod() {
  const Operator &rootOp = pattern.getSourceRootOp();
  int numExpectedResults = rootOp.getNumResults();
  unsigned numProvidedResults = pattern.getNumResults();

  if (numProvidedResults < numExpectedResults)
    PrintFatalError(
        loc, "no enough result patterns to replace root op in source pattern");

  os << R"(
  void rewrite(Operation *op, std::unique_ptr<PatternState> state,
               PatternRewriter &rewriter) const override {
    auto& s = *static_cast<MatchedState *>(state.get());
    auto loc = op->getLoc(); (void)loc;
)";

  // Collect the replacement value for each result
  llvm::SmallVector<std::string, 2> resultValues;
  for (unsigned i = 0; i < numProvidedResults; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    resultValues.push_back(handleRewritePattern(resultTree, i, 0));
    // Keep track of bound symbols at the top-level DAG nodes
    addSymbol(resultTree);
  }

  // Emit the final replaceOp() statement
  os.indent(4) << "rewriter.replaceOp(op, {";
  interleave(
      // We only use the last numExpectedResults ones to replace the root op.
      ArrayRef<std::string>(resultValues).take_back(numExpectedResults),
      [&](const std::string &name) { os << name; }, [&]() { os << ", "; });
  os << "});\n  }\n";
}

std::string PatternEmitter::getUniqueValueName(const Operator *op) {
  return formatv("v{0}{1}", op->getCppClassName(), nextValueId++);
}

std::string PatternEmitter::handleRewritePattern(DagNode resultTree,
                                                 int resultIndex, int depth) {
  if (resultTree.isNativeCodeCall())
    return emitReplaceWithNativeCodeCall(resultTree);

  if (resultTree.isVerifyUnusedValue()) {
    if (depth > 0) {
      // TODO: Revisit this when we have use cases of matching an intermediate
      // multi-result op with no uses of its certain results.
      PrintFatalError(loc, "verifyUnusedValue directive can only be used to "
                           "verify top-level result");
    }

    if (!resultTree.getOpName().empty()) {
      PrintFatalError(loc, "cannot bind symbol to verifyUnusedValue");
    }

    // The C++ statements to check that this result value is unused are already
    // emitted in the match() method. So returning a nullptr here directly
    // should be safe because the C++ RewritePattern harness will use it to
    // replace nothing.
    return "nullptr";
  }

  if (resultTree.isReplaceWithValue())
    return handleReplaceWithValue(resultTree);

  return emitOpCreate(resultTree, resultIndex, depth);
}

std::string PatternEmitter::handleReplaceWithValue(DagNode tree) {
  assert(tree.isReplaceWithValue());

  if (tree.getNumArgs() != 1) {
    PrintFatalError(
        loc, "replaceWithValue directive must take exactly one argument");
  }

  if (!tree.getOpName().empty()) {
    PrintFatalError(loc, "cannot bind symbol to verifyUnusedValue");
  }

  auto name = tree.getArgName(0);
  pattern.ensureBoundInSourcePattern(name);

  return getBoundSymbol(name).str();
}

void PatternEmitter::handleVerifyUnusedValue(DagNode tree, int index) {
  assert(tree.isVerifyUnusedValue());

  os.indent(4) << "if (!op0->getResult(" << index
               << ")->use_empty()) return matchFailure();\n";
}

std::string PatternEmitter::handleOpArgument(DagLeaf leaf,
                                             llvm::StringRef argName) {
  if (leaf.isConstantAttr()) {
    auto constAttr = leaf.getAsConstantAttr();
    return handleConstantAttr(constAttr.getAttribute(),
                              constAttr.getConstantValue());
  }
  if (leaf.isEnumAttrCase()) {
    auto enumCase = leaf.getAsEnumAttrCase();
    return handleConstantAttr(enumCase, enumCase.getSymbol());
  }
  pattern.ensureBoundInSourcePattern(argName);
  std::string result = getBoundSymbol(argName).str();
  if (leaf.isUnspecified() || leaf.isOperandMatcher()) {
    return result;
  }
  if (leaf.isNativeCodeCall()) {
    return tgfmt(leaf.getNativeCodeTemplate(), &rewriteCtx.withSelf(result));
  }
  PrintFatalError(loc, "unhandled case when rewriting op");
}

std::string PatternEmitter::emitReplaceWithNativeCodeCall(DagNode tree) {
  auto fmt = tree.getNativeCodeTemplate();
  // TODO(fengliuai): replace formatv arguments with the exact specified args.
  SmallVector<std::string, 8> attrs(8);
  if (tree.getNumArgs() > 8) {
    PrintFatalError(loc, "unsupported NativeCodeCall argument numbers: " +
                             Twine(tree.getNumArgs()));
  }
  for (unsigned i = 0, e = tree.getNumArgs(); i != e; ++i) {
    attrs[i] = handleOpArgument(tree.getArgAsLeaf(i), tree.getArgName(i));
  }
  return tgfmt(fmt, &rewriteCtx, attrs[0], attrs[1], attrs[2], attrs[3],
               attrs[4], attrs[5], attrs[6], attrs[7]);
}

void PatternEmitter::addSymbol(DagNode node) {
  StringRef symbol = node.getOpName();
  // Skip empty-named symbols, which happen for unbound ops in result patterns.
  if (symbol.empty())
    return;
  if (!symbolResolver.add(symbol))
    PrintFatalError(loc, formatv("symbol '{0}' bound more than once", symbol));
}

std::string PatternEmitter::resolveSymbol(StringRef symbol) {
  auto subst = symbolResolver.query(symbol);
  if (subst.empty())
    PrintFatalError(loc, formatv("referencing unbound symbol '{0}'", symbol));
  return subst;
}

std::string PatternEmitter::emitOpCreate(DagNode tree, int resultIndex,
                                         int depth) {
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
  // the key. This includes both bound and unbound child nodes. Bound child
  // nodes will additionally be tracked in `symbolResolver` so they can be
  // referenced by other patterns. Unbound child nodes will only be used once
  // to build this op.
  llvm::DenseMap<unsigned, std::string> childNodeNames;

  // First go through all the child nodes who are nested DAG constructs to
  // create ops for them, so that we can use the results in the current node.
  // This happens in a recursive manner.
  for (unsigned i = 0, e = resultOp.getNumOperands(); i != e; ++i) {
    if (auto child = tree.getArgAsNestedDag(i)) {
      childNodeNames[i] = handleRewritePattern(child, i, depth + 1);
      // Keep track of bound symbols at the middle-level DAG nodes
      addSymbol(child);
    }
  }

  // Use the specified name for this op if available. Generate one otherwise.
  std::string resultValue = tree.getOpName();
  if (resultValue.empty())
    resultValue = getUniqueValueName(&resultOp);

  // Then we build the new op corresponding to this DAG node.

  // TODO: this is a hack to support various constant ops. We are assuming
  // all of them have no operands and one attribute here. Figure out a better
  // way to do this.
  bool isConstOp =
      resultOp.getNumOperands() == 0 && resultOp.getNumNativeAttributes() == 1;

  bool isSameValueType = resultOp.hasTrait("SameOperandsAndResultType");
  bool isBroadcastable = resultOp.hasTrait("BroadcastableTwoOperandsOneResult");
  bool useFirstAttr = resultOp.hasTrait("FirstAttrDerivedResultType");

  if (isConstOp || isSameValueType || isBroadcastable || useFirstAttr) {
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
  for (int e = resultOp.getNumOperands(); i < e; ++i) {
    const auto &operand = resultOp.getOperand(i);

    // Start each operand on its own line.
    (os << ",\n").indent(6);

    if (!operand.name.empty())
      os << "/*" << operand.name << "=*/";

    if (tree.isNestedDagArg(i)) {
      os << childNodeNames[i];
    } else {
      DagLeaf leaf = tree.getArgAsLeaf(i);
      auto symbol = resolveSymbol(tree.getArgName(i));
      if (leaf.isNativeCodeCall()) {
        os << tgfmt(leaf.getNativeCodeTemplate(), &rewriteCtx.withSelf(symbol));
      } else {
        os << symbol;
      }
    }
    // TODO(jpienaar): verify types
  }

  // Add attributes.
  for (int e = tree.getNumArgs(); i != e; ++i) {
    // Start each attribute on its own line.
    (os << ",\n").indent(6);
    // The argument in the op definition.
    auto opArgName = resultOp.getArgName(i);
    if (auto subTree = tree.getArgAsNestedDag(i)) {
      if (!subTree.isNativeCodeCall())
        PrintFatalError(loc, "only NativeCodeCall allowed in nested dag node "
                             "for creating attribute");
      os << formatv("/*{0}=*/{1}", opArgName,
                    emitReplaceWithNativeCodeCall(subTree));
    } else {
      auto leaf = tree.getArgAsLeaf(i);
      // The argument in the result DAG pattern.
      auto patArgName = tree.getArgName(i);
      if (leaf.isConstantAttr() || leaf.isEnumAttrCase()) {
        // TODO(jpienaar): Refactor out into map to avoid recomputing these.
        auto argument = resultOp.getArg(i);
        if (!argument.is<NamedAttribute *>())
          PrintFatalError(loc, Twine("expected attribute ") + Twine(i));
        if (!patArgName.empty())
          os << "/*" << patArgName << "=*/";
      } else {
        os << "/*" << opArgName << "=*/";
      }
      os << handleOpArgument(leaf, patArgName);
    }
  }
  os << "\n    );\n";

  return resultValue;
}

void PatternEmitter::emit(StringRef rewriteName, Record *p,
                          RecordOperatorMap *mapper, raw_ostream &os) {
  PatternEmitter(p, mapper, os).emit(rewriteName);
}

static void emitRewriters(const RecordKeeper &recordKeeper, raw_ostream &os) {
  emitSourceFileHeader("Rewriters", os);

  const auto &patterns = recordKeeper.getAllDerivedDefinitions("Pattern");
  auto numPatterns = patterns.size();

  // We put the map here because it can be shared among multiple patterns.
  RecordOperatorMap recordOpMap;

  std::vector<std::string> rewriterNames;
  rewriterNames.reserve(numPatterns);

  std::string baseRewriterName = "GeneratedConvert";
  int rewriterIndex = 0;

  for (Record *p : patterns) {
    std::string name;
    if (p->isAnonymous()) {
      // If no name is provided, ensure unique rewriter names simply by
      // appending unique suffix.
      name = baseRewriterName + llvm::utostr(rewriterIndex++);
    } else {
      name = p->getName();
    }
    PatternEmitter::emit(name, p, &recordOpMap, os);
    rewriterNames.push_back(std::move(name));
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void populateWithGenerated(MLIRContext *context, "
     << "OwningRewritePatternList *patterns) {\n";
  for (const auto &name : rewriterNames) {
    os << "  patterns->push_back(llvm::make_unique<" << name
       << ">(context));\n";
  }
  os << "}\n";
}

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
