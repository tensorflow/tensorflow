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
#include "llvm/Support/FormatAdapters.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

namespace llvm {
template <> struct format_provider<mlir::tblgen::Pattern::IdentifierLine> {
  static void format(const mlir::tblgen::Pattern::IdentifierLine &v,
                     raw_ostream &os, StringRef style) {
    os << v.first << ":" << v.second;
  }
};
} // end namespace llvm

//===----------------------------------------------------------------------===//
// PatternEmitter
//===----------------------------------------------------------------------===//

namespace {
class PatternEmitter {
public:
  PatternEmitter(Record *pat, RecordOperatorMap *mapper, raw_ostream &os);

  // Emits the mlir::RewritePattern struct named `rewriteName`.
  void emit(StringRef rewriteName);

private:
  // Emits the code for matching ops.
  void emitMatchLogic(DagNode tree);

  // Emits the code for rewriting ops.
  void emitRewriteLogic();

  //===--------------------------------------------------------------------===//
  // Match utilities
  //===--------------------------------------------------------------------===//

  // Emits C++ statements for matching the op constrained by the given DAG
  // `tree`.
  void emitOpMatch(DagNode tree, int depth);

  // Emits C++ statements for matching the `index`-th argument of the given DAG
  // `tree` as an operand.
  void emitOperandMatch(DagNode tree, int index, int depth, int indent);

  // Emits C++ statements for matching the `index`-th argument of the given DAG
  // `tree` as an attribute.
  void emitAttributeMatch(DagNode tree, int index, int depth, int indent);

  //===--------------------------------------------------------------------===//
  // Rewrite utilities
  //===--------------------------------------------------------------------===//

  // Entry point for handling a result pattern rooted at `resultTree` and
  // dispatches to concrete handlers. The given tree is the `resultIndex`-th
  // argument of the enclosing DAG.
  std::string handleResultPattern(DagNode resultTree, int resultIndex,
                                  int depth);

  // Emits the C++ statement to replace the matched DAG with a value built via
  // calling native C++ code.
  std::string handleReplaceWithNativeCodeCall(DagNode resultTree);

  // Returns the C++ expression referencing the old value serving as the
  // replacement.
  std::string handleReplaceWithValue(DagNode tree);

  // Emits the C++ statement to build a new op out of the given DAG `tree` and
  // returns the variable name that this op is assigned to. If the root op in
  // DAG `tree` has a specified name, the created op will be assigned to a
  // variable of the given name. Otherwise, a unique name will be used as the
  // result value name.
  std::string handleOpCreation(DagNode tree, int resultIndex, int depth);

  // Returns the C++ expression to construct a constant attribute of the given
  // `value` for the given attribute kind `attr`.
  std::string handleConstantAttr(Attribute attr, StringRef value);

  // Returns the C++ expression to build an argument from the given DAG `leaf`.
  // `patArgName` is used to bound the argument to the source pattern.
  std::string handleOpArgument(DagLeaf leaf, StringRef patArgName);

  //===--------------------------------------------------------------------===//
  // General utilities
  //===--------------------------------------------------------------------===//

  // Collects all of the operations within the given dag tree.
  void collectOps(DagNode tree, llvm::SmallPtrSetImpl<const Operator *> &ops);

  // Returns a unique symbol for a local variable of the given `op`.
  std::string getUniqueSymbol(const Operator *op);

  //===--------------------------------------------------------------------===//
  // Symbol utilities
  //===--------------------------------------------------------------------===//

  // Gets the substitution for `symbol`. Aborts if `symbol` is not bound.
  std::string resolveSymbol(StringRef symbol);

  // Returns how many static values the given DAG `node` correspond to.
  int getNodeValueCount(DagNode node);

private:
  // Pattern instantiation location followed by the location of multiclass
  // prototypes used. This is intended to be used as a whole to
  // PrintFatalError() on errors.
  ArrayRef<llvm::SMLoc> loc;

  // Op's TableGen Record to wrapper object.
  RecordOperatorMap *opMap;

  // Handy wrapper for pattern being emitted.
  Pattern pattern;

  // Map for all bound symbols' info.
  SymbolInfoMap symbolInfoMap;

  // The next unused ID for newly created values.
  unsigned nextValueId;

  raw_ostream &os;

  // Format contexts containing placeholder substitutations.
  FmtContext fmtCtx;

  // Number of op processed.
  int opCounter = 0;
};
} // end anonymous namespace

PatternEmitter::PatternEmitter(Record *pat, RecordOperatorMap *mapper,
                               raw_ostream &os)
    : loc(pat->getLoc()), opMap(mapper), pattern(pat, mapper),
      symbolInfoMap(pat->getLoc()), nextValueId(0), os(os) {
  fmtCtx.withBuilder("rewriter");
}

std::string PatternEmitter::handleConstantAttr(Attribute attr,
                                               StringRef value) {
  if (!attr.isConstBuildable())
    PrintFatalError(loc, "Attribute " + attr.getAttrDefName() +
                             " does not have the 'constBuilderCall' field");

  // TODO(jpienaar): Verify the constants here
  return tgfmt(attr.getConstBuilderTemplate(), &fmtCtx, value);
}

// Helper function to match patterns.
void PatternEmitter::emitOpMatch(DagNode tree, int depth) {
  Operator &op = tree.getDialectOp(opMap);
  if (op.isVariadic()) {
    PrintFatalError(loc, formatv("matching op '{0}' with variadic "
                                 "operands/results is unsupported right now",
                                 op.getOperationName()));
  }

  int indent = 4 + 2 * depth;
  os.indent(indent) << formatv(
      "auto castedOp{0} = dyn_cast_or_null<{1}>(op{0}); (void)castedOp{0};\n",
      depth, op.getQualCppClassName());
  // Skip the operand matching at depth 0 as the pattern rewriter already does.
  if (depth != 0) {
    // Skip if there is no defining operation (e.g., arguments to function).
    os.indent(indent) << formatv("if (!castedOp{0}) return matchFailure();\n",
                                 depth);
  }
  if (tree.getNumArgs() != op.getNumArgs()) {
    PrintFatalError(loc, formatv("op '{0}' argument number mismatch: {1} in "
                                 "pattern vs. {2} in definition",
                                 op.getOperationName(), tree.getNumArgs(),
                                 op.getNumArgs()));
  }

  // If the operand's name is set, set to that variable.
  auto name = tree.getSymbol();
  if (!name.empty())
    os.indent(indent) << formatv("{0} = castedOp{1};\n", name, depth);

  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    auto opArg = op.getArg(i);

    // Handle nested DAG construct first
    if (DagNode argTree = tree.getArgAsNestedDag(i)) {
      os.indent(indent) << "{\n";
      os.indent(indent + 2)
          << formatv("auto *op{0} = op{1}->getOperand({2})->getDefiningOp();\n",
                     depth + 1, depth, i);
      emitOpMatch(argTree, depth + 1);
      os.indent(indent + 2)
          << formatv("tblgen_ops[{0}] = op{1};\n", ++opCounter, depth + 1);
      os.indent(indent) << "}\n";
      continue;
    }

    // Next handle DAG leaf: operand or attribute
    if (opArg.is<NamedTypeConstraint *>()) {
      emitOperandMatch(tree, i, depth, indent);
    } else if (opArg.is<NamedAttribute *>()) {
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
                                 &fmtCtx.withSelf(self))
                        << ")) return matchFailure();\n";
    }
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << formatv("{0} = op{1}->getOperand({2});\n", name, depth,
                                 index);
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
      "auto tblgen_attr = op{0}->getAttrOfType<{1}>(\"{2}\");\n", depth,
      attr.getStorageType(), namedAttr->name);

  // TODO(antiagainst): This should use getter method to avoid duplication.
  if (attr.hasDefaultValueInitializer()) {
    os.indent(indent) << "if (!tblgen_attr) tblgen_attr = "
                      << tgfmt(attr.getConstBuilderTemplate(), &fmtCtx,
                               attr.getDefaultValueInitializer())
                      << ";\n";
  } else if (attr.isOptional()) {
    // For a missing attribute that is optional according to definition, we
    // should just capature a mlir::Attribute() to signal the missing state.
    // That is precisely what getAttr() returns on missing attributes.
  } else {
    os.indent(indent) << "if (!tblgen_attr) return matchFailure();\n";
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
                               &fmtCtx.withSelf("tblgen_attr"))
                      << ")) return matchFailure();\n";
  }

  // Capture the value
  auto name = tree.getArgName(index);
  if (!name.empty()) {
    os.indent(indent) << formatv("{0} = tblgen_attr;\n", name);
  }

  indent -= 2;
  os.indent(indent) << "}\n";
}

void PatternEmitter::emitMatchLogic(DagNode tree) {
  emitOpMatch(tree, 0);

  for (auto &appliedConstraint : pattern.getConstraints()) {
    auto &constraint = appliedConstraint.constraint;
    auto &entities = appliedConstraint.entities;

    auto condition = constraint.getConditionTemplate();
    auto cmd = "if (!({0})) return matchFailure();\n";

    if (isa<TypeConstraint>(constraint)) {
      auto self = formatv("({0}->getType())", resolveSymbol(entities.front()));
      os.indent(4) << formatv(cmd,
                              tgfmt(condition, &fmtCtx.withSelf(self.str())));
    } else if (isa<AttrConstraint>(constraint)) {
      PrintFatalError(
          loc, "cannot use AttrConstraint in Pattern multi-entity constraints");
    } else {
      // TODO(b/138794486): replace formatv arguments with the exact specified
      // args.
      if (entities.size() > 4) {
        PrintFatalError(loc, "only support up to 4-entity constraints now");
      }
      SmallVector<std::string, 4> names;
      int i = 0;
      for (int e = entities.size(); i < e; ++i)
        names.push_back(resolveSymbol(entities[i]));
      std::string self = appliedConstraint.self;
      if (!self.empty())
        self = resolveSymbol(self);
      for (; i < 4; ++i)
        names.push_back("<unused>");
      os.indent(4) << formatv(cmd,
                              tgfmt(condition, &fmtCtx.withSelf(self), names[0],
                                    names[1], names[2], names[3]));
    }
  }
}

void PatternEmitter::collectOps(DagNode tree,
                                llvm::SmallPtrSetImpl<const Operator *> &ops) {
  // Check if this tree is an operation.
  if (tree.isOperation())
    ops.insert(&tree.getDialectOp(opMap));

  // Recurse the arguments of the tree.
  for (unsigned i = 0, e = tree.getNumArgs(); i != e; ++i)
    if (auto child = tree.getArgAsNestedDag(i))
      collectOps(child, ops);
}

void PatternEmitter::emit(StringRef rewriteName) {
  // Get the DAG tree for the source pattern.
  DagNode sourceTree = pattern.getSourcePattern();

  const Operator &rootOp = pattern.getSourceRootOp();
  auto rootName = rootOp.getOperationName();

  // Collect the set of result operations.
  llvm::SmallPtrSet<const Operator *, 4> resultOps;
  for (unsigned i = 0, e = pattern.getNumResultPatterns(); i != e; ++i)
    collectOps(pattern.getResultPattern(i), resultOps);

  // Emit RewritePattern for Pattern.
  auto locs = pattern.getLocation();
  os << formatv("/* Generated from:\n\t{0:$[ instantiating\n\t]}\n*/\n",
                make_range(locs.rbegin(), locs.rend()));
  os << formatv(R"(struct {0} : public RewritePattern {
  {0}(MLIRContext *context)
      : RewritePattern("{1}", {{)",
                rewriteName, rootName);
  interleaveComma(resultOps, os, [&](const Operator *op) {
    os << '"' << op->getOperationName() << '"';
  });
  os << formatv(R"(}, {0}, context) {{})", pattern.getBenefit()) << "\n";

  // Emit matchAndRewrite() function.
  os << R"(
  PatternMatchResult matchAndRewrite(Operation *op0,
                                     PatternRewriter &rewriter) const override {
)";

  // Register all symbols bound in the source pattern.
  pattern.collectSourcePatternBoundSymbols(symbolInfoMap);

  os.indent(4) << "// Variables for capturing values and attributes used for "
                  "creating ops\n";
  // Create local variables for storing the arguments and results bound
  // to symbols.
  for (const auto &symbolInfoPair : symbolInfoMap) {
    StringRef symbol = symbolInfoPair.getKey();
    auto &info = symbolInfoPair.getValue();
    os.indent(4) << info.getVarDecl(symbol);
  }
  // TODO(jpienaar): capture ops with consistent numbering so that it can be
  // reused for fused loc.
  os.indent(4) << formatv("Operation *tblgen_ops[{0}];\n\n",
                          pattern.getSourcePattern().getNumOps());

  os.indent(4) << "// Match\n";
  os.indent(4) << "tblgen_ops[0] = op0;\n";
  emitMatchLogic(sourceTree);
  os << "\n";

  os.indent(4) << "// Rewrite\n";
  emitRewriteLogic();

  os.indent(4) << "return matchSuccess();\n";
  os << "  };\n";
  os << "};\n";
}

void PatternEmitter::emitRewriteLogic() {
  const Operator &rootOp = pattern.getSourceRootOp();
  int numExpectedResults = rootOp.getNumResults();
  int numResultPatterns = pattern.getNumResultPatterns();

  // First register all symbols bound to ops generated in result patterns.
  pattern.collectResultPatternBoundSymbols(symbolInfoMap);

  // Only the last N static values generated are used to replace the matched
  // root N-result op. We need to calculate the starting index (of the results
  // of the matched op) each result pattern is to replace.
  SmallVector<int, 4> offsets(numResultPatterns + 1, numExpectedResults);
  // If we don't need to replace any value at all, set the replacement starting
  // index as the number of result patterns so we skip all of them when trying
  // to replace the matched op's results.
  int replStartIndex = numExpectedResults == 0 ? numResultPatterns : -1;
  for (int i = numResultPatterns - 1; i >= 0; --i) {
    auto numValues = getNodeValueCount(pattern.getResultPattern(i));
    offsets[i] = offsets[i + 1] - numValues;
    if (offsets[i] == 0) {
      if (replStartIndex == -1)
        replStartIndex = i;
    } else if (offsets[i] < 0 && offsets[i + 1] > 0) {
      auto error = formatv(
          "cannot use the same multi-result op '{0}' to generate both "
          "auxiliary values and values to be used for replacing the matched op",
          pattern.getResultPattern(i).getSymbol());
      PrintFatalError(loc, error);
    }
  }

  if (offsets.front() > 0) {
    const char error[] = "no enough values generated to replace the matched op";
    PrintFatalError(loc, error);
  }

  os.indent(4) << "auto loc = rewriter.getFusedLoc({";
  for (int i = 0, e = pattern.getSourcePattern().getNumOps(); i != e; ++i) {
    os << (i ? ", " : "") << "tblgen_ops[" << i << "]->getLoc()";
  }
  os << "}); (void)loc;\n";

  // Collect the replacement value for each result
  llvm::SmallVector<std::string, 2> resultValues;
  for (int i = 0; i < numResultPatterns; ++i) {
    DagNode resultTree = pattern.getResultPattern(i);
    resultValues.push_back(handleResultPattern(resultTree, offsets[i], 0));
  }

  // Emit the final replaceOp() statement
  os.indent(4) << "rewriter.replaceOp(op0, {";
  interleaveComma(
      ArrayRef<std::string>(resultValues).drop_front(replStartIndex), os,
      [&](const std::string &symbol) { os << resolveSymbol(symbol); });
  os << "});\n";
}

std::string PatternEmitter::getUniqueSymbol(const Operator *op) {
  return formatv("tblgen_{0}_{1}", op->getCppClassName(), nextValueId++);
}

std::string PatternEmitter::handleResultPattern(DagNode resultTree,
                                                int resultIndex, int depth) {
  if (resultTree.isNativeCodeCall()) {
    auto symbol = handleReplaceWithNativeCodeCall(resultTree);
    symbolInfoMap.bindValue(symbol);
    return symbol;
  }

  if (resultTree.isReplaceWithValue()) {
    return handleReplaceWithValue(resultTree);
  }

  // Normal op creation.
  auto symbol = handleOpCreation(resultTree, resultIndex, depth);
  if (resultTree.getSymbol().empty()) {
    // This is an op not explicitly bound to a symbol in the rewrite rule.
    // Register the auto-generated symbol for it.
    symbolInfoMap.bindOpResult(symbol, pattern.getDialectOp(resultTree));
  }
  return symbol;
}

std::string PatternEmitter::handleReplaceWithValue(DagNode tree) {
  assert(tree.isReplaceWithValue());

  if (tree.getNumArgs() != 1) {
    PrintFatalError(
        loc, "replaceWithValue directive must take exactly one argument");
  }

  if (!tree.getSymbol().empty()) {
    PrintFatalError(loc, "cannot bind symbol to replaceWithValue");
  }

  return resolveSymbol(tree.getArgName(0));
}

std::string PatternEmitter::handleOpArgument(DagLeaf leaf, StringRef argName) {
  if (leaf.isConstantAttr()) {
    auto constAttr = leaf.getAsConstantAttr();
    return handleConstantAttr(constAttr.getAttribute(),
                              constAttr.getConstantValue());
  }
  if (leaf.isEnumAttrCase()) {
    auto enumCase = leaf.getAsEnumAttrCase();
    if (enumCase.isStrCase())
      return handleConstantAttr(enumCase, enumCase.getSymbol());
    // This is an enum case backed by an IntegerAttr. We need to get its value
    // to build the constant.
    std::string val = std::to_string(enumCase.getValue());
    return handleConstantAttr(enumCase, val);
  }
  if (leaf.isUnspecified() || leaf.isOperandMatcher()) {
    return argName;
  }
  if (leaf.isNativeCodeCall()) {
    return tgfmt(leaf.getNativeCodeTemplate(), &fmtCtx.withSelf(argName));
  }
  PrintFatalError(loc, "unhandled case when rewriting op");
}

std::string PatternEmitter::handleReplaceWithNativeCodeCall(DagNode tree) {
  auto fmt = tree.getNativeCodeTemplate();
  // TODO(b/138794486): replace formatv arguments with the exact specified args.
  SmallVector<std::string, 8> attrs(8);
  if (tree.getNumArgs() > 8) {
    PrintFatalError(loc, "unsupported NativeCodeCall argument numbers: " +
                             Twine(tree.getNumArgs()));
  }
  for (int i = 0, e = tree.getNumArgs(); i != e; ++i) {
    attrs[i] = handleOpArgument(tree.getArgAsLeaf(i), tree.getArgName(i));
  }
  return tgfmt(fmt, &fmtCtx, attrs[0], attrs[1], attrs[2], attrs[3], attrs[4],
               attrs[5], attrs[6], attrs[7]);
}

std::string PatternEmitter::resolveSymbol(StringRef symbol) {
  auto subst = symbolInfoMap.getValueAndRangeUse(symbol);
  if (subst.empty()) {
    PrintFatalError(loc, formatv("referencing unbound symbol '{0}'", symbol));
  }
  return subst;
}

int PatternEmitter::getNodeValueCount(DagNode node) {
  if (node.isOperation()) {
    // If the op is bound to a symbol in the rewrite rule, query its result
    // count from the symbol info map.
    auto symbol = node.getSymbol();
    if (!symbol.empty()) {
      return symbolInfoMap.getStaticValueCount(symbol);
    }
    // Otherwise this is an unbound op; we will use all its results.
    return pattern.getDialectOp(node).getNumResults();
  }
  // TODO(antiagainst): This considers all NativeCodeCall as returning one
  // value. Enhance if multi-value ones are needed.
  return 1;
}

std::string PatternEmitter::handleOpCreation(DagNode tree, int resultIndex,
                                             int depth) {
  Operator &resultOp = tree.getDialectOp(opMap);
  auto numOpArgs = resultOp.getNumArgs();

  if (resultOp.isVariadic()) {
    PrintFatalError(loc, formatv("generating op '{0}' with variadic "
                                 "operands/results is unsupported now",
                                 resultOp.getOperationName()));
  }

  if (numOpArgs != tree.getNumArgs()) {
    PrintFatalError(loc, formatv("resultant op '{0}' argument number mismatch: "
                                 "{1} in pattern vs. {2} in definition",
                                 resultOp.getOperationName(), tree.getNumArgs(),
                                 numOpArgs));
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
  for (int i = 0, e = resultOp.getNumOperands(); i != e; ++i) {
    if (auto child = tree.getArgAsNestedDag(i)) {
      childNodeNames[i] = handleResultPattern(child, i, depth + 1);
    }
  }

  // Use the specified name for this op if available. Generate one otherwise.
  std::string resultValue = tree.getSymbol();
  if (resultValue.empty())
    resultValue = getUniqueSymbol(&resultOp);
  // Strip the index to get the name for the value pack. This will be used to
  // name the local variable for the op.
  StringRef valuePackName = SymbolInfoMap::getValuePackName(resultValue);

  // Then we build the new op corresponding to this DAG node.

  // Right now we don't have general type inference in MLIR. Except a few
  // special cases listed below, we need to supply types for all results
  // when building an op.
  bool isSameOperandsAndResultType =
      resultOp.hasTrait("OpTrait::SameOperandsAndResultType");
  bool isBroadcastable =
      resultOp.hasTrait("OpTrait::BroadcastableTwoOperandsOneResult");
  bool useFirstAttr = resultOp.hasTrait("OpTrait::FirstAttrDerivedResultType");
  bool usePartialResults = valuePackName != resultValue;

  if (isSameOperandsAndResultType || isBroadcastable || useFirstAttr ||
      usePartialResults || depth > 0 || resultIndex < 0) {
    os.indent(4) << formatv("auto {0} = rewriter.create<{1}>(loc",
                            valuePackName, resultOp.getQualCppClassName());
  } else {
    // If depth == 0 and resultIndex >= 0, it means we are replacing the values
    // generated from the source pattern root op. Then we can use the source
    // pattern's value types to determine the value type of the generated op
    // here.

    // We need to specify the types for all results.
    SmallVector<std::string, 4> resultTypes;
    int numResults = resultOp.getNumResults();
    resultTypes.reserve(numResults);
    for (int i = 0; i < numResults; ++i) {
      resultTypes.push_back(
          formatv("op0->getResult({0})->getType()", resultIndex + i));
    }

    os.indent(4) << formatv("auto {0} = rewriter.create<{1}>(loc",
                            valuePackName, resultOp.getQualCppClassName())
                 << (resultTypes.empty() ? "" : ", ")
                 << llvm::join(resultTypes, ", ");
  }

  // Create the builder call for the result.
  // Add operands.
  int argIndex = 0;
  for (int e = resultOp.getNumOperands(); argIndex < e; ++argIndex) {
    const auto &operand = resultOp.getOperand(argIndex);

    // Start each operand on its own line.
    (os << ",\n").indent(6);

    if (!operand.name.empty())
      os << "/*" << operand.name << "=*/";

    if (tree.isNestedDagArg(argIndex)) {
      os << childNodeNames[argIndex];
    } else {
      DagLeaf leaf = tree.getArgAsLeaf(argIndex);
      auto symbol = resolveSymbol(tree.getArgName(argIndex));
      if (leaf.isNativeCodeCall()) {
        os << tgfmt(leaf.getNativeCodeTemplate(), &fmtCtx.withSelf(symbol));
      } else {
        os << symbol;
      }
    }
    // TODO(jpienaar): verify types
  }

  // Add attributes.
  for (; argIndex != numOpArgs; ++argIndex) {
    // Start each attribute on its own line.
    (os << ",\n").indent(6);
    // The argument in the op definition.
    auto opArgName = resultOp.getArgName(argIndex);
    if (auto subTree = tree.getArgAsNestedDag(argIndex)) {
      if (!subTree.isNativeCodeCall())
        PrintFatalError(loc, "only NativeCodeCall allowed in nested dag node "
                             "for creating attribute");
      os << formatv("/*{0}=*/{1}", opArgName,
                    handleReplaceWithNativeCodeCall(subTree));
    } else {
      auto leaf = tree.getArgAsLeaf(argIndex);
      // The argument in the result DAG pattern.
      auto patArgName = tree.getArgName(argIndex);
      if (leaf.isConstantAttr() || leaf.isEnumAttrCase()) {
        // TODO(jpienaar): Refactor out into map to avoid recomputing these.
        auto argument = resultOp.getArg(argIndex);
        if (!argument.is<NamedAttribute *>())
          PrintFatalError(loc, Twine("expected attribute ") + Twine(argIndex));
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
    PatternEmitter(p, &recordOpMap, os).emit(name);
    rewriterNames.push_back(std::move(name));
  }

  // Emit function to add the generated matchers to the pattern list.
  os << "void populateWithGenerated(MLIRContext *context, "
     << "OwningRewritePatternList *patterns) {\n";
  for (const auto &name : rewriterNames) {
    os << "  patterns->insert<" << name << ">(context);\n";
  }
  os << "}\n";
}

static mlir::GenRegistration
    genRewriters("gen-rewriters", "Generate pattern rewriters",
                 [](const RecordKeeper &records, raw_ostream &os) {
                   emitRewriters(records, os);
                   return false;
                 });
