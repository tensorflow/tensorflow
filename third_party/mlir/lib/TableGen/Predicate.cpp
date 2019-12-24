//===- Predicate.cpp - Predicate class ------------------------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Wrapper around predicates defined in TableGen.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Predicate.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

// Construct a Predicate from a record.
tblgen::Pred::Pred(const llvm::Record *record) : def(record) {
  assert(def->isSubClassOf("Pred") &&
         "must be a subclass of TableGen 'Pred' class");
}

// Construct a Predicate from an initializer.
tblgen::Pred::Pred(const llvm::Init *init) : def(nullptr) {
  if (const auto *defInit = dyn_cast_or_null<llvm::DefInit>(init))
    def = defInit->getDef();
}

std::string tblgen::Pred::getCondition() const {
  // Static dispatch to subclasses.
  if (def->isSubClassOf("CombinedPred"))
    return static_cast<const CombinedPred *>(this)->getConditionImpl();
  if (def->isSubClassOf("CPred"))
    return static_cast<const CPred *>(this)->getConditionImpl();
  llvm_unreachable("Pred::getCondition must be overridden in subclasses");
}

bool tblgen::Pred::isCombined() const {
  return def && def->isSubClassOf("CombinedPred");
}

ArrayRef<llvm::SMLoc> tblgen::Pred::getLoc() const { return def->getLoc(); }

tblgen::CPred::CPred(const llvm::Record *record) : Pred(record) {
  assert(def->isSubClassOf("CPred") &&
         "must be a subclass of Tablegen 'CPred' class");
}

tblgen::CPred::CPred(const llvm::Init *init) : Pred(init) {
  assert((!def || def->isSubClassOf("CPred")) &&
         "must be a subclass of Tablegen 'CPred' class");
}

// Get condition of the C Predicate.
std::string tblgen::CPred::getConditionImpl() const {
  assert(!isNull() && "null predicate does not have a condition");
  return def->getValueAsString("predExpr");
}

tblgen::CombinedPred::CombinedPred(const llvm::Record *record) : Pred(record) {
  assert(def->isSubClassOf("CombinedPred") &&
         "must be a subclass of Tablegen 'CombinedPred' class");
}

tblgen::CombinedPred::CombinedPred(const llvm::Init *init) : Pred(init) {
  assert((!def || def->isSubClassOf("CombinedPred")) &&
         "must be a subclass of Tablegen 'CombinedPred' class");
}

const llvm::Record *tblgen::CombinedPred::getCombinerDef() const {
  assert(def->getValue("kind") && "CombinedPred must have a value 'kind'");
  return def->getValueAsDef("kind");
}

const std::vector<llvm::Record *> tblgen::CombinedPred::getChildren() const {
  assert(def->getValue("children") &&
         "CombinedPred must have a value 'children'");
  return def->getValueAsListOfDefs("children");
}

namespace {
// Kinds of nodes in a logical predicate tree.
enum class PredCombinerKind {
  Leaf,
  And,
  Or,
  Not,
  SubstLeaves,
  Concat,
  // Special kinds that are used in simplification.
  False,
  True
};

// A node in a logical predicate tree.
struct PredNode {
  PredCombinerKind kind;
  const tblgen::Pred *predicate;
  SmallVector<PredNode *, 4> children;
  std::string expr;

  // Prefix and suffix are used by ConcatPred.
  std::string prefix;
  std::string suffix;
};
} // end anonymous namespace

// Get a predicate tree node kind based on the kind used in the predicate
// TableGen record.
static PredCombinerKind getPredCombinerKind(const tblgen::Pred &pred) {
  if (!pred.isCombined())
    return PredCombinerKind::Leaf;

  const auto &combinedPred = static_cast<const tblgen::CombinedPred &>(pred);
  return llvm::StringSwitch<PredCombinerKind>(
             combinedPred.getCombinerDef()->getName())
      .Case("PredCombinerAnd", PredCombinerKind::And)
      .Case("PredCombinerOr", PredCombinerKind::Or)
      .Case("PredCombinerNot", PredCombinerKind::Not)
      .Case("PredCombinerSubstLeaves", PredCombinerKind::SubstLeaves)
      .Case("PredCombinerConcat", PredCombinerKind::Concat);
}

namespace {
// Substitution<pattern, replacement>.
using Subst = std::pair<StringRef, StringRef>;
} // end anonymous namespace

// Build the predicate tree starting from the top-level predicate, which may
// have children, and perform leaf substitutions inplace.  Note that after
// substitution, nodes are still pointing to the original TableGen record.
// All nodes are created within "allocator".
static PredNode *buildPredicateTree(const tblgen::Pred &root,
                                    llvm::BumpPtrAllocator &allocator,
                                    ArrayRef<Subst> substitutions) {
  auto *rootNode = allocator.Allocate<PredNode>();
  new (rootNode) PredNode;
  rootNode->kind = getPredCombinerKind(root);
  rootNode->predicate = &root;
  if (!root.isCombined()) {
    rootNode->expr = root.getCondition();
    // Apply all parent substitutions from innermost to outermost.
    for (const auto &subst : llvm::reverse(substitutions)) {
      auto pos = rootNode->expr.find(subst.first);
      while (pos != std::string::npos) {
        rootNode->expr.replace(pos, subst.first.size(), subst.second);
        // Skip the newly inserted substring, which itself may consider the
        // pattern to match.
        pos += subst.second.size();
        // Find the next possible match position.
        pos = rootNode->expr.find(subst.first, pos);
      }
    }
    return rootNode;
  }

  // If the current combined predicate is a leaf substitution, append it to the
  // list before continuing.
  auto allSubstitutions = llvm::to_vector<4>(substitutions);
  if (rootNode->kind == PredCombinerKind::SubstLeaves) {
    const auto &substPred = static_cast<const tblgen::SubstLeavesPred &>(root);
    allSubstitutions.push_back(
        {substPred.getPattern(), substPred.getReplacement()});
  }
  // If the current predicate is a ConcatPred, record the prefix and suffix.
  else if (rootNode->kind == PredCombinerKind::Concat) {
    const auto &concatPred = static_cast<const tblgen::ConcatPred &>(root);
    rootNode->prefix = concatPred.getPrefix();
    rootNode->suffix = concatPred.getSuffix();
  }

  // Build child subtrees.
  auto combined = static_cast<const tblgen::CombinedPred &>(root);
  for (const auto *record : combined.getChildren()) {
    auto childTree =
        buildPredicateTree(tblgen::Pred(record), allocator, allSubstitutions);
    rootNode->children.push_back(childTree);
  }
  return rootNode;
}

// Simplify a predicate tree rooted at "node" using the predicates that are
// known to be true(false).  For AND(OR) combined predicates, if any of the
// children is known to be false(true), the result is also false(true).
// Furthermore, for AND(OR) combined predicates, children that are known to be
// true(false) don't have to be checked dynamically.
static PredNode *propagateGroundTruth(
    PredNode *node, const llvm::SmallPtrSetImpl<tblgen::Pred *> &knownTruePreds,
    const llvm::SmallPtrSetImpl<tblgen::Pred *> &knownFalsePreds) {
  // If the current predicate is known to be true or false, change the kind of
  // the node and return immediately.
  if (knownTruePreds.count(node->predicate) != 0) {
    node->kind = PredCombinerKind::True;
    node->children.clear();
    return node;
  }
  if (knownFalsePreds.count(node->predicate) != 0) {
    node->kind = PredCombinerKind::False;
    node->children.clear();
    return node;
  }

  // If the current node is a substitution, stop recursion now.
  // The expressions in the leaves below this node were rewritten, but the nodes
  // still point to the original predicate records.  While the original
  // predicate may be known to be true or false, it is not necessarily the case
  // after rewriting.
  // TODO(zinenko,jpienaar): we can support ground truth for rewritten
  // predicates by either (a) having our own unique'ing of the predicates
  // instead of relying on TableGen record pointers or (b) taking ground truth
  // values optionally prefixed with a list of substitutions to apply, e.g.
  // "predX is true by itself as well as predSubY leaf substitution had been
  // applied to it".
  if (node->kind == PredCombinerKind::SubstLeaves) {
    return node;
  }

  // Otherwise, look at child nodes.

  // Move child nodes into some local variable so that they can be optimized
  // separately and re-added if necessary.
  llvm::SmallVector<PredNode *, 4> children;
  std::swap(node->children, children);

  for (auto &child : children) {
    // First, simplify the child.  This maintains the predicate as it was.
    auto simplifiedChild =
        propagateGroundTruth(child, knownTruePreds, knownFalsePreds);

    // Just add the child if we don't know how to simplify the current node.
    if (node->kind != PredCombinerKind::And &&
        node->kind != PredCombinerKind::Or) {
      node->children.push_back(simplifiedChild);
      continue;
    }

    // Second, based on the type define which known values of child predicates
    // immediately collapse this predicate to a known value, and which others
    // may be safely ignored.
    //   OR(..., True, ...) = True
    //   OR(..., False, ...) = OR(..., ...)
    //   AND(..., False, ...) = False
    //   AND(..., True, ...) = AND(..., ...)
    auto collapseKind = node->kind == PredCombinerKind::And
                            ? PredCombinerKind::False
                            : PredCombinerKind::True;
    auto eraseKind = node->kind == PredCombinerKind::And
                         ? PredCombinerKind::True
                         : PredCombinerKind::False;
    const auto &collapseList =
        node->kind == PredCombinerKind::And ? knownFalsePreds : knownTruePreds;
    const auto &eraseList =
        node->kind == PredCombinerKind::And ? knownTruePreds : knownFalsePreds;
    if (simplifiedChild->kind == collapseKind ||
        collapseList.count(simplifiedChild->predicate) != 0) {
      node->kind = collapseKind;
      node->children.clear();
      return node;
    } else if (simplifiedChild->kind == eraseKind ||
               eraseList.count(simplifiedChild->predicate) != 0) {
      continue;
    }
    node->children.push_back(simplifiedChild);
  }
  return node;
}

// Combine a list of predicate expressions using a binary combiner.  If a list
// is empty, return "init".
static std::string combineBinary(ArrayRef<std::string> children,
                                 std::string combiner, std::string init) {
  if (children.empty())
    return init;

  auto size = children.size();
  if (size == 1)
    return children.front();

  std::string str;
  llvm::raw_string_ostream os(str);
  os << '(' << children.front() << ')';
  for (unsigned i = 1; i < size; ++i) {
    os << ' ' << combiner << " (" << children[i] << ')';
  }
  return os.str();
}

// Prepend negation to the only condition in the predicate expression list.
static std::string combineNot(ArrayRef<std::string> children) {
  assert(children.size() == 1 && "expected exactly one child predicate of Neg");
  return (Twine("!(") + children.front() + Twine(')')).str();
}

// Recursively traverse the predicate tree in depth-first post-order and build
// the final expression.
static std::string getCombinedCondition(const PredNode &root) {
  // Immediately return for non-combiner predicates that don't have children.
  if (root.kind == PredCombinerKind::Leaf)
    return root.expr;
  if (root.kind == PredCombinerKind::True)
    return "true";
  if (root.kind == PredCombinerKind::False)
    return "false";

  // Recurse into children.
  llvm::SmallVector<std::string, 4> childExpressions;
  childExpressions.reserve(root.children.size());
  for (const auto &child : root.children)
    childExpressions.push_back(getCombinedCondition(*child));

  // Combine the expressions based on the predicate node kind.
  if (root.kind == PredCombinerKind::And)
    return combineBinary(childExpressions, "&&", "true");
  if (root.kind == PredCombinerKind::Or)
    return combineBinary(childExpressions, "||", "false");
  if (root.kind == PredCombinerKind::Not)
    return combineNot(childExpressions);
  if (root.kind == PredCombinerKind::Concat) {
    assert(childExpressions.size() == 1 &&
           "ConcatPred should only have one child");
    return root.prefix + childExpressions.front() + root.suffix;
  }

  // Substitutions were applied before so just ignore them.
  if (root.kind == PredCombinerKind::SubstLeaves) {
    assert(childExpressions.size() == 1 &&
           "substitution predicate must have one child");
    return childExpressions[0];
  }

  llvm::PrintFatalError(root.predicate->getLoc(), "unsupported predicate kind");
}

std::string tblgen::CombinedPred::getConditionImpl() const {
  llvm::BumpPtrAllocator allocator;
  auto predicateTree = buildPredicateTree(*this, allocator, {});
  predicateTree = propagateGroundTruth(
      predicateTree,
      /*knownTruePreds=*/llvm::SmallPtrSet<tblgen::Pred *, 2>(),
      /*knownFalsePreds=*/llvm::SmallPtrSet<tblgen::Pred *, 2>());

  return getCombinedCondition(*predicateTree);
}

StringRef tblgen::SubstLeavesPred::getPattern() const {
  return def->getValueAsString("pattern");
}

StringRef tblgen::SubstLeavesPred::getReplacement() const {
  return def->getValueAsString("replacement");
}

StringRef tblgen::ConcatPred::getPrefix() const {
  return def->getValueAsString("prefix");
}

StringRef tblgen::ConcatPred::getSuffix() const {
  return def->getValueAsString("suffix");
}
