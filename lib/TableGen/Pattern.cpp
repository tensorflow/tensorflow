//===- Pattern.cpp - Pattern wrapper class ----------------------*- C++ -*-===//
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
// Pattern wrapper class to simplify using TableGen Record defining a MLIR
// Pattern.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/Pattern.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

using mlir::tblgen::Operator;

bool tblgen::DagLeaf::isUnspecified() const {
  return !def || isa<llvm::UnsetInit>(def);
}

bool tblgen::DagLeaf::isOperandMatcher() const {
  if (!def || !isa<llvm::DefInit>(def))
    return false;
  // Operand matchers specify a type constraint.
  return cast<llvm::DefInit>(def)->getDef()->isSubClassOf("TypeConstraint");
}

bool tblgen::DagLeaf::isAttrMatcher() const {
  if (!def || !isa<llvm::DefInit>(def))
    return false;
  // Attribute matchers specify an attribute constraint.
  return cast<llvm::DefInit>(def)->getDef()->isSubClassOf("AttrConstraint");
}

bool tblgen::DagLeaf::isAttrTransformer() const {
  if (!def || !isa<llvm::DefInit>(def))
    return false;
  return cast<llvm::DefInit>(def)->getDef()->isSubClassOf("tAttr");
}

bool tblgen::DagLeaf::isConstantAttr() const {
  if (!def || !isa<llvm::DefInit>(def))
    return false;
  return cast<llvm::DefInit>(def)->getDef()->isSubClassOf("ConstantAttr");
}

tblgen::TypeConstraint tblgen::DagLeaf::getAsTypeConstraint() const {
  assert(isOperandMatcher() && "the DAG leaf must be operand");
  return TypeConstraint(*cast<llvm::DefInit>(def)->getDef());
}

tblgen::AttrConstraint tblgen::DagLeaf::getAsAttrConstraint() const {
  assert(isAttrMatcher() && "the DAG leaf must be attribute");
  return AttrConstraint(cast<llvm::DefInit>(def)->getDef());
}

tblgen::ConstantAttr tblgen::DagLeaf::getAsConstantAttr() const {
  assert(isConstantAttr() && "the DAG leaf must be constant attribute");
  return ConstantAttr(cast<llvm::DefInit>(def));
}

std::string tblgen::DagLeaf::getConditionTemplate() const {
  assert((isOperandMatcher() || isAttrMatcher()) &&
         "the DAG leaf must be operand/attribute matcher");
  if (isOperandMatcher()) {
    return getAsTypeConstraint().getConditionTemplate();
  }
  return getAsAttrConstraint().getConditionTemplate();
}

std::string tblgen::DagLeaf::getTransformationTemplate() const {
  assert(isAttrTransformer() && "the DAG leaf must be attribute transformer");
  return cast<llvm::DefInit>(def)
      ->getDef()
      ->getValueAsString("attrTransform")
      .str();
}

bool tblgen::DagNode::isAttrTransformer() const {
  auto op = node->getOperator();
  if (!op || !isa<llvm::DefInit>(op))
    return false;
  return cast<llvm::DefInit>(op)->getDef()->isSubClassOf("tAttr");
}

std::string tblgen::DagNode::getTransformationTemplate() const {
  assert(isAttrTransformer() && "the DAG leaf must be attribute transformer");
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsString("attrTransform")
      .str();
}

llvm::StringRef tblgen::DagNode::getOpName() const {
  return node->getNameStr();
}

Operator &tblgen::DagNode::getDialectOp(RecordOperatorMap *mapper) const {
  llvm::Record *opDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return mapper->try_emplace(opDef, opDef).first->second;
}

unsigned tblgen::DagNode::getNumOps() const {
  unsigned count = isReplaceWithValue() ? 0 : 1;
  for (unsigned i = 0, e = getNumArgs(); i != e; ++i) {
    if (auto child = getArgAsNestedDag(i))
      count += child.getNumOps();
  }
  return count;
}

unsigned tblgen::DagNode::getNumArgs() const { return node->getNumArgs(); }

bool tblgen::DagNode::isNestedDagArg(unsigned index) const {
  return isa<llvm::DagInit>(node->getArg(index));
}

tblgen::DagNode tblgen::DagNode::getArgAsNestedDag(unsigned index) const {
  return DagNode(dyn_cast_or_null<llvm::DagInit>(node->getArg(index)));
}

tblgen::DagLeaf tblgen::DagNode::getArgAsLeaf(unsigned index) const {
  assert(!isNestedDagArg(index));
  return DagLeaf(node->getArg(index));
}

StringRef tblgen::DagNode::getArgName(unsigned index) const {
  return node->getArgNameStr(index);
}

static void collectBoundArguments(const llvm::DagInit *tree,
                                  tblgen::Pattern *pattern) {
  auto &op = pattern->getDialectOp(tblgen::DagNode(tree));
  if (llvm::StringInit *si = tree->getName()) {
    auto name = si->getAsUnquotedString();
    if (!name.empty())
      pattern->getSourcePatternBoundResults().insert(name);
  }

  // TODO(jpienaar): Expand to multiple matches.
  for (unsigned i = 0, e = tree->getNumArgs(); i != e; ++i) {
    auto *arg = tree->getArg(i);

    if (auto *argTree = dyn_cast<llvm::DagInit>(arg)) {
      collectBoundArguments(argTree, pattern);
      continue;
    }

    StringRef name = tree->getArgNameStr(i);
    if (name.empty())
      continue;

    pattern->getSourcePatternBoundArgs().try_emplace(name, op.getArg(i));
  }
}

void tblgen::DagNode::collectBoundArguments(tblgen::Pattern *pattern) const {
  ::collectBoundArguments(node, pattern);
}

bool tblgen::DagNode::isReplaceWithValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "replaceWithValue";
}

bool tblgen::DagNode::isVerifyUnusedValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "verifyUnusedValue";
}

bool tblgen::DagNode::isNativeCodeBuilder() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->isSubClassOf("cOp");
}

llvm::StringRef tblgen::DagNode::getNativeCodeBuilder() const {
  assert(isNativeCodeBuilder());
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getValueAsString("function");
}

// Returns whether this is a type constraint.
bool tblgen::PatternConstraint::isTypeConstraint() const {
  if (!node)
    return false;
  auto op = node->getOperator();
  if (!op || !isa<llvm::DefInit>(op))
    return false;
  // Operand matchers specify a type constraint.
  return cast<llvm::DefInit>(op)->getDef()->isSubClassOf("TypeConstraint");
}

// Returns this constraint as a TypeConstraint. Asserts if fails.
tblgen::TypeConstraint tblgen::PatternConstraint::getAsTypeConstraint() const {
  assert(isTypeConstraint());
  // Constraint specify a type constraint.
  return TypeConstraint(*cast<llvm::DefInit>(node->getOperator())->getDef());
}

static std::string toStringRef(const llvm::StringInit *si) {
  return si->getAsUnquotedString();
}

tblgen::PatternConstraint::const_name_iterator
tblgen::PatternConstraint::name_begin() const {
  return const_name_iterator(node->getArgNames().begin(), &toStringRef);
}
tblgen::PatternConstraint::const_name_iterator
tblgen::PatternConstraint::name_end() const {
  return const_name_iterator(node->getArgNames().end(), &toStringRef);
}

// Returns whether this is a native pattern constraint.
bool tblgen::PatternConstraint::isNativeConstraint() const {
  if (!node)
    return false;
  auto op = node->getOperator();
  if (!op || !isa<llvm::DefInit>(op))
    return false;
  // Operand matchers specify a type constraint.
  return cast<llvm::DefInit>(op)->getDef()->isSubClassOf("mPat");
}

// Returns the C++ function invoked as part of native constraint.
llvm::StringRef tblgen::PatternConstraint::getNativeConstraintFunction() const {
  assert(isNativeConstraint());
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsString("function");
}

tblgen::Pattern::Pattern(const llvm::Record *def, RecordOperatorMap *mapper)
    : def(*def), recordOpMap(mapper) {
  getSourcePattern().collectBoundArguments(this);
}

tblgen::DagNode tblgen::Pattern::getSourcePattern() const {
  return tblgen::DagNode(def.getValueAsDag("patternToMatch"));
}

unsigned tblgen::Pattern::getNumResults() const {
  auto *results = def.getValueAsListInit("resultOps");
  return results->size();
}

tblgen::DagNode tblgen::Pattern::getResultPattern(unsigned index) const {
  auto *results = def.getValueAsListInit("resultOps");
  return tblgen::DagNode(cast<llvm::DagInit>(results->getElement(index)));
}

bool tblgen::Pattern::isArgBoundInSourcePattern(llvm::StringRef name) const {
  return boundArguments.find(name) != boundArguments.end();
}

bool tblgen::Pattern::isResultBoundInSourcePattern(llvm::StringRef name) const {
  return boundResults.count(name);
}

void tblgen::Pattern::ensureArgBoundInSourcePattern(
    llvm::StringRef name) const {
  if (!isArgBoundInSourcePattern(name))
    PrintFatalError(def.getLoc(),
                    Twine("referencing unbound variable '") + name + "'");
}

llvm::StringMap<tblgen::Argument> &
tblgen::Pattern::getSourcePatternBoundArgs() {
  return boundArguments;
}

llvm::StringSet<> &tblgen::Pattern::getSourcePatternBoundResults() {
  return boundResults;
}

const tblgen::Operator &tblgen::Pattern::getSourceRootOp() {
  return getSourcePattern().getDialectOp(recordOpMap);
}

tblgen::Operator &tblgen::Pattern::getDialectOp(DagNode node) {
  return node.getDialectOp(recordOpMap);
}

std::vector<tblgen::PatternConstraint> tblgen::Pattern::getConstraints() const {
  auto *listInit = def.getValueAsListInit("constraints");
  std::vector<tblgen::PatternConstraint> ret;
  ret.reserve(listInit->size());
  for (auto it : *listInit) {
    auto *dagInit = cast<llvm::DagInit>(it);
    ret.emplace_back(dagInit);
  }
  return ret;
}
