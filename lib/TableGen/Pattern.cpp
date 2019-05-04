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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"

using namespace mlir;

using llvm::formatv;
using mlir::tblgen::Operator;

bool tblgen::DagLeaf::isUnspecified() const {
  return dyn_cast_or_null<llvm::UnsetInit>(def);
}

bool tblgen::DagLeaf::isOperandMatcher() const {
  // Operand matchers specify a type constraint.
  return isSubClassOf("TypeConstraint");
}

bool tblgen::DagLeaf::isAttrMatcher() const {
  // Attribute matchers specify an attribute constraint.
  return isSubClassOf("AttrConstraint");
}

bool tblgen::DagLeaf::isNativeCodeCall() const {
  return isSubClassOf("NativeCodeCall");
}

bool tblgen::DagLeaf::isConstantAttr() const {
  return isSubClassOf("ConstantAttr");
}

bool tblgen::DagLeaf::isEnumAttrCase() const {
  return isSubClassOf("EnumAttrCase");
}

tblgen::Constraint tblgen::DagLeaf::getAsConstraint() const {
  assert((isOperandMatcher() || isAttrMatcher()) &&
         "the DAG leaf must be operand or attribute");
  return Constraint(cast<llvm::DefInit>(def)->getDef());
}

tblgen::ConstantAttr tblgen::DagLeaf::getAsConstantAttr() const {
  assert(isConstantAttr() && "the DAG leaf must be constant attribute");
  return ConstantAttr(cast<llvm::DefInit>(def));
}

tblgen::EnumAttrCase tblgen::DagLeaf::getAsEnumAttrCase() const {
  assert(isEnumAttrCase() && "the DAG leaf must be an enum attribute case");
  return EnumAttrCase(cast<llvm::DefInit>(def));
}

std::string tblgen::DagLeaf::getConditionTemplate() const {
  return getAsConstraint().getConditionTemplate();
}

llvm::StringRef tblgen::DagLeaf::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(def)->getDef()->getValueAsString("expression");
}

bool tblgen::DagLeaf::isSubClassOf(StringRef superclass) const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(def))
    return defInit->getDef()->isSubClassOf(superclass);
  return false;
}

bool tblgen::DagNode::isNativeCodeCall() const {
  if (auto *defInit = dyn_cast_or_null<llvm::DefInit>(node->getOperator()))
    return defInit->getDef()->isSubClassOf("NativeCodeCall");
  return false;
}

llvm::StringRef tblgen::DagNode::getNativeCodeTemplate() const {
  assert(isNativeCodeCall() && "the DAG leaf must be NativeCodeCall");
  return cast<llvm::DefInit>(node->getOperator())
      ->getDef()
      ->getValueAsString("expression");
}

llvm::StringRef tblgen::DagNode::getOpName() const {
  return node->getNameStr();
}

Operator &tblgen::DagNode::getDialectOp(RecordOperatorMap *mapper) const {
  llvm::Record *opDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  auto it = mapper->find(opDef);
  if (it != mapper->end())
    return *it->second;
  return *mapper->try_emplace(opDef, llvm::make_unique<Operator>(opDef))
              .first->second;
}

int tblgen::DagNode::getNumOps() const {
  int count = isReplaceWithValue() ? 0 : 1;
  for (int i = 0, e = getNumArgs(); i != e; ++i) {
    if (auto child = getArgAsNestedDag(i))
      count += child.getNumOps();
  }
  return count;
}

int tblgen::DagNode::getNumArgs() const { return node->getNumArgs(); }

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

bool tblgen::DagNode::isReplaceWithValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "replaceWithValue";
}

bool tblgen::DagNode::isVerifyUnusedValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "verifyUnusedValue";
}

tblgen::Pattern::Pattern(const llvm::Record *def, RecordOperatorMap *mapper)
    : def(*def), recordOpMap(mapper) {
  collectBoundArguments(getSourcePattern());
}

tblgen::DagNode tblgen::Pattern::getSourcePattern() const {
  return tblgen::DagNode(def.getValueAsDag("sourcePattern"));
}

int tblgen::Pattern::getNumResults() const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return results->size();
}

tblgen::DagNode tblgen::Pattern::getResultPattern(unsigned index) const {
  auto *results = def.getValueAsListInit("resultPatterns");
  return tblgen::DagNode(cast<llvm::DagInit>(results->getElement(index)));
}

void tblgen::Pattern::ensureBoundInSourcePattern(llvm::StringRef name) const {
  if (boundArguments.find(name) == boundArguments.end() &&
      boundOps.find(name) == boundOps.end())
    PrintFatalError(def.getLoc(),
                    Twine("referencing unbound variable '") + name + "'");
}

llvm::StringMap<tblgen::Argument> &
tblgen::Pattern::getSourcePatternBoundArgs() {
  return boundArguments;
}

llvm::StringSet<> &tblgen::Pattern::getSourcePatternBoundOps() {
  return boundOps;
}

const tblgen::Operator &tblgen::Pattern::getSourceRootOp() {
  return getSourcePattern().getDialectOp(recordOpMap);
}

tblgen::Operator &tblgen::Pattern::getDialectOp(DagNode node) {
  return node.getDialectOp(recordOpMap);
}

std::vector<tblgen::AppliedConstraint> tblgen::Pattern::getConstraints() const {
  auto *listInit = def.getValueAsListInit("constraints");
  std::vector<tblgen::AppliedConstraint> ret;
  ret.reserve(listInit->size());

  for (auto it : *listInit) {
    auto *dagInit = dyn_cast<llvm::DagInit>(it);
    if (!dagInit)
      PrintFatalError(def.getLoc(), "all elemements in Pattern multi-entity "
                                    "constraints should be DAG nodes");

    std::vector<std::string> entities;
    entities.reserve(dagInit->arg_size());
    for (auto *argName : dagInit->getArgNames())
      entities.push_back(argName->getValue());

    ret.emplace_back(cast<llvm::DefInit>(dagInit->getOperator())->getDef(),
                     std::move(entities));
  }
  return ret;
}

int tblgen::Pattern::getBenefit() const {
  // The initial benefit value is a heuristic with number of ops in the source
  // pattern.
  int initBenefit = getSourcePattern().getNumOps();
  llvm::DagInit *delta = def.getValueAsDag("benefitDelta");
  if (delta->getNumArgs() != 1 || !isa<llvm::IntInit>(delta->getArg(0))) {
    PrintFatalError(def.getLoc(),
                    "The 'addBenefit' takes and only takes one integer value");
  }
  return initBenefit + dyn_cast<llvm::IntInit>(delta->getArg(0))->getValue();
}

void tblgen::Pattern::collectBoundArguments(DagNode tree) {
  auto &op = getDialectOp(tree);
  auto numOpArgs = op.getNumArgs();
  auto numTreeArgs = tree.getNumArgs();

  if (numOpArgs != numTreeArgs) {
    PrintFatalError(def.getLoc(),
                    formatv("op '{0}' argument number mismatch: "
                            "{1} in pattern vs. {2} in definition",
                            op.getOperationName(), numTreeArgs, numOpArgs));
  }

  // The name attached to the DAG node's operator is for representing the
  // results generated from this op. It should be remembered as bound results.
  auto treeName = tree.getOpName();
  if (!treeName.empty())
    boundOps.insert(treeName);

  // TODO(jpienaar): Expand to multiple matches.
  for (int i = 0; i != numTreeArgs; ++i) {
    if (auto treeArg = tree.getArgAsNestedDag(i)) {
      // This DAG node argument is a DAG node itself. Go inside recursively.
      collectBoundArguments(treeArg);
    } else {
      auto treeArgName = tree.getArgName(i);
      if (!treeArgName.empty())
        boundArguments.try_emplace(treeArgName, op.getArg(i));
    }
  }
}
