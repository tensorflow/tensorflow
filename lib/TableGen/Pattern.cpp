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

bool tblgen::DagArg::isAttr() const {
  return arg.is<tblgen::NamedAttribute *>();
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

llvm::DefInit *tblgen::DagNode::getArgAsDefInit(unsigned index) const {
  return dyn_cast<llvm::DefInit>(node->getArg(index));
}

StringRef tblgen::DagNode::getArgName(unsigned index) const {
  return node->getArgNameStr(index);
}

static void collectBoundArguments(const llvm::DagInit *tree,
                                  tblgen::Pattern *pattern) {
  auto &op = pattern->getDialectOp(tblgen::DagNode(tree));

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

    pattern->getSourcePatternBoundArgs().try_emplace(name, op.getArg(i), arg);
  }
}

void tblgen::DagNode::collectBoundArguments(tblgen::Pattern *pattern) const {
  ::collectBoundArguments(node, pattern);
}

bool tblgen::DagNode::isReplaceWithValue() const {
  auto *dagOpDef = cast<llvm::DefInit>(node->getOperator())->getDef();
  return dagOpDef->getName() == "replaceWithValue";
}

tblgen::Pattern::Pattern(const llvm::Record *def, RecordOperatorMap *mapper)
    : def(*def), recordOpMap(mapper) {
  getSourcePattern().collectBoundArguments(this);
}

tblgen::DagNode tblgen::Pattern::getSourcePattern() const {
  return tblgen::DagNode(def.getValueAsDag("PatternToMatch"));
}

unsigned tblgen::Pattern::getNumResults() const {
  auto *results = def.getValueAsListInit("ResultOps");
  return results->size();
}

tblgen::DagNode tblgen::Pattern::getResultPattern(unsigned index) const {
  auto *results = def.getValueAsListInit("ResultOps");
  return tblgen::DagNode(cast<llvm::DagInit>(results->getElement(index)));
}

void tblgen::Pattern::ensureArgBoundInSourcePattern(
    llvm::StringRef name) const {
  if (boundArguments.find(name) == boundArguments.end())
    PrintFatalError(def.getLoc(),
                    Twine("referencing unbound variable '") + name + "'");
}

llvm::StringMap<tblgen::DagArg> &tblgen::Pattern::getSourcePatternBoundArgs() {
  return boundArguments;
}

const tblgen::Operator &tblgen::Pattern::getSourceRootOp() {
  return getSourcePattern().getDialectOp(recordOpMap);
}

tblgen::Operator &tblgen::Pattern::getDialectOp(DagNode node) {
  return node.getDialectOp(recordOpMap);
}
