//===- Pattern.h - Pattern wrapper class ------------------------*- C++ -*-===//
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

#ifndef MLIR_TABLEGEN_PATTERN_H_
#define MLIR_TABLEGEN_PATTERN_H_

#include "mlir/Support/LLVM.h"
#include "mlir/TableGen/Argument.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/TableGen/Error.h"

namespace llvm {
class DagInit;
class Init;
class Record;
class StringRef;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Mapping from TableGen Record to Operator wrapper object
using RecordOperatorMap = llvm::DenseMap<const llvm::Record *, Operator>;

class Pattern;

// Wrapper class providing helper methods for accessing TableGen DAG leaves
// used inside Patterns. This class is lightweight and designed to be used like
// values.
//
// A TableGen DAG construct is of the syntax
//   `(operator, arg0, arg1, ...)`.
//
// This class provides getters to retrieve `arg*` as tblgen:: wrapper objects
// for handy helper methods. It only works on `arg*`s that are not nested DAG
// constructs.
class DagLeaf {
public:
  explicit DagLeaf(const llvm::Init *def) : def(def) {}

  // Returns true if this DAG leaf is not specified in the pattern. That is, it
  // places no further constraints/transforms and just carries over the original
  // value.
  bool isUnspecified() const;

  // Returns true if this DAG leaf is matching an operand. That is, it specifies
  // a type constraint.
  bool isOperandMatcher() const;

  // Returns true if this DAG leaf is matching an attribute. That is, it
  // specifies an attribute constraint.
  bool isAttrMatcher() const;

  // Returns true if this DAG leaf is transforming an attribute.
  bool isAttrTransformer() const;

  // Returns true if this DAG leaf is specifying a constant attribute.
  bool isConstantAttr() const;

  // Returns this DAG leaf as a type constraint. Asserts if fails.
  TypeConstraint getAsTypeConstraint() const;

  // Returns this DAG leaf as an attribute constraint. Asserts if fails.
  AttrConstraint getAsAttrConstraint() const;

  // Returns this DAG leaf as an constant attribute. Asserts if fails.
  ConstantAttr getAsConstantAttr() const;

  // Returns the matching condition template inside this DAG leaf. Assumes the
  // leaf is an operand/attribute matcher and asserts otherwise.
  std::string getConditionTemplate() const;

  // Returns the transformation template inside this DAG leaf. Assumes the
  // leaf is an attribute matcher and asserts otherwise.
  std::string getTransformationTemplate() const;

private:
  const llvm::Init *def;
};

// Wrapper class providing helper methods for accessing TableGen DAG constructs
// used inside Patterns. This class is lightweight and designed to be used like
// values.
//
// A TableGen DAG construct is of the syntax
//   `(operator, arg0, arg1, ...)`.
//
// When used inside Patterns, `operator` corresponds to some dialect op, or
// a known list of verbs that defines special transformation actions. This
// `arg*` can be a nested DAG construct. This class provides getters to
// retrieve `operator` and `arg*` as tblgen:: wrapper objects for handy helper
// methods.
//
// A null DagNode contains a nullptr and converts to false implicitly.
class DagNode {
public:
  explicit DagNode(const llvm::DagInit *node) : node(node) {}

  // Implicit bool converter that returns true if this DagNode is not a null
  // DagNode.
  operator bool() const { return node != nullptr; }

  // Returns the operator wrapper object corresponding to the dialect op matched
  // by this DAG. The operator wrapper will be queried from the given `mapper`
  // and created in it if not existing.
  Operator &getDialectOp(RecordOperatorMap *mapper) const;

  // Returns the number of operations recursively involved in the DAG tree
  // rooted from this node.
  unsigned getNumOps() const;

  // Returns the number of immediate arguments to this DAG node.
  unsigned getNumArgs() const;

  // Returns true if the `index`-th argument is a nested DAG construct.
  bool isNestedDagArg(unsigned index) const;

  // Gets the `index`-th argument as a nested DAG construct if possible. Returns
  // null DagNode otherwise.
  DagNode getArgAsNestedDag(unsigned index) const;

  // Gets the `index`-th argument as a DAG leaf.
  DagLeaf getArgAsLeaf(unsigned index) const;

  // Returns the specified name of the `index`-th argument.
  llvm::StringRef getArgName(unsigned index) const;

  // Returns the native builder for the pattern.
  // Precondition: isNativeCodeBuilder.
  llvm::StringRef getNativeCodeBuilder() const;

  // Collects all recursively bound arguments involved in the DAG tree rooted
  // from this node.
  void collectBoundArguments(Pattern *pattern) const;

  // Returns true if this DAG construct means to replace with an existing SSA
  // value.
  bool isReplaceWithValue() const;

  // Returns true if this DAG construct is meant to invoke a native code
  // constructor.
  bool isNativeCodeBuilder() const;

private:
  const llvm::DagInit *node; // nullptr means null DagNode
};

// Wrapper class providing helper methods for accessing MLIR Pattern defined
// in TableGen. This class should closely reflect what is defined as class
// `Pattern` in TableGen. This class contains maps so it is not intended to be
// used as values.
class Pattern {
public:
  explicit Pattern(const llvm::Record *def, RecordOperatorMap *mapper);

  // Returns the source pattern to match.
  DagNode getSourcePattern() const;

  // Returns the number of results generated by applying this rewrite pattern.
  unsigned getNumResults() const;

  // Returns the DAG tree root node of the `index`-th result pattern.
  DagNode getResultPattern(unsigned index) const;

  // Checks whether an argument with the given `name` is bound in source
  // pattern. Prints fatal error if not; does nothing otherwise.
  void ensureArgBoundInSourcePattern(llvm::StringRef name) const;

  // Returns a reference to all the bound arguments in the source pattern.
  llvm::StringMap<Argument> &getSourcePatternBoundArgs();

  // Returns the op that the root node of the source pattern matches.
  const Operator &getSourceRootOp();

  // Returns the operator wrapper object corresponding to the given `node`'s DAG
  // operator.
  Operator &getDialectOp(DagNode node);

private:
  // The TableGen definition of this pattern.
  const llvm::Record &def;

  // All operators
  RecordOperatorMap *recordOpMap;
  // All bound arguments
  llvm::StringMap<Argument> boundArguments;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PATTERN_H_
