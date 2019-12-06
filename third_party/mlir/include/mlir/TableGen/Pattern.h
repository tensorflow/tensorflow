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
#include "llvm/ADT/StringSet.h"

namespace llvm {
class DagInit;
class Init;
class Record;
} // end namespace llvm

namespace mlir {
namespace tblgen {

// Mapping from TableGen Record to Operator wrapper object.
//
// We allocate each wrapper object in heap to make sure the pointer to it is
// valid throughout the lifetime of this map. This is important because this map
// is shared among multiple patterns to avoid creating the wrapper object for
// the same op again and again. But this map will continuously grow.
using RecordOperatorMap =
    llvm::DenseMap<const llvm::Record *, std::unique_ptr<Operator>>;

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

  // Returns true if this DAG leaf is wrapping native code call.
  bool isNativeCodeCall() const;

  // Returns true if this DAG leaf is specifying a constant attribute.
  bool isConstantAttr() const;

  // Returns true if this DAG leaf is specifying an enum attribute case.
  bool isEnumAttrCase() const;

  // Returns this DAG leaf as a constraint. Asserts if fails.
  Constraint getAsConstraint() const;

  // Returns this DAG leaf as an constant attribute. Asserts if fails.
  ConstantAttr getAsConstantAttr() const;

  // Returns this DAG leaf as an enum attribute case.
  // Precondition: isEnumAttrCase()
  EnumAttrCase getAsEnumAttrCase() const;

  // Returns the matching condition template inside this DAG leaf. Assumes the
  // leaf is an operand/attribute matcher and asserts otherwise.
  std::string getConditionTemplate() const;

  // Returns the native code call template inside this DAG leaf.
  // Precondition: isNativeCodeCall()
  StringRef getNativeCodeTemplate() const;

private:
  // Returns true if the TableGen Init `def` in this DagLeaf is a DefInit and
  // also a subclass of the given `superclass`.
  bool isSubClassOf(StringRef superclass) const;

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

  // Returns the symbol bound to this DAG node.
  StringRef getSymbol() const;

  // Returns the operator wrapper object corresponding to the dialect op matched
  // by this DAG. The operator wrapper will be queried from the given `mapper`
  // and created in it if not existing.
  Operator &getDialectOp(RecordOperatorMap *mapper) const;

  // Returns the number of operations recursively involved in the DAG tree
  // rooted from this node.
  int getNumOps() const;

  // Returns the number of immediate arguments to this DAG node.
  int getNumArgs() const;

  // Returns true if the `index`-th argument is a nested DAG construct.
  bool isNestedDagArg(unsigned index) const;

  // Gets the `index`-th argument as a nested DAG construct if possible. Returns
  // null DagNode otherwise.
  DagNode getArgAsNestedDag(unsigned index) const;

  // Gets the `index`-th argument as a DAG leaf.
  DagLeaf getArgAsLeaf(unsigned index) const;

  // Returns the specified name of the `index`-th argument.
  StringRef getArgName(unsigned index) const;

  // Returns true if this DAG construct means to replace with an existing SSA
  // value.
  bool isReplaceWithValue() const;

  // Returns true if this DAG node is wrapping native code call.
  bool isNativeCodeCall() const;

  // Returns true if this DAG node is an operation.
  bool isOperation() const;

  // Returns the native code call template inside this DAG node.
  // Precondition: isNativeCodeCall()
  StringRef getNativeCodeTemplate() const;

private:
  const llvm::DagInit *node; // nullptr means null DagNode
};

// A class for maintaining information for symbols bound in patterns and
// provides methods for resolving them according to specific use cases.
//
// Symbols can be bound to
//
// * Op arguments and op results in the source pattern and
// * Op results in result patterns.
//
// Symbols can be referenced in result patterns and additional constraints to
// the pattern.
//
// For example, in
//
// ```
// def : Pattern<
//     (SrcOp:$results1 $arg0, %arg1),
//     [(ResOp1:$results2), (ResOp2 $results2 (ResOp3 $arg0, $arg1))]>;
// ```
//
// `$argN` is bound to the `SrcOp`'s N-th argument. `$results1` is bound to
// `SrcOp`. `$results2` is bound to `ResOp1`. $result2 is referenced to build
// `ResOp2`. `$arg0` and `$arg1` are referenced to build `ResOp3`.
//
// If a symbol binds to a multi-result op and it does not have the `__N`
// suffix, the symbol is expanded to represent all results generated by the
// multi-result op. If the symbol has a `__N` suffix, then it will expand to
// only the N-th *static* result as declared in ODS, and that can still
// corresponds to multiple *dynamic* values if the N-th *static* result is
// variadic.
//
// This class keeps track of such symbols and resolves them into their bound
// values in a suitable way.
class SymbolInfoMap {
public:
  explicit SymbolInfoMap(ArrayRef<llvm::SMLoc> loc) : loc(loc) {}

  // Class for information regarding a symbol.
  class SymbolInfo {
  public:
    // Returns a string for defining a variable named as `name` to store the
    // value bound by this symbol.
    std::string getVarDecl(StringRef name) const;

  private:
    // Allow SymbolInfoMap to access private methods.
    friend class SymbolInfoMap;

    // What kind of entity this symbol represents:
    // * Attr: op attribute
    // * Operand: op operand
    // * Result: op result
    // * Value: a value not attached to an op (e.g., from NativeCodeCall)
    enum class Kind : uint8_t { Attr, Operand, Result, Value };

    // Creates a SymbolInfo instance. `index` is only used for `Attr` and
    // `Operand` so should be negative for `Result` and `Value` kind.
    SymbolInfo(const Operator *op, Kind kind, Optional<int> index);

    // Static methods for creating SymbolInfo.
    static SymbolInfo getAttr(const Operator *op, int index) {
      return SymbolInfo(op, Kind::Attr, index);
    }
    static SymbolInfo getOperand(const Operator *op, int index) {
      return SymbolInfo(op, Kind::Operand, index);
    }
    static SymbolInfo getResult(const Operator *op) {
      return SymbolInfo(op, Kind::Result, llvm::None);
    }
    static SymbolInfo getValue() {
      return SymbolInfo(nullptr, Kind::Value, llvm::None);
    }

    // Returns the number of static values this symbol corresponds to.
    // A static value is an operand/result declared in ODS. Normally a symbol
    // only represents one static value, but symbols bound to op results can
    // represent more than one if the op is a multi-result op.
    int getStaticValueCount() const;

    // Returns a string containing the C++ expression for referencing this
    // symbol as a value (if this symbol represents one static value) or a value
    // range (if this symbol represents multiple static values). `name` is the
    // name of the C++ variable that this symbol bounds to. `index` should only
    // be used for indexing results.  `fmt` is used to format each value.
    // `separator` is used to separate values if this is a value range.
    std::string getValueAndRangeUse(StringRef name, int index, const char *fmt,
                                    const char *separator) const;

    // Returns a string containing the C++ expression for referencing this
    // symbol as a value range regardless of how many static values this symbol
    // represents. `name` is the name of the C++ variable that this symbol
    // bounds to. `index` should only be used for indexing results. `fmt` is
    // used to format each value. `separator` is used to separate values in the
    // range.
    std::string getAllRangeUse(StringRef name, int index, const char *fmt,
                               const char *separator) const;

    const Operator *op; // The op where the bound entity belongs
    Kind kind;          // The kind of the bound entity
    // The argument index (for `Attr` and `Operand` only)
    Optional<int> argIndex;
  };

  using BaseT = llvm::StringMap<SymbolInfo>;

  // Iterators for accessing all symbols.
  using iterator = BaseT::iterator;
  iterator begin() { return symbolInfoMap.begin(); }
  iterator end() { return symbolInfoMap.end(); }

  // Const iterators for accessing all symbols.
  using const_iterator = BaseT::const_iterator;
  const_iterator begin() const { return symbolInfoMap.begin(); }
  const_iterator end() const { return symbolInfoMap.end(); }

  // Binds the given `symbol` to the `argIndex`-th argument to the given `op`.
  // Returns false if `symbol` is already bound.
  bool bindOpArgument(StringRef symbol, const Operator &op, int argIndex);

  // Binds the given `symbol` to the results the given `op`. Returns false if
  // `symbol` is already bound.
  bool bindOpResult(StringRef symbol, const Operator &op);

  // Registers the given `symbol` as bound to a value. Returns false if `symbol`
  // is already bound.
  bool bindValue(StringRef symbol);

  // Returns true if the given `symbol` is bound.
  bool contains(StringRef symbol) const;

  // Returns an iterator to the information of the given symbol named as `key`.
  const_iterator find(StringRef key) const;

  // Returns the number of static values of the given `symbol` corresponds to.
  // A static value is a operand/result declared in ODS. Normally a symbol only
  // represents one static value, but symbols bound to op results can represent
  // more than one if the op is a multi-result op.
  int getStaticValueCount(StringRef symbol) const;

  // Returns a string containing the C++ expression for referencing this
  // symbol as a value (if this symbol represents one static value) or a value
  // range (if this symbol represents multiple static values). `fmt` is used to
  // format each value. `separator` is used to separate values if `symbol`
  // represents a value range.
  std::string getValueAndRangeUse(StringRef symbol, const char *fmt = "{0}",
                                  const char *separator = ", ") const;

  // Returns a string containing the C++ expression for referencing this
  // symbol as a value range regardless of how many static values this symbol
  // represents. `fmt` is used to format each value. `separator` is used to
  // separate values in the range.
  std::string getAllRangeUse(StringRef symbol, const char *fmt = "{0}",
                             const char *separator = ", ") const;

  // Splits the given `symbol` into a value pack name and an index. Returns the
  // value pack name and writes the index to `index` on sucess. Returns `symbol`
  // itself if it does not contain an index.
  //
  // We can use `name__N` to access the `N`-th value in the value pack bound to
  // `name`. `name` is typically the results of an multi-result op.
  static StringRef getValuePackName(StringRef symbol, int *index = nullptr);

private:
  llvm::StringMap<SymbolInfo> symbolInfoMap;

  // Pattern instantiation location. This is intended to be used as parameter
  // to PrintFatalError() to report errors.
  ArrayRef<llvm::SMLoc> loc;
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

  // Returns the number of result patterns generated by applying this rewrite
  // rule.
  int getNumResultPatterns() const;

  // Returns the DAG tree root node of the `index`-th result pattern.
  DagNode getResultPattern(unsigned index) const;

  // Collects all symbols bound in the source pattern into `infoMap`.
  void collectSourcePatternBoundSymbols(SymbolInfoMap &infoMap);

  // Collects all symbols bound in result patterns into `infoMap`.
  void collectResultPatternBoundSymbols(SymbolInfoMap &infoMap);

  // Returns the op that the root node of the source pattern matches.
  const Operator &getSourceRootOp();

  // Returns the operator wrapper object corresponding to the given `node`'s DAG
  // operator.
  Operator &getDialectOp(DagNode node);

  // Returns the constraints.
  std::vector<AppliedConstraint> getConstraints() const;

  // Returns the benefit score of the pattern.
  int getBenefit() const;

  using IdentifierLine = std::pair<StringRef, unsigned>;

  // Returns the file location of the pattern (buffer identifier + line number
  // pair).
  std::vector<IdentifierLine> getLocation() const;

private:
  // Recursively collects all bound symbols inside the DAG tree rooted
  // at `tree` and updates the given `infoMap`.
  void collectBoundSymbols(DagNode tree, SymbolInfoMap &infoMap,
                           bool isSrcPattern);

  // The TableGen definition of this pattern.
  const llvm::Record &def;

  // All operators.
  // TODO(antiagainst): we need a proper context manager, like MLIRContext,
  // for managing the lifetime of shared entities.
  RecordOperatorMap *recordOpMap;
};

} // end namespace tblgen
} // end namespace mlir

#endif // MLIR_TABLEGEN_PATTERN_H_
