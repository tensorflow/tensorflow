//===- MLFunction.h - MLIR MLFunction Class ---------------------*- C++ -*-===//
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
// This file defines MLFunction class
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_MLFUNCTION_H_
#define MLIR_IR_MLFUNCTION_H_

#include "mlir/IR/Function.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StmtBlock.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {

template <typename ObjectType, typename ElementType> class ArgumentIterator;

// MLFunction is defined as a sequence of statements that may
// include nested affine for loops, conditionals and operations.
class MLFunction final : public Function {
public:
  MLFunction(Location location, StringRef name, FunctionType type,
             ArrayRef<NamedAttribute> attrs = {});

  // TODO(clattner): drop this, it is redundant.
  static MLFunction *create(Location location, StringRef name,
                            FunctionType type,
                            ArrayRef<NamedAttribute> attrs = {}) {
    return new MLFunction(location, name, type, attrs);
  }

  StmtBlockList &getBlockList() { return body; }
  const StmtBlockList &getBlockList() const { return body; }

  StmtBlock *getBody() { return &body.front(); }
  const StmtBlock *getBody() const { return &body.front(); }

  //===--------------------------------------------------------------------===//
  // Arguments
  //===--------------------------------------------------------------------===//

  /// Returns number of arguments.
  unsigned getNumArguments() const { return getType().getInputs().size(); }

  /// Gets argument.
  BlockArgument *getArgument(unsigned idx) {
    return getBlockList().front().getArgument(idx);
  }

  const BlockArgument *getArgument(unsigned idx) const {
    return getBlockList().front().getArgument(idx);
  }

  // Supports non-const operand iteration.
  using args_iterator = ArgumentIterator<MLFunction, BlockArgument>;
  args_iterator args_begin();
  args_iterator args_end();
  llvm::iterator_range<args_iterator> getArguments();

  // Supports const operand iteration.
  using const_args_iterator =
      ArgumentIterator<const MLFunction, const BlockArgument>;
  const_args_iterator args_begin() const;
  const_args_iterator args_end() const;
  llvm::iterator_range<const_args_iterator> getArguments() const;

  //===--------------------------------------------------------------------===//
  // Other
  //===--------------------------------------------------------------------===//

  ~MLFunction();

  // Return the 'return' statement of this MLFunction.
  const OperationStmt *getReturnStmt() const;
  OperationStmt *getReturnStmt();

  /// Walk the statements in the function in preorder, calling the callback for
  /// each Operation statement.
  void walk(std::function<void(OperationStmt *)> callback);

  /// Walk the statements in the function in postorder, calling the callback for
  /// each Operation statement.
  void walkPostOrder(std::function<void(OperationStmt *)> callback);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Function *func) {
    return func->getKind() == Function::Kind::MLFunc;
  }

private:
  StmtBlockList body;
};

//===--------------------------------------------------------------------===//
// ArgumentIterator
//===--------------------------------------------------------------------===//

/// This template implements the argument iterator in terms of getArgument(idx).
template <typename ObjectType, typename ElementType>
class ArgumentIterator final
    : public IndexedAccessorIterator<ArgumentIterator<ObjectType, ElementType>,
                                     ObjectType, ElementType> {
public:
  /// Initializes the result iterator to the specified index.
  ArgumentIterator(ObjectType *object, unsigned index)
      : IndexedAccessorIterator<ArgumentIterator<ObjectType, ElementType>,
                                ObjectType, ElementType>(object, index) {}

  /// Support converting to the const variant. This will be a no-op for const
  /// variant.
  operator ArgumentIterator<const ObjectType, const ElementType>() const {
    return ArgumentIterator<const ObjectType, const ElementType>(this->object,
                                                                 this->index);
  }

  ElementType *operator*() const {
    return this->object->getArgument(this->index);
  }
};

//===--------------------------------------------------------------------===//
// MLFunction iterator methods.
//===--------------------------------------------------------------------===//

inline MLFunction::args_iterator MLFunction::args_begin() {
  return args_iterator(this, 0);
}

inline MLFunction::args_iterator MLFunction::args_end() {
  return args_iterator(this, getNumArguments());
}

inline llvm::iterator_range<MLFunction::args_iterator>
MLFunction::getArguments() {
  return {args_begin(), args_end()};
}

inline MLFunction::const_args_iterator MLFunction::args_begin() const {
  return const_args_iterator(this, 0);
}

inline MLFunction::const_args_iterator MLFunction::args_end() const {
  return const_args_iterator(this, getNumArguments());
}

inline llvm::iterator_range<MLFunction::const_args_iterator>
MLFunction::getArguments() const {
  return {args_begin(), args_end()};
}

} // end namespace mlir

#endif  // MLIR_IR_MLFUNCTION_H_
