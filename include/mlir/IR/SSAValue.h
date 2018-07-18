//===- SSAValue.h - Base of the value hierarchy -----------------*- C++ -*-===//
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
// This file defines generic SSAValue type and manipulation utilities.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_SSAVALUE_H
#define MLIR_IR_SSAVALUE_H

#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/iterator_range.h"

namespace mlir {
template <typename SSAOperandTy> class SSAValueUseIterator;

/// This is the base class for the MLIR value hierarchy, representing a
/// computable value that has a type etc.  This is a template that is used by
/// both the CFG and ML side of the MLIR representation.
///
/// SSAOperandTy is the concrete instance of SSAOperand to use (including
/// substituted template arguments) and KindTy is the enum 'kind' discriminator
/// that subclasses want to use.
///
template <typename SSAOperandTy, typename KindTy>
class SSAValue {
protected:
  SSAValue(KindTy kind, Type *type) : typeAndKind(type, kind) {}

public:
  ~SSAValue() {
    assert(use_empty() && "Cannot destroy a value that still has uses!");
  }

  KindTy getKind() const { return typeAndKind.getInt(); }

  Type *getType() const { return typeAndKind.getPointer(); }

  /// Replace every use of this value with the corresponding value 'newVal'.
  ///
  void replaceAllUsesWith(SSAValue *newVal);

  /// Returns true if this value has no uses.
  bool use_empty() const { return firstUse == nullptr; }

  using use_iterator = SSAValueUseIterator<SSAOperandTy>;
  using use_range = llvm::iterator_range<use_iterator>;

  inline use_iterator use_begin() const;
  inline use_iterator use_end() const;

  /// Returns a range of all uses, which is useful for iterating over all uses.
  inline use_range getUses() const;

  /// Returns true if this value has exactly one use.
  inline bool hasOneUse() const;

private:
  friend SSAOperandTy;
  const llvm::PointerIntPair<Type *, 3, KindTy> typeAndKind;
  SSAOperandTy *firstUse = nullptr;
};

} // namespace mlir

#endif
