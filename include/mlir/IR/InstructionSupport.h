//===- InstructionSupport.h - MLIR Instruction Utilities --------*- C++ -*-===//
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
// This file defines utilities for the Instruction class.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_INSTRUCTION_SUPPORT_H
#define MLIR_IR_INSTRUCTION_SUPPORT_H

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/TrailingObjects.h"

namespace mlir {
class Instruction;

namespace detail {
/// A utility class holding the information necessary to dynamically resize
/// operands.
struct ResizableStorage {
  ResizableStorage(InstOperand *opBegin, unsigned numOperands)
      : firstOpAndIsDynamic(opBegin, false), capacity(numOperands) {}

  ~ResizableStorage() { cleanupStorage(); }

  /// Cleanup any allocated storage.
  void cleanupStorage() {
    // If the storage is dynamic, then we need to free the storage.
    if (isStorageDynamic())
      free(firstOpAndIsDynamic.getPointer());
  }

  /// Sets the storage pointer to a new dynamically allocated block.
  void setDynamicStorage(InstOperand *opBegin) {
    /// Cleanup the old storage if necessary.
    cleanupStorage();
    firstOpAndIsDynamic.setPointerAndInt(opBegin, true);
  }

  /// Returns the current storage pointer.
  InstOperand *getPointer() { return firstOpAndIsDynamic.getPointer(); }
  const InstOperand *getPointer() const {
    return firstOpAndIsDynamic.getPointer();
  }

  /// Returns if the current storage of operands is in the trailing objects is
  /// in a dynamically allocated memory block.
  bool isStorageDynamic() const { return firstOpAndIsDynamic.getInt(); }

  /// A pointer to the first operand element. This is either to the trailing
  /// objects storage, or a dynamically allocated block of memory.
  llvm::PointerIntPair<InstOperand *, 1, bool> firstOpAndIsDynamic;

  // The maximum number of operands that can be currently held by the storage.
  unsigned capacity;
};

/// This class handles the management of instruction operands. Operands are
/// stored similarly to the elements of a SmallVector except for two key
/// differences. The first is the inline storage, which is a trailing objects
/// array. The second is that being able to dynamically resize the operand list
/// is optional.
class OperandStorage final
    : private llvm::TrailingObjects<OperandStorage, ResizableStorage,
                                    InstOperand> {
public:
  OperandStorage(unsigned numOperands, bool resizable)
      : numOperands(numOperands), resizable(resizable) {
    // Initialize the resizable storage.
    if (resizable) {
      new (&getResizableStorage())
          ResizableStorage(getTrailingObjects<InstOperand>(), numOperands);
    }
  }

  ~OperandStorage() {
    // Manually destruct the operands.
    for (auto &operand : getInstOperands())
      operand.~InstOperand();

    // If the storage is resizable then destruct the utility.
    if (resizable)
      getResizableStorage().~ResizableStorage();
  }

  /// Replace the operands contained in the storage with the ones provided in
  /// 'operands'.
  void setOperands(Instruction *owner, ArrayRef<Value *> operands);

  /// Erase an operand held by the storage.
  void eraseOperand(unsigned index);

  /// Get the instruction operands held by the storage.
  ArrayRef<InstOperand> getInstOperands() const {
    return {getRawOperands(), size()};
  }
  MutableArrayRef<InstOperand> getInstOperands() {
    return {getRawOperands(), size()};
  }

  /// Return the number of operands held in the storage.
  unsigned size() const { return numOperands; }

  /// Returns the additional size necessary for allocating this object.
  static size_t additionalAllocSize(unsigned numOperands, bool resizable) {
    return additionalSizeToAlloc<ResizableStorage, InstOperand>(
        resizable ? 1 : 0, numOperands);
  }

  /// Returns if this storage is resizable.
  bool isResizable() const { return resizable; }

private:
  /// Clear the storage and destroy the current operands held by the storage.
  void clear() { numOperands = 0; }

  /// Returns the current pointer for the raw operands array.
  InstOperand *getRawOperands() {
    return resizable ? getResizableStorage().getPointer()
                     : getTrailingObjects<InstOperand>();
  }
  const InstOperand *getRawOperands() const {
    return resizable ? getResizableStorage().getPointer()
                     : getTrailingObjects<InstOperand>();
  }

  /// Returns the resizable operand utility class.
  ResizableStorage &getResizableStorage() {
    assert(resizable);
    return *getTrailingObjects<ResizableStorage>();
  }
  const ResizableStorage &getResizableStorage() const {
    assert(resizable);
    return *getTrailingObjects<ResizableStorage>();
  }

  /// Grow the internal resizable operand storage.
  void grow(ResizableStorage &resizeUtil, size_t minSize);

  /// The current number of operands, and the current max operand capacity.
  unsigned numOperands : 31;

  /// Whether this storage is resizable or not.
  bool resizable : 1;

  // This stuff is used by the TrailingObjects template.
  friend llvm::TrailingObjects<OperandStorage, ResizableStorage, InstOperand>;
  size_t numTrailingObjects(OverloadToken<ResizableStorage>) const {
    return resizable ? 1 : 0;
  }
};
} // end namespace detail
} // end namespace mlir

#endif // MLIR_IR_INSTRUCTION_SUPPORT_H
