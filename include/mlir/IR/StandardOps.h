//===- StandardOps.h - Standard MLIR Operations -----------------*- C++ -*-===//
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
// This file defines convenience types for working with standard operations
// in the MLIR instruction set.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_STANDARDOPS_H
#define MLIR_IR_STANDARDOPS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationImpl.h"

namespace mlir {
class OperationSet;

/// The "addf" operation takes two operands and returns one result, each of
/// these is required to be of the same type.  This type may be a floating point
/// scalar type, a vector whose element type is a floating point type, or a
/// floating point tensor. For example:
///
///   %2 = addf %0, %1 : f32
///
class AddFOp
    : public OpImpl::Base<AddFOp, OpImpl::TwoOperands, OpImpl::OneResult> {
public:
  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "addf"; }

  const char *verify() const;
  void print(raw_ostream &os) const;

private:
  friend class Operation;
  explicit AddFOp(const Operation *state) : Base(state) {}
};

/// The "constant" operation requires a single attribute named "value".
/// It returns its value as an SSA value.  For example:
///
///   %1 = "constant"(){value: 42} : i32
///   %2 = "constant"(){value: @foo} : (f32)->f32
///
class ConstantOp
    : public OpImpl::Base<ConstantOp, OpImpl::ZeroOperands, OpImpl::OneResult> {
public:
  Attribute *getValue() const { return getAttr("value"); }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "constant"; }

  // Hooks to customize behavior of this op.
  const char *verify() const;

protected:
  friend class Operation;
  explicit ConstantOp(const Operation *state) : Base(state) {}
};

/// This is a refinement of the "constant" op for the case where it is
/// returning an integer value.
///
///   %1 = "constant"(){value: 42}
///
class ConstantIntOp : public ConstantOp {
public:
  int64_t getValue() const {
    return getAttrOfType<IntegerAttr>("value")->getValue();
  }

  static bool isClassFor(const Operation *op);

private:
  friend class Operation;
  explicit ConstantIntOp(const Operation *state) : ConstantOp(state) {}
};

/// The "dim" operation takes a memref or tensor operand and returns an
/// "affineint".  It requires a single integer attribute named "index".  It
/// returns the size of the specified dimension.  For example:
///
///   %1 = dim %0, 2 : tensor<?x?x?xf32>
///
class DimOp
    : public OpImpl::Base<DimOp, OpImpl::OneOperand, OpImpl::OneResult> {
public:
  /// This returns the dimension number that the 'dim' is inspecting.
  unsigned getIndex() const {
    return (unsigned)getAttrOfType<IntegerAttr>("index")->getValue();
  }

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static StringRef getOperationName() { return "dim"; }

  // Hooks to customize behavior of this op.
  const char *verify() const;
  void print(raw_ostream &os) const;

private:
  friend class Operation;
  explicit DimOp(const Operation *state) : Base(state) {}
};

/// Install the standard operations in the specified operation set.
void registerStandardOperations(OperationSet &opSet);

} // end namespace mlir

#endif
