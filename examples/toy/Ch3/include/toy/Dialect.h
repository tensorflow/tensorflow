//===- AST.h - Node definition for the Toy AST ----------------------------===//
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
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure where each node
// references its children using std::unique_ptr<>.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
#define MLIR_TUTORIAL_TOY_DIALECT_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

namespace mlir {
class FuncBuilder;
}

namespace toy {

/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and register custom operations and types (in its constructor).
/// It can also overridding general behavior of dialects exposed as virtual
/// method, for example regarding verification and parsing/printing.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Parse a type registered to this dialect. Overridding this method is
  /// required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  mlir::Type parseType(llvm::StringRef tyData,
                       mlir::Location loc) const override;

  /// Print a type registered to this dialect. Overridding this method is
  /// only required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  void printType(mlir::Type type, llvm::raw_ostream &os) const override;
};

////////////////////////////////////////////////////////////////////////////////
/////////////////////// Custom Types for the Dialect ///////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {
struct ToyArrayTypeStorage;
}

/// LLVM-style RTTI: one entry per subclass to allow dyn_cast/isa.
enum ToyTypeKind {
  // The enum starts at the range reserved for this dialect.
  TOY_TYPE = mlir::OpaqueType::FIRST_TOY_TYPE,
  TOY_ARRAY,
};

/// Type for Toy arrays.
/// In MLIR Types are reference to immutable and uniqued objects owned by the
/// MLIRContext. As such `ToyArrayType` only wraps a pointer to an uniqued
/// instance of `ToyArrayTypeStorage` (defined in our implementation file) and
/// provide the public facade API to interact with the type.
class ToyArrayType : public mlir::Type::TypeBase<ToyArrayType, mlir::Type,
                                                 detail::ToyArrayTypeStorage> {
public:
  using Base::Base;

  /// Returns the dimensions for this array, or and empty range for a generic
  /// array.
  llvm::ArrayRef<int64_t> getShape();

  /// Predicate to test if this array is generic (shape haven't been inferred
  /// yet).
  bool isGeneric() { return getShape().empty(); }

  /// Return the rank of this array (0 if it is generic)
  int getRank() { return getShape().size(); }

  /// Return the type of individual elements in the array.
  mlir::Type getElementType();

  /// Get the unique instance of this Type from the context.
  /// A ToyArrayType is only defined by the shape of the array.
  static ToyArrayType get(mlir::MLIRContext *context,
                          llvm::ArrayRef<int64_t> shape = {});

  /// Support method to enable LLVM-style RTTI type casting.
  static bool kindof(unsigned kind) { return kind == ToyTypeKind::TOY_ARRAY; }
};

////////////////////////////////////////////////////////////////////////////////
//////////////////// Custom Operations for the Dialect /////////////////////////
////////////////////////////////////////////////////////////////////////////////

/// Constant operation turns a literal into an SSA value. The data is attached
/// to the operation as an attribute. For example:
///
///   %0 = "toy.constant"()
///       {value: dense<tensor<2x3xf64>, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>}
///     : () -> !toy<"array<2, 3>">
///
/// An operation inherit from `class Op` and specify optional traits. Here we
/// indicate that `toy.constant` does not have any operand and return a single
/// result. The traits are making some methods available on the operation, for
/// instance we will be able to use `getResult()` but `getOperand()` won't be
/// available.
class ConstantOp : public mlir::Op<ConstantOp, mlir::OpTrait::ZeroOperands,
                                   mlir::OpTrait::OneResult,
                                   mlir::OpTrait::HasNoSideEffect> {
public:
  /// This is the name used by MLIR to match an operation to this class during
  /// parsing.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Operation can have extra verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<PrintOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.constant` operation does not have arguments but attaches a
  /// constant array as attribute and returns it as an SSA value.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    llvm::ArrayRef<int64_t> shape,
                    mlir::DenseElementsAttr value);

  /// Similar to the one above, but takes a single float and returns a
  /// !toy<"array<1>">.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::FloatAttr value);

  /// Inherit Constructor
  using Op::Op;
};

/// Generic calls are representing calls to a user defined function that need to
/// be specialized for the shape of its arguments. The callee name is attached
/// as a literal string as an attribute. The arguments list must match the
/// arguments expected by the callee. For example:
///
///   %4 = "toy.generic_call"(%1, %3) {callee: "my_func"}
///         : (!toy<"array<2, 3>">, !toy<"array<2, 3>">) -> !toy<"array">
///
/// This is only valid if a function named "my_func" exists and takes two
/// arguments.
class GenericCallOp
    : public mlir::Op<GenericCallOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {
public:
  /// MLIR will use this to register the operation with the parser/printer.
  static llvm::StringRef getOperationName() { return "toy.generic_call"; }

  /// Operations can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to the builder to allow:
  ///   mlir::FuncBuilder::create<GenericCallOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.generic_call` operation accepts a callee name and a list of
  /// arguments for the call.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    llvm::StringRef callee,
                    llvm::ArrayRef<mlir::Value *> arguments);

  /// Return the name of the callee.
  llvm::StringRef getCalleeName();

  /// Inherit Constructor
  using Op::Op;
};

/// Return operation terminates blocks (and function as well). They take a
/// single argument and the type must match the function return type.
class ReturnOp
    : public mlir::Op<ReturnOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::ZeroResult, mlir::OpTrait::IsTerminator> {
public:
  static llvm::StringRef getOperationName() { return "toy.return"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<PrintOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.return` operation accepts an optional single array as argument
  /// and does not have any returned value.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *value = nullptr);

  /// Return true if there is a returned value.
  bool hasOperand() { return 0 != getNumOperands(); }

  /// Helper to return the optional operand. Caller must check if the operand
  /// is present before calling this.
  mlir::Value *getOperand() { return getOperation()->getOperand(0); }

  /// Inherit Constructor
  using Op::Op;
};

/// The print builtin takes a single array argument and does not return any.
class PrintOp : public mlir::Op<PrintOp, mlir::OpTrait::OneOperand,
                                mlir::OpTrait::ZeroResult> {
public:
  static llvm::StringRef getOperationName() { return "toy.print"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<PrintOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.print` operation accepts a single array as argument and does
  /// not have any returned value.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *value);

  /// Inherit Constructor
  using Op::Op;
};

class TransposeOp : public mlir::Op<TransposeOp, mlir::OpTrait::OneOperand,
                                    mlir::OpTrait::OneResult> {
public:
  static llvm::StringRef getOperationName() { return "toy.transpose"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<TransposeOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.transpose` operation accepts a single array as argument and
  /// returns the transposed array as its only result.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *value);

  /// Inherit Constructor
  using Op::Op;
};

/// Reshape operation is transforming its input array into a new array with the
/// same number of elements but different shapes. For example:
///
///    %0 = "toy.transpose"(%arg1) : (!toy<"array<10>">) -> !toy<"array<5, 2>">
///
class ReshapeOp : public mlir::Op<ReshapeOp, mlir::OpTrait::OneOperand,
                                  mlir::OpTrait::OneResult> {
public:
  static llvm::StringRef getOperationName() { return "toy.reshape"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<ReshapeOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.reshape` operation accepts a single array as argument and
  /// returns the array with the specified reshapedType as its only result.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *value, ToyArrayType reshapedType);

  /// Inherit Constructor
  using Op::Op;
};

/// Binary operation implementing a multiplication. For two-dimensional array
/// a matrix multiplication is implemented, while for one dimensional array a
/// dot product is performed.
class MulOp : public mlir::Op<MulOp, mlir::OpTrait::NOperands<2>::Impl,
                              mlir::OpTrait::OneResult> {
public:
  static llvm::StringRef getOperationName() { return "toy.mul"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<PrintOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.mul` operation accepts two operands as argument and returns
  /// a single value.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *lhs, mlir::Value *rhs);

  /// Inherit Constructor
  using Op::Op;
};

/// Element wise addition of two arrays. The shape must match.
class AddOp : public mlir::Op<AddOp, mlir::OpTrait::NOperands<3>::Impl,
                              mlir::OpTrait::OneResult> {
public:
  static llvm::StringRef getOperationName() { return "toy.add"; }

  /// Operation can add custom verification beyond the traits they define.
  mlir::LogicalResult verify();

  /// Interface to mlir::FuncBuilder::create<PrintOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.mul` operation accepts two operands as argument and returns
  /// a single value.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    mlir::Value *lhs, mlir::Value *rhs);

  /// Inherit Constructor
  using Op::Op;
};

} // end namespace toy

#endif // MLIR_TUTORIAL_TOY_DIALECT_H_
