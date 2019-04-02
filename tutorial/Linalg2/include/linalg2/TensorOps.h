//===- TensorOps.h - Linalg dialect TensorOps operation definition --------===//
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

#ifndef LINALG2_TENSOROPS_H_
#define LINALG2_TENSOROPS_H_

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class AffineForOp;
} // namespace mlir

namespace linalg {

/// A generic TensorContraction base class which captures the generic behavior
/// of tensor contraction operations (with broadcast).
template <class ConcreteOp> class TensorContractionBase {
protected:
  using TensorContractionBaseType = TensorContractionBase<ConcreteOp>;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  /// Generic implementation of hooks that should be called from `ConcreteType`s
  mlir::LogicalResult verify();
  static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  TensorContractionBase() = default;
  mlir::Operation::operand_range getInputs();
  mlir::Operation::operand_range getOutputs();

public:
  /// These are better as methods calling into the ConcreteOp instead of
  /// template parameters because methods allow more generic behavior and avoid
  /// specializing for number of arguments. All derived classes have
  /// `VariadicOperands` and a build method from both an ArrayRef<mlirValue*>
  /// and the proper number of mlir::Value*.
  unsigned getNumInputs() {
    return static_cast<ConcreteOp *>(this)->numInputs;
  };
  unsigned getNumOutputs() {
    return static_cast<ConcreteOp *>(this)->numOutputs;
  };
  unsigned getNumParallelDims() {
    return static_cast<ConcreteOp *>(this)->numParallelDims;
  };
  unsigned getNumReductionDims() {
    return static_cast<ConcreteOp *>(this)->numReductionDims;
  };

  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  mlir::Value *getInputView(unsigned i);
  mlir::Value *getOutputView(unsigned i);
  /// Computes a mapping from all the ranges of the operands to the enclosing
  /// loops. In order to support "broadcast"-style semantics, we need to
  /// consider all the operands (i.e. input operands are not sufficient).
  /// The operands and their ranges are in the order defined by the particular
  /// ConcreteOp implementation, the resulting map must match those.
  /// This is currently computed but can also be specified explicitly in each
  /// operator to generalize to cases where an analysis is not available.
  mlir::AffineMap operandRangesToLoopsMap();
};

/// Implements c = A * B where c is a scalar and A and B are 1-D vectors.
class DotOp : public TensorContractionBase<DotOp>,
              public mlir::Op<DotOp, mlir::OpTrait::VariadicOperands,
                              mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;
  using TensorContractionBaseType =
      TensorContractionBase::TensorContractionBaseType;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.dot"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    llvm::ArrayRef<mlir::Value *> operands);
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *A, mlir::Value *B, mlir::Value *C) {
    return build(b, result, {A, B, C});
  }
  mlir::LogicalResult verify();
  static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numOutputs = 1;
  static constexpr unsigned numParallelDims = 0;
  static constexpr unsigned numReductionDims = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction();
};

/// Implements C = A * B where A is a 2-D matrix and X and Y are 1-D vectors.
class MatvecOp : public TensorContractionBase<MatvecOp>,
                 public mlir::Op<MatvecOp, mlir::OpTrait::VariadicOperands,
                                 mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;
  using TensorContractionBaseType =
      TensorContractionBase::TensorContractionBaseType;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.matvec"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    llvm::ArrayRef<mlir::Value *> operands);
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *A, mlir::Value *B, mlir::Value *C) {
    return build(b, result, {A, B, C});
  }
  mlir::LogicalResult verify();
  static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numOutputs = 1;
  static constexpr unsigned numParallelDims = 1;
  static constexpr unsigned numReductionDims = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction();
};

/// Implements C = A * B on 2-D matrices.
class MatmulOp : public TensorContractionBase<MatmulOp>,
                 public mlir::Op<MatmulOp, mlir::OpTrait::VariadicOperands,
                                 mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;
  using TensorContractionBaseType =
      TensorContractionBase::TensorContractionBaseType;

  //////////////////////////////////////////////////////////////////////////////
  // Hooks to customize the behavior of this op.
  //////////////////////////////////////////////////////////////////////////////
  static llvm::StringRef getOperationName() { return "linalg.matmul"; }
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    llvm::ArrayRef<mlir::Value *> operands);
  static void build(mlir::Builder *b, mlir::OperationState *result,
                    mlir::Value *A, mlir::Value *B, mlir::Value *C) {
    return build(b, result, {A, B, C});
  }
  mlir::LogicalResult verify();
  static bool parse(mlir::OpAsmParser *parser, mlir::OperationState *result);
  void print(mlir::OpAsmPrinter *p);

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  static constexpr unsigned numInputs = 2;
  static constexpr unsigned numOutputs = 1;
  static constexpr unsigned numParallelDims = 2;
  static constexpr unsigned numReductionDims = 1;

  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction();
};

} // namespace linalg

/// The TensorOp-inl.h inclusion pattern is chosen to allow gradual extension of
/// TensorOps by adding implementations as they are needed in the appropriate
/// step in the tutorial.
#include "linalg2/TensorOps-inl.h"

#endif // LINALG2_TENSOROPS_H_
