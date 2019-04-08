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
class TensorContractionBase {
public:
  virtual ~TensorContractionBase() {}

  //////////////////////////////////////////////////////////////////////////////
  // Op-specific functionality.
  //////////////////////////////////////////////////////////////////////////////
  virtual llvm::StringRef getTensorContractionName() = 0;
  mlir::Operation::operand_range getInputs();
  mlir::Operation::operand_range getOutputs();
  mlir::Operation::operand_range getInputsAndOutputs() {
    return {getInputs().begin(), getOutputs().end()};
  }

  /// These are better as methods calling into the ConcreteOp instead of
  /// template parameters because methods allow more generic behavior and avoid
  /// specializing for number of arguments. All derived classes have
  /// `VariadicOperands` and a build method from both an ArrayRef<mlirValue*>
  /// and the proper number of mlir::Value*.
  virtual unsigned getNumInputs() = 0;
  virtual unsigned getNumOutputs() = 0;
  virtual unsigned getNumParallelDims() = 0;
  virtual unsigned getNumReductionDims() = 0;

  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  mlir::Value *getInputView(unsigned viewIndex);
  mlir::Value *getOutputView(unsigned viewIndex);
  mlir::Value *getView(unsigned viewIndex) {
    return viewIndex < getNumInputs()
               ? getInputView(viewIndex)
               : getOutputView(viewIndex - getNumInputs());
  }

  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  virtual void writeAsFinerGrainTensorContraction() {}

  /// Each op is responsible for declaring how it lowers itself to scalar form,
  /// given the enclosing parallel and reduction induction variables.
  /// `emitScalarImplementation` emits the scalar IR for the op in the nesting
  /// context of the innermost enclosing loop(i.e. `reductionIvs.back()` or
  /// `parallel.back()`).
  virtual void
  emitScalarImplementation(llvm::ArrayRef<mlir::Value *> parallelIvs,
                           llvm::ArrayRef<mlir::Value *> reductionIvs) {}

  /// Represents a mapping from the loops to all the ranges of the operands.
  /// The operands and their ranges are in the order defined by the particular
  /// ConcreteOp implementation, the resulting map must match those.
  /// In favorable cases, this can be calculated by an analysis but specifying
  /// it explicitly is not expensive and generalizes to cases where an analysis
  /// is not available. For details, see the description of
  /// loopsToOperandRangeMaps in each ConcreteOp.
  virtual llvm::SmallVector<mlir::AffineMap, 8> loopsToOperandRangeMaps() {
    return llvm::SmallVector<mlir::AffineMap, 8>();
  }
};

/// Implements c = A * B where c is a scalar and A and B are 1-D vectors.
class DotOp : public TensorContractionBase,
              public mlir::Op<DotOp, mlir::OpTrait::VariadicOperands,
                              mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

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
  llvm::StringRef getTensorContractionName() override {
    return getOperationName();
  }
  unsigned getNumInputs() override { return 2; }
  unsigned getNumOutputs() override { return 1; }
  unsigned getNumParallelDims() override { return 0; }
  unsigned getNumReductionDims() override { return 1; }

#if LINALG_STEP > 2
  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction() override;

  /// Inputs to this map will be (%k) coming from enclosing loops.
  /// Therefore, the mapping to get back to A(K), B(K), C() is:
  ///   (d0) -> (d0, d0)(%k)
  /// And the operands ranges are:
  ///   (%k, %k)
  llvm::SmallVector<mlir::AffineMap, 8> loopsToOperandRangeMaps() override;

  ///  Given an enclosing reduction loop with iv `r_i`, emits MLIR corresponding
  ///  to:
  ///    1. conditionally assign scalarC to 0.0f on the first iteration or load
  ///       C[] from memory (0-D tensor)
  ///    2. multiply A[r_i] by B[r_i] and add to scalarC
  ///    3. store back scalarC at C[]
  ///
  /// In some compact index notation this could be written:
  ///  cond = (r_i == zero)
  ///  scalarC = select(cond, zerof, C[]);
  ///  C[] = scalarC + A[r_i] * B[r_i];
  void
  emitScalarImplementation(llvm::ArrayRef<mlir::Value *> parallelIvs,
                           llvm::ArrayRef<mlir::Value *> reductionIvs) override;
#endif // LINALG_STEP
};

/// Implements C = A * B where A is a 2-D matrix and X and Y are 1-D vectors.
class MatvecOp : public TensorContractionBase,
                 public mlir::Op<MatvecOp, mlir::OpTrait::VariadicOperands,
                                 mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

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
  llvm::StringRef getTensorContractionName() override {
    return getOperationName();
  }
  unsigned getNumInputs() override { return 2; }
  unsigned getNumOutputs() override { return 1; }
  unsigned getNumParallelDims() override { return 1; }
  unsigned getNumReductionDims() override { return 1; }

#if LINALG_STEP > 2
  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction() override;

  /// Inputs to this map will be (%m, %k) coming from enclosing loops.
  /// Therefore, the mapping to get back to A(M, K), B(K), C(M) is:
  ///   (d0, d1) -> (d0, d1, d1, d0)(%m, %k)
  /// And the operands ranges are:
  ///   (%m, %k, %k, %m)
  llvm::SmallVector<mlir::AffineMap, 8> loopsToOperandRangeMaps() override;

  ///  Given an enclosing parallel loop with iv `i` and an enclosing parallel
  ///  loop with iv `r_j`, emits MLIR corresponding to:
  ///    1. conditionally assign scalarC to 0.0f on the first iteration or load
  ///       C[i]
  ///    2. multiply A[i, r_j] by B[r_j] and add to scalarC
  ///    3. store back scalarC at C[i]
  ///
  /// In some compact index notation this could be written:
  ///  cond = (r_j == zero)
  ///  scalarC = select(cond, zerof, C(i));
  ///  C(i) = scalarC + A(i, r_j) * B(r_j);
  void
  emitScalarImplementation(llvm::ArrayRef<mlir::Value *> parallelIvs,
                           llvm::ArrayRef<mlir::Value *> reductionIvs) override;
#endif // LINALG_STEP
};

/// Implements C = A * B on 2-D matrices.
class MatmulOp : public TensorContractionBase,
                 public mlir::Op<MatmulOp, mlir::OpTrait::VariadicOperands,
                                 mlir::OpTrait::ZeroResult> {
public:
  using Op::Op;

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
  llvm::StringRef getTensorContractionName() override {
    return getOperationName();
  }
  unsigned getNumInputs() override { return 2; }
  unsigned getNumOutputs() override { return 1; }
  unsigned getNumParallelDims() override { return 2; }
  unsigned getNumReductionDims() override { return 1; }

#if LINALG_STEP > 2
  //////////////////////////////////////////////////////////////////////////////
  // Used in Linalg3 and later.
  //////////////////////////////////////////////////////////////////////////////
  /// Rewrites this op as a finer-grained tensor contraction (e.g. matmul is a
  /// loop over matvec). Does nothing by default.
  void writeAsFinerGrainTensorContraction() override;

  /// Inputs to this map will be (%m, %n, %k) coming from enclosing loops.
  /// Therefore, the mapping to get back to A(M, K), B(K, N), C(M, N) is:
  ///   (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)(%m, %n, %k)
  /// And the operands ranges are:
  ///   (%m, %k, %k, %n, %m, %n)
  llvm::SmallVector<mlir::AffineMap, 8> loopsToOperandRangeMaps() override;

  ///  Given a enclosing parallel loops with ivs `i` and `j`, and an enclosing
  ///  reduction loop with iv `r_k`, emits MLIR corresponding to:
  ///    1. conditionally assign scalarC to 0.0f on the first iteration or load
  ///       C[i, j]
  ///    2. multiply A[i, r_k] by B[r_k, j] and add to scalarC
  ///    3. store back scalarC at C[i, j]
  ///
  /// In some compact index notation this could be written:
  ///  cond = (r_k == zero)
  ///  scalarC = select(cond, zerof, C[i, j]);
  ///  C[i, j] = scalarC + A[i, r_k] * B[r_k, j];
  void
  emitScalarImplementation(llvm::ArrayRef<mlir::Value *> parallelIvs,
                           llvm::ArrayRef<mlir::Value *> reductionIvs) override;
#endif // LINALG_STEP
};

} // namespace linalg

#endif // LINALG2_TENSOROPS_H_
