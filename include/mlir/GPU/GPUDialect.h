//===- GPUDialect.h - MLIR Dialect for GPU Kernels --------------*- C++ -*-===//
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
// This file defines the GPU kernel-related operations and puts them in the
// corresponding dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_GPU_GPUDIALECT_H
#define MLIR_GPU_GPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace gpu {

/// The dialect containing GPU kernel launching operations and related
/// facilities.
class GPUDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  GPUDialect(MLIRContext *context);

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName();

  /// Get the name of the attribute used to annotate outlined kernel functions.
  static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }
};

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  Value *x;
  Value *y;
  Value *z;
};

/// GPU kernel launch operation.  Takes a 3D grid of thread blocks as leading
/// operands, followed by kernel data operands.  Has one region representing
/// the kernel to be executed.  This region is not allowed to use values defined
/// outside it.
class LaunchOp : public Op<LaunchOp, OpTrait::AtLeastNOperands<6>::Impl,
                           OpTrait::ZeroResult, OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;

  static void build(Builder *builder, OperationState *result, Value *gridSizeX,
                    Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                    Value *blockSizeY, Value *blockSizeZ,
                    ArrayRef<Value *> operands);

  /// Get the kernel region.
  Region &getBody();

  /// Get the SSA values corresponding to kernel block identifiers.
  KernelDim3 getBlockIds();
  /// Get the SSA values corresponding to kernel thread identifiers.
  KernelDim3 getThreadIds();
  /// Get the SSA values corresponding to kernel grid size.
  KernelDim3 getGridSize();
  /// Get the SSA values corresponding to kernel block size.
  KernelDim3 getBlockSize();
  /// Get the operand values passed as kernel arguments.
  operand_range getKernelOperandValues();
  /// Get the operand types passed as kernel arguments.
  operand_type_range getKernelOperandTypes();

  /// Get the SSA values passed as operands to specify the grid size.
  KernelDim3 getGridSizeOperandValues();
  /// Get the SSA values passed as operands to specify the block size.
  KernelDim3 getBlockSizeOperandValues();

  /// Get the SSA values of the kernel arguments.
  llvm::iterator_range<Block::args_iterator> getKernelArguments();

  LogicalResult verify();

  /// Custom syntax support.
  void print(OpAsmPrinter *p);
  static ParseResult parse(OpAsmParser *parser, OperationState *result);

  static StringRef getOperationName() { return "gpu.launch"; }

private:
  static StringRef getBlocksKeyword() { return "blocks"; }
  static StringRef getThreadsKeyword() { return "threads"; }
  static StringRef getArgsKeyword() { return "args"; }

  /// The number of launch configuration operands, placed at the leading
  /// positions of the operand list.
  static constexpr unsigned kNumConfigOperands = 6;

  /// The number of region attributes containing the launch configuration,
  /// placed in the leading positions of the argument list.
  static constexpr unsigned kNumConfigRegionAttributes = 12;
};

/// Operation to launch a kernel given as outlined function.
class LaunchFuncOp : public Op<LaunchFuncOp, OpTrait::AtLeastNOperands<6>::Impl,
                               OpTrait::ZeroResult> {
public:
  using Op::Op;

  static void build(Builder *builder, OperationState *result,
                    Function *kernelFunc, Value *gridSizeX, Value *gridSizeY,
                    Value *gridSizeZ, Value *blockSizeX, Value *blockSizeY,
                    Value *blockSizeZ, ArrayRef<Value *> kernelOperands);

  static void build(Builder *builder, OperationState *result,
                    Function *kernelFunc, KernelDim3 gridSize,
                    KernelDim3 blockSize, ArrayRef<Value *> kernelOperands);

  /// The kernel function specified by the operation's `kernel` attribute.
  StringRef kernel();
  /// The number of operands passed to the kernel function.
  unsigned getNumKernelOperands();
  /// The i-th operand passed to the kernel function.
  Value *getKernelOperand(unsigned i);

  /// Get the SSA values passed as operands to specify the grid size.
  KernelDim3 getGridSizeOperandValues();
  /// Get the SSA values passed as operands to specify the block size.
  KernelDim3 getBlockSizeOperandValues();

  LogicalResult verify();

  static StringRef getOperationName() { return "gpu.launch_func"; }

  /// The number of launch configuration operands, placed at the leading
  /// positions of the operand list.
  static constexpr unsigned kNumConfigOperands = 6;

private:
  /// The name of the function attribute specifying the kernel to launch.
  static StringRef getKernelAttrName() { return "kernel"; }
};

#define GET_OP_CLASSES
#include "mlir/GPU/GPUOps.h.inc"

} // end namespace gpu
} // end namespace mlir

#endif // MLIR_GPUKERNEL_GPUDIALECT_H
