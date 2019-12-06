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

#ifndef MLIR_DIALECT_GPU_GPUDIALECT_H
#define MLIR_DIALECT_GPU_GPUDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
class FuncOp;

namespace gpu {

/// The dialect containing GPU kernel launching operations and related
/// facilities.
class GPUDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  explicit GPUDialect(MLIRContext *context);
  /// Get dialect namespace.
  static StringRef getDialectNamespace() { return "gpu"; }

  /// Get the name of the attribute used to annotate the modules that contain
  /// kernel modules.
  static StringRef getContainerModuleAttrName() {
    return "gpu.container_module";
  }

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName();

  /// Get the name of the attribute used to annotate external kernel functions.
  static StringRef getKernelFuncAttrName() { return "gpu.kernel"; }

  /// Get the name of the attribute used to annotate kernel modules.
  static StringRef getKernelModuleAttrName() { return "gpu.kernel_module"; }

  /// Returns whether the given function is a kernel function, i.e., has the
  /// 'gpu.kernel' attribute.
  static bool isKernel(Operation *op);

  LogicalResult verifyOperationAttribute(Operation *op,
                                         NamedAttribute attr) override;
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

  static void build(Builder *builder, OperationState &result, Value *gridSizeX,
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
  void print(OpAsmPrinter &p);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  static StringRef getOperationName() { return "gpu.launch"; }

  /// Erase the `index`-th kernel argument.  Both the entry block argument and
  /// the operand will be dropped.  The block argument must not have any uses.
  void eraseKernelArgument(unsigned index);

  /// Append canonicalization patterns to `results`.
  static void getCanonicalizationPatterns(OwningRewritePatternList &results,
                                          MLIRContext *context);

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

  static void build(Builder *builder, OperationState &result, FuncOp kernelFunc,
                    Value *gridSizeX, Value *gridSizeY, Value *gridSizeZ,
                    Value *blockSizeX, Value *blockSizeY, Value *blockSizeZ,
                    ArrayRef<Value *> kernelOperands);

  static void build(Builder *builder, OperationState &result, FuncOp kernelFunc,
                    KernelDim3 gridSize, KernelDim3 blockSize,
                    ArrayRef<Value *> kernelOperands);

  /// The kernel function specified by the operation's `kernel` attribute.
  StringRef kernel();
  /// The number of operands passed to the kernel function.
  unsigned getNumKernelOperands();
  /// The name of the kernel module specified by the operation's `kernel_module`
  /// attribute.
  StringRef getKernelModuleName();
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
  // This needs to quietly verify if attributes with names defined below are
  // present since it is run before the verifier of this op.
  friend LogicalResult GPUDialect::verifyOperationAttribute(Operation *,
                                                            NamedAttribute);

  /// The name of the symbolRef attribute specifying the kernel to launch.
  static StringRef getKernelAttrName() { return "kernel"; }

  /// The name of the symbolRef attribute specifying the name of the module
  /// containing the kernel to launch.
  static StringRef getKernelModuleAttrName() { return "kernel_module"; }
};

class GPUFuncOp : public Op<GPUFuncOp, OpTrait::FunctionLike,
                            OpTrait::IsIsolatedFromAbove, OpTrait::Symbol> {
public:
  using Op::Op;

  /// Returns the name of the operation.
  static StringRef getOperationName() { return "gpu.func"; }

  /// Constructs a FuncOp, hook for Builder methods.
  static void build(Builder *builder, OperationState &result, StringRef name,
                    FunctionType type, ArrayRef<Type> workgroupAttributions,
                    ArrayRef<Type> privateAttributions,
                    ArrayRef<NamedAttribute> attrs);

  /// Prints the Op in custom format.
  void print(OpAsmPrinter &p);

  /// Parses the Op in custom format.
  static ParseResult parse(OpAsmParser &parser, OperationState &result);

  /// Returns `true` if the GPU function defined by this Op is a kernel, i.e.
  /// it is intended to be launched from host.
  bool isKernel() {
    return getAttrOfType<UnitAttr>(GPUDialect::getKernelFuncAttrName()) !=
           nullptr;
  }

  /// Returns the type of the function this Op defines.
  FunctionType getType() {
    return getTypeAttr().getValue().cast<FunctionType>();
  }

  /// Returns the number of buffers located in the workgroup memory.
  unsigned getNumWorkgroupAttributions() {
    return getAttrOfType<IntegerAttr>(getNumWorkgroupAttributionsAttrName())
        .getInt();
  }

  /// Returns a list of block arguments that correspond to buffers located in
  /// the workgroup memory
  ArrayRef<BlockArgument *> getWorkgroupAttributions() {
    auto begin =
        std::next(getBody().front().args_begin(), getType().getNumInputs());
    auto end = std::next(begin, getNumWorkgroupAttributions());
    return {begin, end};
  }

  /// Returns a list of block arguments that correspond to buffers located in
  /// the private memory.
  ArrayRef<BlockArgument *> getPrivateAttributions() {
    auto begin =
        std::next(getBody().front().args_begin(),
                  getType().getNumInputs() + getNumWorkgroupAttributions());
    return {begin, getBody().front().args_end()};
  }

private:
  // FunctionLike trait needs access to the functions below.
  friend class OpTrait::FunctionLike<GPUFuncOp>;

  /// Hooks for the input/output type enumeration in FunctionLike .
  unsigned getNumFuncArguments() { return getType().getNumInputs(); }
  unsigned getNumFuncResults() { return getType().getNumResults(); }

  /// Returns the name of the attribute containing the number of buffers located
  /// in the workgroup memory.
  static StringRef getNumWorkgroupAttributionsAttrName() {
    return "workgroup_attibutions";
  }

  /// Returns the keywords used in the custom syntax for this Op.
  static StringRef getWorkgroupKeyword() { return "workgroup"; }
  static StringRef getPrivateKeyword() { return "private"; }
  static StringRef getKernelKeyword() { return "kernel"; }

  /// Hook for FunctionLike verifier.
  LogicalResult verifyType();

  /// Verifies the body of the function.
  LogicalResult verifyBody();
};

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.h.inc"

} // end namespace gpu
} // end namespace mlir

#endif // MLIR_DIALECT_GPU_GPUDIALECT_H
