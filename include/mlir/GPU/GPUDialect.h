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

/// The dialect containing GPU kernel launching operations and related
/// facilities.
class GPUDialect : public Dialect {
public:
  /// Create the dialect in the given `context`.
  GPUDialect(MLIRContext *context);

  /// Get the canonical string name of the dialect.
  static StringRef getDialectName();
};

struct KernelDim3 {
  Value *x;
  Value *y;
  Value *z;
};

class LaunchOp : public Op<LaunchOp, OpTrait::AtLeastNOperands<6>::Impl,
                           OpTrait::ZeroResult,
                           OpTrait::NthRegionIsIsolatedAbove<0>::Impl> {
public:
  using Op::Op;

  static void build(Builder *builder, OperationState *result, Value *gridSizeX,
                    Value *gridSizeY, Value *gridSizeZ, Value *blockSizeX,
                    Value *blockSizeY, Value *blockSizeZ,
                    ArrayRef<Value *> operands);

  Region &getBody();
  KernelDim3 getBlockIds();
  KernelDim3 getThreadIds();
  KernelDim3 getGridSize();
  KernelDim3 getBlockSize();

  LogicalResult verify();

  static StringRef getOperationName() { return "gpu.launch"; }
};

} // end namespace mlir

#endif // MLIR_GPUKERNEL_GPUDIALECT_H
