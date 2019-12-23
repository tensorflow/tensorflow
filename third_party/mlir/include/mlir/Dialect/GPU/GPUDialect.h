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
#include "mlir/IR/OpImplementation.h"
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

  /// Returns the numeric value used to identify the workgroup memory address
  /// space.
  static unsigned getWorkgroupAddressSpace() { return 3; }

  /// Returns the numeric value used to identify the private memory address
  /// space.
  static unsigned getPrivateAddressSpace() { return 5; }

  LogicalResult verifyOperationAttribute(Operation *op,
                                         NamedAttribute attr) override;
};

/// Utility class for the GPU dialect to represent triples of `Value`s
/// accessible through `.x`, `.y`, and `.z` similarly to CUDA notation.
struct KernelDim3 {
  ValuePtr x;
  ValuePtr y;
  ValuePtr z;
};

#define GET_OP_CLASSES
#include "mlir/Dialect/GPU/GPUOps.h.inc"

} // end namespace gpu
} // end namespace mlir

#endif // MLIR_DIALECT_GPU_GPUDIALECT_H
