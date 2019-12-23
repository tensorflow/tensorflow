//==-- GPUToNVVM.td - GPU Ops to NVVM Patterns ---------------*- tablegen -*==//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Patterns to lower GPU ops to NVVM.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTONVVM_TD
#define MLIR_CONVERSION_GPUTONVVM_TD

include "mlir/Dialect/GPU/GPUOps.td"
include "mlir/Dialect/LLVMIR/NVVMOps.td"

def : Pat<(GPU_BarrierOp), (NVVM_Barrier0Op)>;

#endif // MLIR_CONVERSION_GPUTONVVM_TD
