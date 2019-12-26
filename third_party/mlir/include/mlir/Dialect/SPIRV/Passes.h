//===- Passes.h - SPIR-V pass entry points ----------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_PASSES_H_
#define MLIR_DIALECT_SPIRV_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace spirv {

class ModuleOp;
/// Creates a module pass that converts composite types used by objects in the
/// StorageBuffer, PhysicalStorageBuffer, Uniform, and PushConstant storage
/// classes with layout information.
/// Right now this pass only supports Vulkan layout rules.
std::unique_ptr<OpPassBase<mlir::ModuleOp>>
createDecorateSPIRVCompositeTypeLayoutPass();

/// Creates a module pass that lowers the ABI attributes specified during SPIR-V
/// Lowering. Specifically,
/// 1) Creates the global variables for arguments of entry point function using
/// the specification in the ABI attributes for each argument.
/// 2) Inserts the EntryPointOp and the ExecutionModeOp for entry point
/// functions using the specification in the EntryPointAttr.
std::unique_ptr<OpPassBase<spirv::ModuleOp>> createLowerABIAttributesPass();

} // namespace spirv
} // namespace mlir

#endif // MLIR_DIALECT_SPIRV_PASSES_H_
