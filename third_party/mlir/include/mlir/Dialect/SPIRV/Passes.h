//===- Passes.h - SPIR-V pass entry points ----------------------*- C++ -*-===//
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
