//===- ConvertGPUToSPIRVPass.h - GPU to SPIR-V conversion pass --*- C++ -*-===//
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
// Provides a pass to convert GPU ops to SPIRV ops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H
#define MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OpPassBase;

/// Pass to convert GPU Ops to SPIR-V ops.
std::unique_ptr<OpPassBase<ModuleOp>> createConvertGPUToSPIRVPass();

} // namespace mlir
#endif // MLIR_CONVERSION_GPUTOSPIRV_CONVERTGPUTOSPIRVPASS_H
