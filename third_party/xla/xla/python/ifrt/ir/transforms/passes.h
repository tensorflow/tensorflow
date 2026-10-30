/* Copyright 2023 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_

#include <memory>  // IWYU pragma: keep
#include <string>  // IWYU pragma: keep

#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Pass/PassManager.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

// Creates pipeline of all the IFRT IR passes that do not require
// compilation-time information (e.g., device assignments).
void createIfrtToOutlinedAtomProgramsPipeline(mlir::OpPassManager& pm);

// Creates a pipeline that populates metadata info for each atom program.
void createIfrtPopulateAtomProgramMetadataPipeline(mlir::OpPassManager& pm);

// Registers passes and pipelines to ifrt-opt.
void registerIfrtPassesAndPipelines();

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
