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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "xla/autotuning.pb.h"
#include "xla/codegen/tiling/symbolic_tile_analysis.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/model/block_level_parameters.h"

namespace xla::gpu {

// Emits an xtile module.
// fn_name: The name of the function to emit.
// emitter_specific_constraints_builder: A builder for the emitter specific
//   constraints.
// fusion: The fusion instruction to emit.
// block_level_parameters: The block level parameters.
// mlir_context: The MLIR context to use.
// opaque_args_types: The types of the opaque arguments to the function, e.x.
// arguments for collectives in XLA:GPU.
absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> EmitXTileModule(
    absl::string_view fn_name,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
    const HloFusionInstruction* fusion,
    const BlockLevelParameters& block_level_parameters,
    mlir::MLIRContext& mlir_context,
    absl::Span<mlir::Type> opaque_args_types = {});

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_FUSION_EMITTER_H_
