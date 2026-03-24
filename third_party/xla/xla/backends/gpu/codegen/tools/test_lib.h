/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_
#define XLA_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"

namespace xla {

namespace gpu {

// Returns the MLIR fusion emitter for the given module, which should have been
// loaded using LoadTestModule.
struct EmitterData {
  HloFusionInstruction* fusion;
  std::optional<se::DeviceDescription> device;
  std::optional<HloFusionAnalysis> analysis;
  std::unique_ptr<EmitterBase> emitter;
};
absl::StatusOr<std::unique_ptr<EmitterData>> GetEmitter(
    const HloModule& module, mlir::MLIRContext& mlir_context);

// Returns an MLIR context with all the dialects needed for testing.
mlir::MLIRContext GetMlirContextForTest();

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TOOLS_TEST_LIB_H_
