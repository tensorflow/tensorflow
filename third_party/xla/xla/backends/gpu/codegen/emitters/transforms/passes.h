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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateConvertFloatNvidiaPass();
std::optional<std::unique_ptr<mlir::Pass>> MaybeCreateConvertFloatNvidiaPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateConvertIndexTypePass();
std::unique_ptr<mlir::Pass> CreateOptimizeLoopsPass();
std::unique_ptr<mlir::Pass> CreateFuseLoopsPass();
std::unique_ptr<mlir::Pass> CreatePeelLoopsPass();

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
