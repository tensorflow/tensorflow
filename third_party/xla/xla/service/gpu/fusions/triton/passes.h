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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla::gpu {

#define GEN_PASS_DECL
#include "xla/service/gpu/fusions/triton/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateGeneralizeKernelSignaturePass();

#define GEN_PASS_REGISTRATION
#include "xla/service/gpu/fusions/triton/passes.h.inc"

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_PASSES_H_
