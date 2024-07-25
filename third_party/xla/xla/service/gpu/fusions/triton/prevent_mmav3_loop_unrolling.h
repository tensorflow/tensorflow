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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_PREVENT_MMAV3_LOOP_UNROLLING_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_PREVENT_MMAV3_LOOP_UNROLLING_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace xla::gpu {

// This pass is a result of b/344841434:
// PTX sometimes unrolls wgmma loops that can cause a 1000x slow down in
// compilation time. Most unrolling has already been done before PTX,
// this pragma prevents ptxas from doing more.
std::unique_ptr<mlir::Pass> CreatePreventMmaV3LoopUnrollingPass();

void RegisterPreventMmaV3LoopUnrollingPass();

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_PREVENT_MMAV3_LOOP_UNROLLING_H_
