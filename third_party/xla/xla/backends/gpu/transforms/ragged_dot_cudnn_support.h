/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_

#include "absl/base/nullability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/decision.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// Returns an allowed Decision iff cuDNN positively reports an execution plan
// for the cuDNN fusion that RaggedDotFusionRewriter would build for
// `ragged_dot` on `device_description`. The ragged dot is cloned into a scratch
// module (with the debug options of its module) and RaggedDotFusionRewriter is
// run on it, then cuDNN is asked for the plan count
// (CuDnnFusionCompiler::GetAvailablePlanCount). The cuDNN handle is resolved
// with ResolveCudnnSupport, like CuDnnFusionCompiler does: it uses the live
// handle of `stream_exec`, or deviceless cuDNN when there is no `stream_exec`
// or when xla_gpu_cudnn_deviceless_compilation_mode forces it.
// Anything but a positive answer -- no plans, a probe failure, or no way to
// probe -- returns a forbidden Decision with the reason, so the caller falls
// back to the regular dense expansion.
//
// The probe normalizes all layouts to the default layout, matching what
// GpuLayoutAssignment later assigns to the ragged dot / cuDNN fusion.
Decision IsRaggedDotSupportedByCuDnn(
    const HloInstruction& ragged_dot,
    const stream_executor::DeviceDescription& device_description,
    stream_executor::StreamExecutor* absl_nullable stream_exec = nullptr);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_RAGGED_DOT_CUDNN_SUPPORT_H_
