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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_GPU_COPY_ASYNC_WRAPPER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_GPU_COPY_ASYNC_WRAPPER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Converts eligible device-to-device HLO `copy` instructions into
// `copy-start`/`copy-done` pairs so that the latency-hiding scheduler can
// overlap the D2D memcpy with compute-bound kernels on the main stream.
//
// The pass is disabled when `xla_gpu_async_copy_min_bytes` is -1. Otherwise, a
// copy is eligible when:
//   - Both source and destination reside in device memory (no host memory
//     space annotation), i.e. the copy is a true D2D transfer.
//   - Source and destination layouts are identical, so the copy is a plain
//     memcpy rather than a layout-changing transpose.
//   - The transfer size is at or above `xla_gpu_async_copy_min_bytes` to
//     amortize stream-synchronization overhead.
//   - The instruction is not already inside an async computation.
//
// The resulting `copy-start`/`copy-done` pair is handled by existing machinery:
//   - `ExecutionStreamAssignment` assigns a `ComputationStreamId`.
//   - `ThunkEmitter::EmitCopyStartThunk` wraps the copy in an `AsyncStartThunk`
//     that issues the memcpy on an auxiliary compute stream.
//   - `ThunkEmitter::EmitCopyDoneThunk` emits an `AsyncDoneThunk` that makes
//     the main stream wait for completion.
//   - The LHS scheduler maximises the overlap window between start and done.
class GpuCopyAsyncWrapper : public HloModulePass {
 public:
  absl::string_view name() const override { return "gpu-copy-async-wrapper"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_GPU_COPY_ASYNC_WRAPPER_H_
