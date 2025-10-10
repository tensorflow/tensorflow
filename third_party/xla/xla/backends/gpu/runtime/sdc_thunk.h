/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SDC_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_SDC_THUNK_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/sdc_buffer_id.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/cuda/sdc_xor_checksum_kernel_cuda.h"

namespace xla::gpu {

class SdcThunk : public Thunk {
 public:
  explicit SdcThunk(
      ThunkInfo info, BufferAllocation::Slice log_slice,
      absl::flat_hash_map<SdcBufferId, BufferAllocation::Slice> buffers)
      : Thunk(Thunk::Kind::kSdc, std::move(info)),
        log_slice_(log_slice),
        buffers_(std::move(buffers)) {}

  absl::Status Initialize(const InitializeParams& params) override;
  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

  std::string ToString(int indent) const override;

  BufferUses buffer_uses() const override {
    // Intentionally left empty to not checksum the checksumming thunk.
    return {};
  }

 private:
  std::unique_ptr<stream_executor::cuda::SdcXorChecksumKernel::KernelType>
      xor_checksum_kernel_ = nullptr;
  BufferAllocation::Slice log_slice_;
  absl::flat_hash_map<SdcBufferId, BufferAllocation::Slice> buffers_;
};

class SdcDumpLogThunk : public Thunk {
 public:
  explicit SdcDumpLogThunk(ThunkInfo info, BufferAllocation::Slice log_slice,
                           const HloModule& hlo_module,
                           const DebugOptions& debug_options)
      : Thunk(Thunk::Kind::kSdcDumpLog, std::move(info)),
        log_slice_(log_slice),
        hlo_module_(hlo_module),
        debug_options_(debug_options) {}

  absl::Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  BufferAllocation::Slice log_slice_;
  const HloModule& hlo_module_;
  const DebugOptions debug_options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SDC_THUNK_H_
