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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

// RngSeedThunk handles the custom "GetRngSeed" call, which is
// used to pass an RNG seed supplied at runtime. The custom call is emitted
// during the rng_expander pass.
//
// This thunk initializes a buffer with a U64 seed and writes the required seed
// into it during execution. The seed is passed in via the ExecuteParams and is
// either directly written into the buffer (if seed != 0), or a new random seed
// is generated and written into the buffer (if seed == 0).
class RngSeedThunk : public xla::gpu::Thunk {
 public:
  RngSeedThunk(xla::gpu::Thunk::ThunkInfo thunk_info,
               const BufferAllocation::Slice& dest)
      : xla::gpu::Thunk(xla::gpu::Thunk::Kind::kRngSeed, std::move(thunk_info)),
        dest_(dest) {}

  absl::Status ExecuteOnStream(
      const xla::gpu::Thunk::ExecuteParams& params) override;

  BufferAllocation::Slice dest() const { return dest_; }

  xla::gpu::Thunk::BufferUses buffer_uses() const override {
    return {BufferUse::Write(dest_, ShapeUtil::MakeShape(U64, {}))};
  }

  absl::StatusOr<ThunkProto> ToProto() const override;

  static absl::StatusOr<std::unique_ptr<RngSeedThunk>> FromProto(
      ThunkInfo thunk_info, const RngSeedThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

 private:
  // Generate random seed if requested, otherwise use current params seed.
  uint64_t ResolveSeed(const xla::gpu::Thunk::ExecuteParams& params) const;

  const BufferAllocation::Slice dest_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RNG_SEED_THUNK_H_
