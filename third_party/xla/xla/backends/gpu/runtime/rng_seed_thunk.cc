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

#include "xla/backends/gpu/runtime/rng_seed_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "tsl/platform/random.h"

namespace xla::gpu {

uint64_t RngSeedThunk::ResolveSeed(const Thunk::ExecuteParams& params) const {
  uint64_t seed = params.rng_seed;
  if (seed == 0) {
    // 0 is avoided because it is used as a sentinel value indicating that a
    // random seed should be generated. Generate a random non-zero seed as
    // fallback.
    do {
      seed = tsl::random::New64();
    } while (seed == 0);
  }
  return seed;
}

absl::Status RngSeedThunk::ExecuteOnStream(const Thunk::ExecuteParams& params) {
  if (params.buffer_allocations == nullptr) {
    return absl::InternalError("buffer_allocations cannot be null");
  }
  auto dest_addr = params.buffer_allocations->GetDeviceAddress(dest_);
  if (dest_addr.opaque() == nullptr) {
    return absl::InternalError("seed buffer opaque address cannot be null");
  }
  uint64_t seed = ResolveSeed(params);
  VLOG(3) << "RngSeedThunk executing with seed " << seed;
  uint32_t seed_low = static_cast<uint32_t>(seed & 0xFFFFFFFF);
  uint32_t seed_high = static_cast<uint32_t>(seed >> 32);
  auto dest_high = dest_addr.GetByteSlice(4, 4);
  RETURN_IF_ERROR(params.stream->Memset32(&dest_addr, seed_low, 4));
  return params.stream->Memset32(&dest_high, seed_high, 4);
}

absl::StatusOr<ThunkProto> RngSeedThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  auto* rng_seed_thunk_proto = proto.mutable_rng_seed_thunk();
  ASSIGN_OR_RETURN(*rng_seed_thunk_proto->mutable_dest_buffer(),
                   dest().ToProto());
  return proto;
}

absl::StatusOr<std::unique_ptr<RngSeedThunk>> RngSeedThunk::FromProto(
    ThunkInfo thunk_info, const RngSeedThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ASSIGN_OR_RETURN(BufferAllocation::Slice dest,
                   BufferAllocation::Slice::FromProto(thunk_proto.dest_buffer(),
                                                      buffer_allocations));
  return std::make_unique<RngSeedThunk>(std::move(thunk_info), dest);
}

}  // namespace xla::gpu
