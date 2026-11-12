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

#ifndef XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_THUNK_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/rng_state_resource.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/resource_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Reads and advances the executable wide RNG state resource by `delta`, writing
// the previous 128 bit value into `state_buffer`.
class RngGetAndUpdateStateThunk final : public Thunk {
 public:
  static absl::StatusOr<std::unique_ptr<RngGetAndUpdateStateThunk>> Create(
      Info info, BufferAllocation::Slice state_buffer, int64_t delta,
      std::shared_ptr<RngStateResource> rng_state);

  tsl::AsyncValueRef<ExecuteEvent> Execute(const ExecuteParams& params) final;

  BufferUses buffer_uses() const final {
    return {BufferUse::Write(state_buffer_, ShapeUtil::MakeShape(U64, {2}))};
  }

  ResourceUses resource_uses() const final {
    return {ResourceUse::Write(rng_state_)};
  }

  int64_t delta() const { return delta_; }

  const BufferAllocation::Slice& state_buffer() const { return state_buffer_; }

  const std::shared_ptr<RngStateResource>& rng_state() const {
    return rng_state_;
  }

 private:
  RngGetAndUpdateStateThunk(Info info, BufferAllocation::Slice state_buffer,
                            int64_t delta,
                            std::shared_ptr<RngStateResource> rng_state);

  BufferAllocation::Slice state_buffer_;
  int64_t delta_;
  std::shared_ptr<RngStateResource> rng_state_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_THUNK_H_
