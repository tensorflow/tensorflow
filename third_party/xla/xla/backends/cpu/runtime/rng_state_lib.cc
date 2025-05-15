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

#include "xla/backends/cpu/runtime/rng_state_lib.h"

#include <cstdint>
#include <cstring>

#include "absl/base/config.h"
#include "absl/numeric/int128.h"
#include "absl/synchronization/mutex.h"

namespace xla::cpu {

// Use a non-zero initial value as zero state can cause the result of the first
// random number generation not passing the chi-square test. The value used here
// is arbitrarily chosen, any non-zero values should be fine.
static constexpr absl::int128 kRngStateInitialValue = 0x7012395ull;

RngState::RngState(int64_t delta)
    : delta_(delta), state_(kRngStateInitialValue) {}

void RngState::GetAndUpdateState(uint64_t* data) {
  absl::MutexLock lock(&mu_);

  uint64_t low = absl::Int128Low64(state_);
  uint64_t high = absl::Int128High64(state_);

  static_assert(ABSL_IS_LITTLE_ENDIAN, "Big endian not supported");
  std::memcpy(data, &low, sizeof(low));
  std::memcpy(data + 1, &high, sizeof(high));

  state_ += delta_;
}

}  // namespace xla::cpu
