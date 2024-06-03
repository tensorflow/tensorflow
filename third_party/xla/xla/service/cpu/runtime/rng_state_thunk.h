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

#ifndef XLA_SERVICE_CPU_RUNTIME_RNG_STATE_THUNK_H_
#define XLA_SERVICE_CPU_RUNTIME_RNG_STATE_THUNK_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/numeric/int128.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"

namespace xla::cpu {

// Keeps rng state as a "global" variable (global for the parent cpu
// executable) and adds a delta value to it and returns the old value on every
// call to execute.
class RngGetAndUpdateStateThunk final : public Thunk {
 public:
  RngGetAndUpdateStateThunk(Info info, BufferAllocation::Slice state_buffer,
                            int64_t delta);

  absl::Status Execute(const ExecuteParams& params) final;

 private:
  BufferAllocation::Slice state_buffer_;
  int64_t delta_;

  absl::Mutex mu_;
  absl::int128 state_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_RNG_STATE_THUNK_H_
