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

#ifndef XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_RESOURCE_H_
#define XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_RESOURCE_H_

#include <cstdint>

#include "xla/backends/cpu/runtime/rng_state_lib.h"
#include "xla/runtime/resource_use.h"

namespace xla::cpu {

// Resource wrapper around an executable scoped `RngState`, so one shared object
// is both the `ThunkExecutor` ordering token and the counter storage. This is
// the only valid way to create a `Resource::kRngState` resource.
class RngStateResource final : public Resource {
 public:
  RngStateResource() : Resource(Kind::kRngState) {}

  void GetAndUpdateState(uint64_t* data, int64_t delta,
                         int64_t device_ordinal = 0) {
    state_.GetAndUpdateState(data, delta, device_ordinal);
  }

 private:
  RngState state_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_RESOURCE_H_
