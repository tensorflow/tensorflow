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

#ifndef XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_LIB_H_

#include <cstdint>

#include "absl/base/thread_annotations.h"
#include "absl/numeric/int128.h"
#include "absl/synchronization/mutex.h"

namespace xla::cpu {

class RngState {
 public:
  explicit RngState(int64_t delta);

  void GetAndUpdateState(uint64_t* data);

  int64_t delta() const { return delta_; }

 private:
  int64_t delta_;

  absl::Mutex mu_;
  absl::int128 state_ ABSL_GUARDED_BY(mu_);
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_RNG_STATE_LIB_H_
