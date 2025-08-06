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

#ifndef XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_
#define XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_

#include <optional>

#include "absl/status/status.h"
#include "xla/backends/autotuner/autotuner_cache.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

// Class to manage the autotuning cache.
// Note: We do not enforce cross implementation compatibility. E.g. different
// AutotunerCacheInterface implementations may have different cache keys.
class AutotunerCacheInterface {
 public:
  virtual ~AutotunerCacheInterface() = default;

  virtual std::optional<AutotunerCacheEntry> Lookup(
      const HloInstruction* instr) = 0;

  virtual absl::Status Insert(const HloInstruction* instr,
                              AutotunerCacheEntry& entry) = 0;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_AUTOTUNER_CACHE_INTERFACE_H_
