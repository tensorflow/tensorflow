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

#ifndef XLA_CODEGEN_AUTOTUNER_H_
#define XLA_CODEGEN_AUTOTUNER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/thunk.h"

namespace xla {

// The XLA autotuner is responsible for choosing the best backend for compiling
// an XLA fusion to thunks. It relies on an executable provider to provide a set
// of candidates to choose from and on the runtime profiles to measure execution
// time and other metrics.
//
// Example: autotuning a dot fusion
//
//   %dot_fusion {
//     p0 = f32[1024, 1024] parameter(0)
//     p1 = f32[1024, 1024] parameter(1)
//     ROOT dot = f32[1024, 1024] dot(p0, p1)
//   }
//
// In XLA we have various backends that can generate thunks for the given
// fusion, and each backend also can choose from various strategies:
//
//   1. We can use vendor libraries (i.e. cuBLAS on NVIDIA GPUs), which
//      typically have various algorithms to choose from.
//
//   2. We can use a compiler that generates machine code for the given dot
//      (i.e. Triton on NVIDIA GPUs), and we can choose from various tile sizes
//      and other compiler options.
//
// Executable provider implementations are responsible for compiling (or
// simply lowering to library calls) a fusion to a sequence of thunks, and
// returning them to the autotuner.
//
// Autotuner chooses the best candidate, and uses an executable config to
// annotate the fusion with the chosen configuration (backend name and
// backend-specific knobs), so that later at compile time we can immediately
// select the best strategy and emit the most efficient thunks.
class Autotuner {
 public:
  virtual ~Autotuner() = default;

  using Knob = std::variant<int64_t, double, std::string>;

  // ExecutableConfig that can be used to annotate fusion and let
  // codegen pick the right backends + knobs (tiles, etc.)
  //
  // Example: ExecutableConfig {"triton", {"t0": 128, "t1": 256}}
  struct ExecutableConfig {
    std::string backend;
    absl::flat_hash_map<std::string, Knob> knobs;
  };

  // Result of compiling a fusion to thunk sequence for given executable config.
  //
  // Example: Executable {0, {KernelThunk<triton-kernel-inside>}}
  struct Executable {
    ThunkSequence sequence;
    size_t scratch_allocation;
  };

  // Result of running and profiling an executable candidate.
  struct ExecutableProfile {
    absl::Duration execution_time;
  };

  // Executable provider is an interface for providing a set of candidates for
  // an XLA autotuner to choose from.
  class ExecutableProvider {
   public:
    virtual ~ExecutableProvider() = default;

    // Returns the default executable config for the given fusion.
    virtual ExecutableConfig GetDefaultConfig(
        const HloFusionInstruction* fusion) = 0;

    // Returns a list of executable configs that can be used to compile
    // a given fusion to a thunk sequence.
    virtual absl::StatusOr<std::vector<ExecutableConfig>>
    SupportedExecutableConfigs(const HloFusionInstruction* fusion) = 0;

    // Compiles a given fusion with a given executable config.
    virtual absl::StatusOr<Executable> Compile(
        const HloFusionInstruction* fusion, const ExecutableConfig& config) = 0;
  };

  // Runs an autotuner result and measures execution time + whatever metric is
  // important.
  virtual absl::StatusOr<ExecutionProfile> Run(
      const HloFusionInstruction* fusion, Executable executable) = 0;
};

}  // namespace xla

#endif  // XLA_CODEGEN_AUTOTUNER_H_
