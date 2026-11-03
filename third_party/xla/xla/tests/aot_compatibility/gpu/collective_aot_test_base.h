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

#ifndef XLA_TESTS_AOT_COMPATIBILITY_GPU_COLLECTIVE_AOT_TEST_BASE_H_
#define XLA_TESTS_AOT_COMPATIBILITY_GPU_COLLECTIVE_AOT_TEST_BASE_H_

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/literal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tests/aot_compatibility/test_lib.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"

namespace xla {
namespace aot_compatibility {

// Fixture for multi-GPU AOT compatibility tests of collective thunks.
//
// Each test case compiles one HLO module, so every distinct runtime
// configuration (debug options, replica/partition layout, device assignment)
// is its own case and golden. Cases that share a configuration share a module.
class CollectiveAotTestBase
    : public AotCompatibilityTest,
      public ::testing::WithParamInterface<AotTestParam> {
 public:
  CollectiveAotTestBase() : AotCompatibilityTest(GetParam()) {}

 protected:
  // Parses `hlo` under `config` and runs it with one thread per device on the
  // devices of the config's static device assignment, which
  // GetModuleConfigForTest always sets.
  absl::StatusOr<std::vector<Literal>> RunReplicated(
      absl::string_view hlo, const HloModuleConfig& config,
      bool run_hlo_passes = true);

  // Like RunReplicated, but compiles (or loads the golden) once and executes
  // the same executable `num_runs` times. Returns the results of every run.
  // Needed for thunks whose state carries over between executions, e.g.
  // barriers that only synchronize from the second run on.
  absl::StatusOr<std::vector<std::vector<Literal>>> RunReplicatedRepeatedly(
      absl::string_view hlo, const HloModuleConfig& config, int num_runs);
};

// Keeps collectives asynchronous, so the executable holds AsyncStart/AsyncDone
// thunks instead of synchronous collective thunks.
void KeepCollectivesAsync(DebugOptions& debug_options);

// Sets use_symmetric_buffer on every collective (NCCL symmetric memory).
void EnableNcclSymmetricBuffers(DebugOptions& debug_options);

// Lowers all-reduce through NCCL (AllReduceThunk) instead of the
// CollectiveKernelThunk that xla_gpu_experimental_use_collective_kernels
// selects by default.
void DisableCollectiveKernels(DebugOptions& debug_options);

// Test parameters for the two suites every GPU AOT test target instantiates.
// `target_base` is the target name without the arch suffix, e.g.
// "p2p_collectives_aot_test_2gpu".
std::vector<AotTestParam> GpuBackwardsCompatibilityParams(
    absl::string_view target_base);
std::vector<AotTestParam> GpuGoldenFileVerificationParams(
    absl::string_view target_base);

// Names a parameterized case after its golden version, e.g. `v3`.
// TODO(aliia): Remove once cl/1000070191 moves it into test_lib.
std::string AotTestParamName(
    const ::testing::TestParamInfo<AotTestParam>& info);

}  // namespace aot_compatibility
}  // namespace xla

#endif  // XLA_TESTS_AOT_COMPATIBILITY_GPU_COLLECTIVE_AOT_TEST_BASE_H_
