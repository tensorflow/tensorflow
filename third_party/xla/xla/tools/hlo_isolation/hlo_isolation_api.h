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

#ifndef XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_API_H_
#define XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_API_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/tools/hlo_isolation/hlo_isolation.pb.h"

namespace xla {
namespace hlo_isolation {

struct ModuleIsolationOptions {
  double abs_error_bound = 0.01;
  double rel_error_bound = 0.1;
  bool run_hlo_passes = false;
  int64_t max_module_size_bytes = 0;

  std::function<absl::StatusOr<Literal>(std::unique_ptr<HloModule> module,
                                        HloRunnerInterface* runner,
                                        absl::Span<const Literal> input_data)>
      run_module_fn;

  std::function<void(const HloModule& module, const Literal& test_output,
                     const Literal& reference_output,
                     const absl::Status& compare_status)>
      on_mismatch_fn;

  std::function<absl::StatusOr<std::vector<Literal>>(const HloModule& module)>
      make_fake_arguments_fn;

  std::function<int64_t(const HloModule&)> estimate_module_size_fn;
};

struct PipelineIsolationOptions {
  // Sharding
  int64_t shard_index = -1;
  int64_t num_shards = 1;

  // Filtering
  std::string filter_by_name = ".*";
  std::string skip_by_name;
  std::string filter_by_opcode = ".*";
  std::string skip_by_opcode;

  ModuleIsolationOptions module_options;
};

absl::StatusOr<Literal> RunModule(std::unique_ptr<HloModule> module,
                                  HloRunnerInterface* runner,
                                  absl::Span<const Literal> input_data,
                                  bool run_hlo_passes = false);

absl::StatusOr<HloIsolationTestResult> RunIsolationTestOnModule(
    const HloModule& module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, ModuleIsolationOptions options,
    absl::Span<const Literal> input_data = {});

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const std::string& input_path, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options);

absl::StatusOr<std::vector<HloIsolationTestResult>> RunIsolationPipeline(
    const HloModule& module, HloRunnerInterface* test_runner,
    HloRunnerInterface* reference_runner, PipelineIsolationOptions options);

absl::Status DefuseModule(HloModule* module);

absl::StatusOr<std::vector<NumericMismatch>> ExtractAndEnrichTopMismatches(
    std::string error_message, const HloModule* module);

absl::StatusOr<std::vector<NumericMismatch>> ExtractTopMismatches(
    std::string error_message, bool is_tuple = false);

absl::StatusOr<std::vector<bool>> DetectReducesInModuleOutput(
    const HloModule* module);

absl::StatusOr<NumericMismatch> ExtractTopRelativeErrorMismatch(
    std::string error_message);

int64_t GetFusionCountInNestedFusion(const HloInstruction* fusion_instr);

bool ModuleContainsLargeKeyValueSort(const HloModule& module);
bool ModuleTestsFloatsForEquality(const HloModule& module);
bool ComputationHasRng(const HloComputation* computation);
bool LiteralContainsInfOrNan(const LiteralSlice& literal);

}  // namespace hlo_isolation
}  // namespace xla

#endif  // XLA_TOOLS_HLO_ISOLATION_HLO_ISOLATION_API_H_
