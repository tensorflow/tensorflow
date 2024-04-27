/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/tools/multihost_hlo_runner/hlo_runner_flags.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "tsl/platform/logging.h"

namespace xla {

void MultiHostHloRunnerFlags::AppendFlags(std::vector<tsl::Flag>* flags) {
  CHECK_NE(flags, nullptr);
  CHECK(!added_flags_);

  flags->emplace_back("num_replicas", &flag_values_.num_replicas,
                      "The number of replicas; set to -1 for multihost "
                      "execution, which then uses all devices on all host.");
  flags->emplace_back("num_partitions", &flag_values_.num_partitions,
                      "Number of partitions for SPMD.");
  flags->emplace_back("log_output", &flag_values_.log_output,
                      "Log the input and output to stderr.");
  flags->emplace_back("run_xla_backend_only",
                      &flag_values_.run_xla_backend_only,
                      "Call only XLA's RunBackend during the compilation. "
                      "This is used to run a post-optimization HLO module "
                      "(dumped as 'xxx.after_optimizations.hlo.xxx'");
  flags->emplace_back("disable_all_hlo_passes",
                      &flag_values_.disable_all_hlo_passes,
                      "Disable HLO passes or not.");
  flags->emplace_back("use_spmd_partitioning",
                      &flag_values_.use_spmd_partitioning,
                      "Partition the module using SPMD.");
  flags->emplace_back(
      "is_spmd_partitioned_module", &flag_values_.is_spmd_partitioned_module,
      "The module is the partitioned result of SPMD. Setting this flag also "
      "disables all HLO passes and sets use_spmd_partitioning.");
  flags->emplace_back("xla_dump_to", &flag_values_.xla_dump_to,
                      "A directory to dump xla debug data to.");
  flags->emplace_back("xla_dump_as_text", &flag_values_.xla_dump_as_text,
                      "Whether to dump xla debug data as text.");
  flags->emplace_back("xla_dump_as_proto", &flag_values_.xla_dump_as_proto,
                      "Whether to dump xla debug data as protobuf.");
  flags->emplace_back(
      "hlo_argument_mode", &flag_values_.hlo_argument_mode,
      "Specify how arguments to the HLO module are generated. Accepted values: "
      "use_device_id_as_input, use_random_inputs, use_shared_random_inputs, "
      "use_zeros_as_input or uninitialized.");
  flags->emplace_back("while_execution_count",
                      &flag_values_.while_execution_count,
                      "If set to a positive number, flatten all while loops to "
                      "a certain number of iterations.");
  flags->emplace_back(
      "remove_infeed_outfeed", &flag_values_.remove_infeed_outfeed,
      "If set, we will remove all infeed and outfeed operations.");
  flags->emplace_back("num_repeats", &flag_values_.num_repeats,
                      "Repeatedly execute the HLO for this many times.");
  flags->emplace_back("execution_options_path",
                      &flag_values_.execution_options_path,
                      "A path to a protobuf text file which stores the "
                      "ExecutionOptions message for this HLO module.");

  added_flags_ = true;
}

void MultiHostHloRunnerFlags::PreprocessFlags() {
  CHECK(added_flags_);

  if (flag_values_.is_spmd_partitioned_module) {
    flag_values_.use_spmd_partitioning = true;
    flag_values_.disable_all_hlo_passes = true;
  } else if (flag_values_.use_spmd_partitioning) {
    flag_values_.disable_all_hlo_passes = false;
  }
}

bool MultiHostHloRunnerFlags::CreateOptionsFromFlags(
    FunctionalHloRunner::PreprocessingOptions* preproc_options,
    FunctionalHloRunner::RawCompileOptions* raw_compile_options,
    FunctionalHloRunner::RunningOptions* running_options, std::string* error) {
  CHECK(added_flags_);

  PreprocessFlags();

  FunctionalHloRunner::ModuleArgumentMode module_argument_mode;
  if (!xla::AbslParseFlag(flag_values_.hlo_argument_mode, &module_argument_mode,
                          error)) {
    return false;
  }

  *preproc_options = FunctionalHloRunner::PreprocessingOptions();
  preproc_options->spmd_partitioned_mode =
      flag_values_.is_spmd_partitioned_module
          ? FunctionalHloRunner::SpmdPartitionedMode::kIsSpmdPartitionedModule
          : FunctionalHloRunner::SpmdPartitionedMode::
                kIsNotSpmdPartitionedModule;
  preproc_options->while_execution_count =
      flag_values_.while_execution_count > 0
          ? std::make_optional(flag_values_.while_execution_count)
          : std::nullopt;
  preproc_options->remove_infeed_outfeed = flag_values_.remove_infeed_outfeed;

  *raw_compile_options = FunctionalHloRunner::RawCompileOptions();
  raw_compile_options->hlo_passes_mode =
      flag_values_.run_xla_backend_only
          ? FunctionalHloRunner::HloPassesMode::kRunXLABackendOnly
          : (flag_values_.disable_all_hlo_passes
                 ? FunctionalHloRunner::HloPassesMode::kDisableAllHloPasses
                 : FunctionalHloRunner::HloPassesMode::kStandardCompile);
  raw_compile_options->spmd_mode =
      flag_values_.use_spmd_partitioning
          ? FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning
          : FunctionalHloRunner::SpmdMode::kNotUseSpmdPartitioning;
  if (!flag_values_.execution_options_path.empty()) {
    absl::StatusOr<ExecutionOptions> execution_options =
        FunctionalHloRunner::LoadExecutionOptions(
            flag_values_.execution_options_path);
    if (!execution_options.ok()) {
      *error = absl::StrCat("Could not read execution options from ",
                            flag_values_.execution_options_path,
                            ". Error: ", execution_options.status().ToString());
      return false;
    }

    raw_compile_options->execution_options = execution_options.value();
  }
  raw_compile_options->num_replicas =
      flag_values_.num_replicas < 0
          ? std::nullopt
          : std::optional<int>(flag_values_.num_replicas);
  raw_compile_options->num_partitions =
      flag_values_.num_partitions < 0
          ? std::nullopt
          : std::optional<int>(flag_values_.num_partitions);
  raw_compile_options->xla_dump_to = flag_values_.xla_dump_to;
  raw_compile_options->xla_text_dump_mode =
      flag_values_.xla_dump_as_text
          ? FunctionalHloRunner::XlaTextDumpMode::kDumpAsText
          : FunctionalHloRunner::XlaTextDumpMode::kNotDumpAsText;
  raw_compile_options->xla_proto_dump_mode =
      flag_values_.xla_dump_as_proto
          ? FunctionalHloRunner::XlaProtoDumpMode::kDumpAsProto
          : FunctionalHloRunner::XlaProtoDumpMode::kNotDumpAsProto;

  *running_options = FunctionalHloRunner::RunningOptions();
  running_options->module_argument_mode = module_argument_mode;
  running_options->module_output_mode =
      FunctionalHloRunner::ModuleOutputMode::kReturnOutputs;
  running_options->num_repeats = static_cast<size_t>(flag_values_.num_repeats);
  running_options->log_input_output_mode =
      flag_values_.log_output
          ? FunctionalHloRunner::LogOutputMode::kLogOutput
          : FunctionalHloRunner::LogOutputMode::kNotLogOutput;

  return true;
}

}  // namespace xla
