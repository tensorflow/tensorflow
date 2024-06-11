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

// Utility for launching some HLO text that supports multiple hosts/devices.

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/status_macros.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace {
const char* const kUsage = R"(
This tool lets you run an HLO module on one or more GPUs.
You can also pass in debug option flags for the HloModule.

Note that SPMD options are set inside the module header (number of partitions
and number of replicas), as those are fixed for a given module.

Usage:

  bazel run hlo_runner_main -- /path/to/module.hlo

The tool can be used to just compile the HLO and not run it:

  bazel run hlo_runner_main -- /path/to/module1.hlo --run=false

Note that multiple HLOs can also be launched:

  bazel run hlo_runner_main -- /path/to/module1.hlo /path/to/module2.hlo

If multiple HLOs are launched, we assume that they are encoded in the same
format (HLO text by default). Running multiple HLOs is convenient when replaying
all HLOs from an execution dump, with e.g.:

  bazel run hlo_runner_main -- /dump/*before_optimizations.txt


Mock GPU usage:
  bazel run hlo_runner_main -- --enable_mock_gpu=true /path/to/hlo_module.hlo

Tip: If the input generation takes too long or uses too much host memory,
consider using --hlo_argument_mode=uninitialized.
)";

struct HloRunnerConfig {
  std::string input_format_str = "text";
  xla::InputFormat input_format;
  bool should_run = true;
  bool enable_mock_nccl = false;
  std::string dump_output_literal_to = "";
  int task_id = 0;
  int num_nodes = 1;
  std::string device_type_str = "gpu";
  std::string address_str = "";
  int32_t num_replicas = -1;
  int32_t num_partitions = 1;
  bool log_output = false;
  bool run_xla_backend_only = false;
  bool disable_all_hlo_passes = false;
  bool use_spmd_partitioning = false;
  bool is_spmd_partitioned_module = false;
  std::string xla_dump_to = "";
  bool xla_dump_as_text = false;
  bool xla_dump_as_proto = false;
  std::string hlo_argument_mode = "use_random_inputs";
  int32_t while_execution_count = -1;
  bool remove_infeed_outfeed = true;
  int32_t num_repeats = 1;
  std::string execution_options_path = "";
};

}  // namespace

namespace xla {

static void PreprocessFlags(HloRunnerConfig& opts) {
  if (opts.is_spmd_partitioned_module) {
    opts.use_spmd_partitioning = true;
    opts.disable_all_hlo_passes = true;
  } else if (opts.use_spmd_partitioning) {
    opts.disable_all_hlo_passes = false;
  }
}

static absl::StatusOr<FunctionalHloRunner::ModuleArgumentMode>
ArgumentModeFromString(absl::string_view text) {
  if (text == "use_device_id_as_input") {
    return FunctionalHloRunner::ModuleArgumentMode::kUseDeviceIdAsInput;
  } else if (text == "use_random_inputs") {
    return FunctionalHloRunner::ModuleArgumentMode::kUseRandomInputs;
  } else if (text == "use_shared_random_inputs") {
    return FunctionalHloRunner::ModuleArgumentMode::kUseSharedRandomInputs;
  } else if (text == "use_zeros_as_input") {
    return FunctionalHloRunner::ModuleArgumentMode::kUseZerosAsInput;
  } else if (text == "uninitialized") {
    return FunctionalHloRunner::ModuleArgumentMode::kUninitialized;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unrecognized module argument mode specified. Expect "
                   "\"use_device_id_as_input\", \"use_random_inputs\", or "
                   "\"use_shared_random_inputs\"., got: ",
                   text));
}

static absl::StatusOr<FunctionalHloRunner::PreprocessingOptions>
PreprocessingOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::PreprocessingOptions out;
  out.spmd_partitioned_mode =
      opts.is_spmd_partitioned_module
          ? FunctionalHloRunner::SpmdPartitionedMode::kIsSpmdPartitionedModule
          : FunctionalHloRunner::SpmdPartitionedMode::
                kIsNotSpmdPartitionedModule;
  out.while_execution_count =
      opts.while_execution_count > 0
          ? std::make_optional(opts.while_execution_count)
          : std::nullopt;
  out.remove_infeed_outfeed = opts.remove_infeed_outfeed;
  return out;
}

static absl::StatusOr<FunctionalHloRunner::RunningOptions>
RunningOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::RunningOptions out;
  TF_ASSIGN_OR_RETURN(out.module_argument_mode,
                      ArgumentModeFromString(opts.hlo_argument_mode));

  out.module_output_mode =
      FunctionalHloRunner::ModuleOutputMode::kReturnOutputs;
  out.num_repeats = static_cast<size_t>(opts.num_repeats);
  out.log_input_output_mode =
      opts.log_output ? FunctionalHloRunner::LogOutputMode::kLogOutput
                      : FunctionalHloRunner::LogOutputMode::kNotLogOutput;
  return out;
}

static absl::StatusOr<FunctionalHloRunner::RawCompileOptions>
RawCompileOptionsFromFlags(const HloRunnerConfig& opts) {
  FunctionalHloRunner::RawCompileOptions out;
  out.hlo_passes_mode =
      opts.run_xla_backend_only
          ? FunctionalHloRunner::HloPassesMode::kRunXLABackendOnly
          : (opts.disable_all_hlo_passes
                 ? FunctionalHloRunner::HloPassesMode::kDisableAllHloPasses
                 : FunctionalHloRunner::HloPassesMode::kStandardCompile);
  out.spmd_mode = opts.use_spmd_partitioning
                      ? FunctionalHloRunner::SpmdMode::kUseSpmdPartitioning
                      : FunctionalHloRunner::SpmdMode::kNotUseSpmdPartitioning;
  if (!opts.execution_options_path.empty()) {
    TF_ASSIGN_OR_RETURN(
        out.execution_options,
        FunctionalHloRunner::LoadExecutionOptions(opts.execution_options_path));
  }
  out.num_replicas = opts.num_replicas < 0
                         ? std::nullopt
                         : std::optional<int>(opts.num_replicas);
  out.num_partitions = opts.num_partitions < 0
                           ? std::nullopt
                           : std::optional<int>(opts.num_partitions);
  out.xla_dump_to = opts.xla_dump_to;
  out.xla_text_dump_mode =
      opts.xla_dump_as_text
          ? FunctionalHloRunner::XlaTextDumpMode::kDumpAsText
          : FunctionalHloRunner::XlaTextDumpMode::kNotDumpAsText;
  out.xla_proto_dump_mode =
      opts.xla_dump_as_proto
          ? FunctionalHloRunner::XlaProtoDumpMode::kDumpAsProto
          : FunctionalHloRunner::XlaProtoDumpMode::kNotDumpAsProto;
  return out;
}

static absl::Status RunMultihostHloRunner(int argc, char** argv,
                                          HloRunnerConfig& opts) {
  if (std::string error;
      !AbslParseFlag(opts.input_format_str, &opts.input_format, &error)) {
    return absl::InvalidArgumentError(error);
  }
  TF_RET_CHECK(opts.device_type_str == "gpu" || opts.device_type_str == "host");
  PreprocessFlags(opts);

  TF_ASSIGN_OR_RETURN(
      xla::FunctionalHloRunner::PreprocessingOptions preproc_options,
      PreprocessingOptionsFromFlags(opts));
  TF_ASSIGN_OR_RETURN(
      xla::FunctionalHloRunner::RawCompileOptions raw_compile_options,
      RawCompileOptionsFromFlags(opts));
  TF_ASSIGN_OR_RETURN(xla::FunctionalHloRunner::RunningOptions running_options,
                      RunningOptionsFromFlags(opts));

  // tsl::Flags::Parse() leaves unknown flags in argv, we assume that those are
  // HLO files to run. Note that argv[0] is the binary name and is excluded.
  QCHECK_GT(argc, 1) << "No HLO file specified";
  QCHECK(opts.dump_output_literal_to.empty() || argc == 2)
      << "Can only dump output literal when single input file is specified";

  std::unique_ptr<DistributedRuntimeService> service;
  std::shared_ptr<KeyValueStoreInterface> kv_store;

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtClient> client,
      GetPjRtClient(opts.device_type_str, opts.address_str, opts.task_id,
                    opts.num_nodes, opts.enable_mock_nccl, service, kv_store));

  for (int c = 1; c < argc; c++) {
    const char* filename = argv[c];
    std::cout << "\n** Running " << filename << " **\n";
    if (opts.should_run) {
      TF_RETURN_IF_ERROR(xla::FunctionalHloRunner::LoadAndRunAndDump(
          *client, GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, running_options, filename, opts.input_format,
          opts.dump_output_literal_to, opts.task_id));
    } else {
      TF_RETURN_IF_ERROR(FunctionalHloRunner::LoadAndCompile(
          *client, GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, argv[c], opts.input_format, opts.task_id));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla

int main(int argc, char** argv) {
  HloRunnerConfig opts;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &opts.input_format_str,
                "HLO input mode: text, proto_text, proto_binary, or "
                "snapshot_proto_binary"),
      tsl::Flag("run", &opts.should_run, "Should we run the compiled HLO?"),
      tsl::Flag("dump_output_literal_to", &opts.dump_output_literal_to,
                "A path to which the HLO output will be dumped. "
                "Example: /a/b/literal.txt."),
      tsl::Flag("task_id", &opts.task_id, "Borg task id."),
      tsl::Flag("device_type", &opts.device_type_str, "Device type: gpu, host"),
      tsl::Flag("num_nodes", &opts.num_nodes,
                "Number of nodes (hosts). If greater than 1, a distributed "
                "service will be created for task_id 0"),
      tsl::Flag(
          "enable_mock_nccl", &opts.enable_mock_nccl,
          "Should we simulate multi-hosts run with mock nccl collectives?"),
      tsl::Flag("address", &opts.address_str,
                "Coordinator address with port for when num_nodes > 1. "
                "Example: 127.0.0.1:12345"),
      tsl::Flag("num_replicas", &opts.num_replicas,
                "The number of replicas; set to -1 for multihost "
                "execution, which then uses all devices on all host."),
      tsl::Flag("num_partitions", &opts.num_partitions,
                "Number of partitions for SPMD."),
      tsl::Flag("log_output", &opts.log_output,
                "Log the input and output to stderr."),
      tsl::Flag("run_xla_backend_only", &opts.run_xla_backend_only,
                "Call only XLA's RunBackend during the compilation. "
                "This is used to run a post-optimization HLO module "
                "(dumped as 'xxx.after_optimizations.hlo.xxx'"),
      tsl::Flag("disable_all_hlo_passes", &opts.disable_all_hlo_passes,
                "Disable HLO passes or not."),
      tsl::Flag("use_spmd_partitioning", &opts.use_spmd_partitioning,
                "Partition the module using SPMD."),
      tsl::Flag("is_spmd_partitioned_module", &opts.is_spmd_partitioned_module,
                "The module is the partitioned result of SPMD. Setting this "
                "flag also "
                "disables all HLO passes and sets use_spmd_partitioning."),
      tsl::Flag("xla_dump_to", &opts.xla_dump_to,
                "A directory to dump xla debug data to."),
      tsl::Flag("xla_dump_as_text", &opts.xla_dump_as_text,
                "Whether to dump xla debug data as text."),
      tsl::Flag("xla_dump_as_proto", &opts.xla_dump_as_proto,
                "Whether to dump xla debug data as protobuf."),
      tsl::Flag("hlo_argument_mode", &opts.hlo_argument_mode,
                "Specify how arguments to the HLO module are generated. "
                "Accepted values: "
                "use_device_id_as_input, use_random_inputs, "
                "use_shared_random_inputs, "
                "use_zeros_as_input or uninitialized."),
      tsl::Flag("while_execution_count", &opts.while_execution_count,
                "If set to a positive number, flatten all while loops to "
                "a certain number of iterations."),
      tsl::Flag("remove_infeed_outfeed", &opts.remove_infeed_outfeed,
                "If set, we will remove all infeed and outfeed operations."),
      tsl::Flag("num_repeats", &opts.num_repeats,
                "Repeatedly execute the HLO for this many times."),
      tsl::Flag("execution_options_path", &opts.execution_options_path,
                "A path to a protobuf text file which stores the "
                "ExecutionOptions message for this HLO module.")};

  xla::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    LOG(QFATAL) << kUsageString;
  }
  absl::Status s = xla::RunMultihostHloRunner(argc, argv, opts);
  if (!s.ok()) {
    std::cerr << s;
    return 1;
  }

  return 0;
}
