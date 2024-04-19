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

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner_flags.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace {
const char* const kUsage = R"(
This tool lets you run an HLO module on one or more GPUs.
You can also pass in debug option flags for the HloModule.

Usage:

Single-GPU HLO:

  bazel run hlo_runner_main -- \
    --num_replicas=1 \
    --num_partitions=1 \
    --hlo_file=path/to/hlo_module

2-GPU sharded HLO:

  bazel run hlo_runner_main -- \
    --use_spmd_partitioning=true \
    --num_replicas=1 \
    --num_partitions=2 \
    --hlo_file=path/to/hlo_module

2 hosts-Mock GPU sharded HLO:
  bazel run hlo_runner_main -- \
     --use_spmd_partitioning=true \
    --num_replicas=1 \
    --num_partitions=2 \
    --num_nodes=2 \
    --enable_mock_gpu=true \
    --hlo_file=path/to/hlo_module

Tip: If the input generation takes too long or uses too much host memory,
consider using --hlo_argument_mode=uninitialized.
)";

}  // namespace

int main(int argc, char** argv) {
  std::string input_format_str = "text";
  xla::InputFormat input_format;
  std::string hlo_file = "";
  bool should_run = true;
  bool enable_mock_nccl = false;
  std::string dump_output_literal_to = "";
  int task_id = 0;
  int num_nodes = 1;
  std::string device_type_str = "gpu";
  xla::FunctionalHloRunner::PreprocessingOptions preproc_options;
  xla::FunctionalHloRunner::RawCompileOptions raw_compile_options;
  xla::FunctionalHloRunner::RunningOptions running_options;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &input_format_str,
                "HLO input mode: text, proto_text, proto_binary, or "
                "snapshot_proto_binary"),
      tsl::Flag("hlo_file", &hlo_file,
                "A text or proto buf file for HLO input"),
      tsl::Flag("run", &should_run, "Should we run the compiled HLO?"),
      tsl::Flag("dump_output_literal_to", &dump_output_literal_to,
                "A path to which the HLO output will be dumped. "
                "Example: /a/b/literal.txt."),
      tsl::Flag("task_id", &task_id, "Borg task id."),
      tsl::Flag("device_type", &device_type_str, "Device type: gpu, host"),
      tsl::Flag("num_nodes", &num_nodes, "Number of nodes (hosts)"),
      tsl::Flag(
          "enable_mock_nccl", &enable_mock_nccl,
          "Should we simulate multi-hosts run with mock nccl collectives?"),
  };

  xla::MultiHostHloRunnerFlags hlo_runner_flags;
  hlo_runner_flags.AppendFlags(&flag_list);
  xla::AppendDebugOptionsFlags(&flag_list);

  // The usage string includes the message at the top of the file, the
  // DebugOptions flags and the flags defined above.
  const std::string kUsageString =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  std::string parse_error;
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  parse_ok = parse_ok &&
             xla::AbslParseFlag(input_format_str, &input_format, &parse_error);
  parse_ok =
      parse_ok && (device_type_str == "gpu" || device_type_str == "host");
  parse_ok = parse_ok && hlo_runner_flags.CreateOptionsFromFlags(
                             &preproc_options, &raw_compile_options,
                             &running_options, &parse_error);

  tsl::port::InitMain(kUsageString.c_str(), &argc, &argv);
  if (!parse_ok) {
    if (!parse_error.empty()) {
      LOG(QFATAL) << kUsageString << "\n\n" << parse_error;
    } else {
      LOG(QFATAL) << kUsageString;
    }
  }

  // The main logic:
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client = [&] {
    if (device_type_str == "host") {
      CHECK_EQ(num_nodes, 1);
      return xla::FunctionalHloRunner::CreateHostClient();
    }

    CHECK_EQ(device_type_str, "gpu");

    if (enable_mock_nccl) {
      CHECK_GT(num_nodes, 1);
      return xla::FunctionalHloRunner::CreateMockGpuClient(num_nodes);
    } else {
      CHECK_EQ(num_nodes, 1);
      return xla::FunctionalHloRunner::CreateGpuClient();
    }
  }();

  TF_QCHECK_OK(client.status());

  if (should_run) {
    TF_QCHECK_OK(xla::FunctionalHloRunner::LoadAndRunAndDump(
        *client.value(), xla::GetDebugOptionsFromFlags(), preproc_options,
        raw_compile_options, running_options, {hlo_file}, input_format,
        dump_output_literal_to, task_id));
  } else {
    TF_QCHECK_OK(xla::FunctionalHloRunner::LoadAndCompile(
        *client.value(), xla::GetDebugOptionsFromFlags(), preproc_options,
        raw_compile_options, hlo_file, input_format, task_id));
  }

  return 0;
}
