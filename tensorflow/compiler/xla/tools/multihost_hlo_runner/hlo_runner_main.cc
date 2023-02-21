/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_flags.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/util/command_line_flags.h"

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

Tip: If the input generation takes too long, consider using
--hlo_argument_mode=use_zeros_as_input.
)";

}  // namespace

int main(int argc, char** argv) {
  std::string input_format_str = "text";
  xla::InputFormat input_format;
  std::string hlo_file = "";
  std::string dump_output_literal_to = "";
  int task_id = 0;
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
      tsl::Flag("dump_output_literal_to", &dump_output_literal_to,
                "A path to which the HLO output will be dumped. "
                "Example: /a/b/literal.txt."),
      tsl::Flag("task_id", &task_id, "Borg task id."),
      tsl::Flag("device_type", &device_type_str, "Device type: gpu"),
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
  parse_ok = parse_ok && device_type_str == "gpu";
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
  xla::StatusOr<std::unique_ptr<xla::PjRtClient>> client =
      xla::FunctionalHloRunner::CreateGpuClient();
  TF_QCHECK_OK(client.status());

  TF_QCHECK_OK(xla::FunctionalHloRunner::LoadAndRunAndDump(
      *client.value(), preproc_options, raw_compile_options, running_options,
      {hlo_file}, input_format, dump_output_literal_to, task_id));

  return 0;
}
