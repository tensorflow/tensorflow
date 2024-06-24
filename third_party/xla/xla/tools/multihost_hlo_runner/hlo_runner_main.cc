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

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/statusor.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_runner_flags.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

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

  bazel run hlo_runner_main -- /dump/*before_optimizations*.txt

Mock GPU usage:
  bazel run hlo_runner_main -- --enable_mock_gpu=true /path/to/hlo_module.hlo

Tip: If the input generation takes too long or uses too much host memory,
consider using --hlo_argument_mode=uninitialized.
)";

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetClient(
    const std::string& device_type_str, bool enable_mock_nccl, int num_nodes,
    const std::string& address_str, int task_id,
    std::unique_ptr<xla::DistributedRuntimeService>* service) {
  if (device_type_str == "host") {
    CHECK_EQ(num_nodes, 1);
    return xla::FunctionalHloRunner::CreateHostClient();
  }

  CHECK_EQ(device_type_str, "gpu");

  if (enable_mock_nccl) {
    CHECK_GT(num_nodes, 1);
    return xla::FunctionalHloRunner::CreateMockGpuClient(num_nodes);
  } else {
    if (num_nodes == 1) {
      return xla::FunctionalHloRunner::CreateGpuClient();
    } else {
      CHECK_GT(address_str.length(), 0);
      // Multinode. Start service on task 0.
      if (task_id == 0) {
        std::string coordinator_bind_address =
            "[::]:" + address_str.substr(address_str.rfind(":") + 1);
        xla::CoordinationServiceImpl::Options options;
        options.num_nodes = num_nodes;
        auto status_or = xla::GetDistributedRuntimeService(
            coordinator_bind_address, options);
        TF_QCHECK_OK(status_or.status());
        *service = std::move(status_or.value());
      }
      xla::DistributedRuntimeClient::Options options;
      options.node_id = task_id;
      options.init_timeout = absl::Seconds(300);
      auto distributed_client =
          xla::GetDistributedRuntimeClient(address_str, options);
      TF_QCHECK_OK(distributed_client->Connect());
      return xla::FunctionalHloRunner::CreateGpuClient(distributed_client,
                                                       task_id, num_nodes);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_format_str = "text";
  xla::InputFormat input_format;
  bool should_run = true;
  bool enable_mock_nccl = false;
  std::string dump_output_literal_to = "";
  int task_id = 0;
  int num_nodes = 1;
  std::string device_type_str = "gpu";
  std::string address_str = "";
  xla::FunctionalHloRunner::PreprocessingOptions preproc_options;
  xla::FunctionalHloRunner::RawCompileOptions raw_compile_options;
  xla::FunctionalHloRunner::RunningOptions running_options;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_format", &input_format_str,
                "HLO input mode: text, proto_text, proto_binary, or "
                "snapshot_proto_binary"),
      tsl::Flag("run", &should_run, "Should we run the compiled HLO?"),
      tsl::Flag("dump_output_literal_to", &dump_output_literal_to,
                "A path to which the HLO output will be dumped. "
                "Example: /a/b/literal.txt."),
      tsl::Flag("task_id", &task_id, "Borg task id."),
      tsl::Flag("device_type", &device_type_str, "Device type: gpu, host"),
      tsl::Flag("num_nodes", &num_nodes,
                "Number of nodes (hosts). If greater than 1, a distributed "
                "service will be created for task_id 0"),
      tsl::Flag(
          "enable_mock_nccl", &enable_mock_nccl,
          "Should we simulate multi-hosts run with mock nccl collectives?"),
      tsl::Flag("address", &address_str,
                "Coordinator address with port for when num_nodes > 1. "
                "Example: 127.0.0.1:12345"),
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

  // tsl::Flags::Parse() leaves unknown flags in argv, we assume that those are
  // HLO files to run. Note that argv[0] is the binary name and is excluded.
  QCHECK_GT(argc, 1) << "No HLO file specified";

  QCHECK(dump_output_literal_to.empty() || argc == 2)
      << "Can only dump output literal when single input file is specified";

  std::unique_ptr<xla::DistributedRuntimeService> service;
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client =
      GetClient(device_type_str, enable_mock_nccl, num_nodes, address_str,
                task_id, &service);
  TF_QCHECK_OK(client.status());

  for (int c = 1; c < argc; c++) {
    const char* filename = argv[c];
    std::cout << "\n** Running " << filename << " **\n";
    if (should_run) {
      TF_QCHECK_OK(xla::FunctionalHloRunner::LoadAndRunAndDump(
          *client.value(), xla::GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, running_options, {filename}, input_format,
          dump_output_literal_to, task_id));
    } else {
      TF_QCHECK_OK(xla::FunctionalHloRunner::LoadAndCompile(
          *client.value(), xla::GetDebugOptionsFromFlags(), preproc_options,
          raw_compile_options, {argv[c]}, input_format, task_id));
    }
  }

  return 0;
}
