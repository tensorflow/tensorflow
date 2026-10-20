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

#include <signal.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/log_severity.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/backends/gpu/tests/multi_process_gpu_test_base.h"
#include "xla/debug_options_flags.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/service.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/subprocess.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/path.h"

namespace {

bool IsWorkerInternalFlag(absl::string_view arg) {
  return absl::StartsWith(arg, "--multiprocess_worker_id=") ||
         absl::StartsWith(arg, "--multiprocess_coordinator_address=");
}

void WriteWorkerLogsToUndeclaredOutputs(int node_id,
                                        absl::string_view stdout_str,
                                        absl::string_view stderr_str) {
  const char* undeclared_outputs_dir =
      std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  if (undeclared_outputs_dir == nullptr || undeclared_outputs_dir[0] == '\0') {
    return;
  }
  if (!stdout_str.empty()) {
    std::string stdout_file = tsl::io::JoinPath(
        undeclared_outputs_dir,
        absl::StrFormat("subprocess_node_%d_stdout.log", node_id));
    absl::Status status =
        tsl::WriteStringToFile(tsl::Env::Default(), stdout_file, stdout_str);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to write stdout to " << stdout_file << ": "
                   << status;
    }
  }
  if (!stderr_str.empty()) {
    std::string stderr_file = tsl::io::JoinPath(
        undeclared_outputs_dir,
        absl::StrFormat("subprocess_node_%d_stderr.log", node_id));
    absl::Status status =
        tsl::WriteStringToFile(tsl::Env::Default(), stderr_file, stderr_str);
    if (!status.ok()) {
      LOG(WARNING) << "Failed to write stderr to " << stderr_file << ": "
                   << status;
    }
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  // Snapshot original argv before flag parsers mutate argc/argv so the parent
  // orchestrator can forward all flags (including XLA and GTest flags) to
  // worker subprocesses.
  std::vector<std::string> original_argv;
  original_argv.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    original_argv.emplace_back(argv[i]);
  }

  int num_workers = 2;
  int gpus_per_worker = 1;
  int worker_id = -1;
  std::string coordinator_address;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("multiprocess_num_workers", &num_workers,
                "Number of worker processes to launch."),
      tsl::Flag("multiprocess_gpus_per_worker", &gpus_per_worker,
                "Number of GPUs assigned per worker process."),
      tsl::Flag("multiprocess_worker_id", &worker_id,
                "Internal flag: worker rank in [0, num_workers). -1 for the "
                "parent orchestrator."),
      tsl::Flag("multiprocess_coordinator_address", &coordinator_address,
                "Internal flag: address of the DistributedRuntimeService "
                "coordinator."),
  };

  xla::AppendDebugOptionsFlags(&flag_list);
  tsl::Flags::Parse(&argc, argv, flag_list);
  testing::InitGoogleTest(&argc, argv);

  // Worker subprocess mode: connect to the coordinator and run tests.
  if (worker_id >= 0) {
    const char* undeclared_outputs_dir =
        std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (undeclared_outputs_dir != nullptr &&
        undeclared_outputs_dir[0] != '\0') {
      std::string worker_outputs_dir = tsl::io::JoinPath(
          undeclared_outputs_dir, absl::StrFormat("node_%d", worker_id));
      absl::Status dir_status =
          tsl::Env::Default()->RecursivelyCreateDir(worker_outputs_dir);
      if (!dir_status.ok()) {
        LOG(WARNING) << "Failed to create worker outputs directory "
                     << worker_outputs_dir << ": " << dir_status;
      } else {
        tsl::setenv("TEST_UNDECLARED_OUTPUTS_DIR", worker_outputs_dir.c_str(),
                    /*overwrite=*/true);
      }
    }
    xla::InitializeMultiProcessGpuWorker(worker_id, num_workers,
                                         gpus_per_worker, coordinator_address);
    int result = RUN_ALL_TESTS();
    xla::ShutDownMultiProcessGpuWorker();
    return result;
  }

  // Parent orchestrator mode.
  xla::ConfigureMultiProcessGpuParent(num_workers, gpus_per_worker);

  if (GTEST_FLAG_GET(list_tests)) {
    return RUN_ALL_TESTS();
  }

  const int required_gpus = num_workers * gpus_per_worker;
  absl::StatusOr<stream_executor::Platform*> platform =
      xla::PlatformUtil::GetPlatform("gpu");
  if (!platform.ok() || (*platform)->VisibleDeviceCount() < required_gpus) {
    LOG(WARNING) << "Multi-process GPU test requires at least " << required_gpus
                 << " GPUs, but found "
                 << (platform.ok() ? (*platform)->VisibleDeviceCount() : 0)
                 << "; running in-process to skip tests.";
    return RUN_ALL_TESTS();
  }

  // Re-export XLA_FLAGS so child processes inherit it.
  const char* xla_flags = std::getenv("XLA_FLAGS");
  if (xla_flags != nullptr) {
    tsl::setenv("XLA_FLAGS", xla_flags, /*overwrite=*/true);
  }

  // Ensure all workers use the exact same shuffle seed when test shuffling is
  // enabled without an explicit non-zero seed. Otherwise each worker
  // independently samples the current millisecond clock and may shuffle tests
  // into different orders, deadlocking on per-test barriers.
  if (GTEST_FLAG_GET(shuffle) && GTEST_FLAG_GET(random_seed) == 0) {
    int32_t seed = 0;
    const char* env_seed = std::getenv("TEST_RANDOM_SEED");
    if (env_seed == nullptr || !absl::SimpleAtoi(env_seed, &seed) ||
        seed <= 0) {
      seed = static_cast<int32_t>(absl::ToUnixMillis(absl::Now()) % 99999);
    } else {
      seed = seed % 99999;
    }
    if (seed <= 0) {
      seed = 1;
    }
    const std::string seed_str = absl::StrCat(seed);
    tsl::setenv("GUNIT_RANDOM_SEED", seed_str.c_str(), /*overwrite=*/true);
    tsl::setenv("GTEST_RANDOM_SEED", seed_str.c_str(), /*overwrite=*/true);
  }

  const int port = tsl::testing::PickUnusedPortOrDie();
  xla::CoordinationServiceImpl::Options service_options;
  service_options.num_nodes = num_workers;
  absl::StatusOr<std::unique_ptr<xla::DistributedRuntimeService>> service =
      xla::GetDistributedRuntimeService(absl::StrFormat("[::]:%d", port),
                                        service_options);
  CHECK_OK(service.status()) << "Failed to start DistributedRuntimeService";
  const std::string resolved_coordinator_address =
      absl::StrFormat("127.0.0.1:%d", port);

  std::vector<std::unique_ptr<tsl::SubProcess>> children(num_workers);
  for (int node_id = 0; node_id < num_workers; ++node_id) {
    if (node_id > 0) {
      // Only worker 0 should write Bazel/GoogleTest XML output and shard status
      // files so workers 1..N-1 do not race on or overwrite them.
      tsl::unsetenv("XML_OUTPUT_FILE");
      tsl::unsetenv("GUNIT_OUTPUT");
      tsl::unsetenv("GTEST_OUTPUT");
      tsl::unsetenv("TEST_SHARD_STATUS_FILE");
      tsl::unsetenv("TEST_PREMATURE_EXIT_FILE");
    }

    std::vector<std::string> worker_argv;
    worker_argv.reserve(original_argv.size() + 3);
    for (const std::string& arg : original_argv) {
      if (IsWorkerInternalFlag(arg)) {
        continue;
      }
      if (arg == "--gunit_random_seed=0" || arg == "--gtest_random_seed=0") {
        continue;
      }
      if (node_id > 0 && (absl::StartsWith(arg, "--gtest_output=") ||
                          absl::StartsWith(arg, "--gunit_output="))) {
        continue;
      }
      worker_argv.push_back(arg);
    }
    worker_argv.push_back(
        absl::StrFormat("--multiprocess_worker_id=%d", node_id));
    worker_argv.push_back(absl::StrFormat(
        "--multiprocess_coordinator_address=%s", resolved_coordinator_address));

    children[node_id] = std::make_unique<tsl::SubProcess>();
    children[node_id]->SetProgram(original_argv[0], worker_argv);
    children[node_id]->SetChannelAction(tsl::CHAN_STDOUT, tsl::ACTION_PIPE);
    children[node_id]->SetChannelAction(tsl::CHAN_STDERR, tsl::ACTION_PIPE);
    CHECK(children[node_id]->Start())
        << "Failed to start worker process " << node_id;
  }

  std::vector<int> statuses(num_workers, -1);
  std::vector<std::string> stdout_strs(num_workers);
  std::vector<std::string> stderr_strs(num_workers);
  std::atomic<int> remaining_workers{num_workers};
  std::atomic<bool> killer_triggered{false};
  absl::Notification all_done;
  {
    tsl::thread::ThreadPool pool(tsl::Env::Default(), "multiprocess_workers",
                                 num_workers);
    for (int node_id = 0; node_id < num_workers; ++node_id) {
      pool.Schedule([&, node_id]() {
        statuses[node_id] = children[node_id]->Communicate(
            nullptr, &stdout_strs[node_id], &stderr_strs[node_id]);
        if (remaining_workers.fetch_sub(1) == 1) {
          all_done.Notify();
        } else if (statuses[node_id] != 0 && !killer_triggered.exchange(true)) {
          if (!all_done.WaitForNotificationWithTimeout(absl::Seconds(5))) {
            LOG(ERROR) << "Worker " << node_id
                       << " exited with non-zero status " << statuses[node_id]
                       << "; killing remaining workers after grace period.";
            for (int i = 0; i < num_workers; ++i) {
              children[i]->Kill(SIGKILL);
            }
          }
        }
      });
    }
  }

  if (!stdout_strs.empty() && !stdout_strs[0].empty()) {
    std::cout << stdout_strs[0] << std::flush;
  }

  bool any_failed = false;
  for (int node_id = 0; node_id < num_workers; ++node_id) {
    WriteWorkerLogsToUndeclaredOutputs(node_id, stdout_strs[node_id],
                                       stderr_strs[node_id]);
    if (statuses[node_id] != 0) {
      any_failed = true;
      LOG(ERROR) << "Worker " << node_id << " failed with status "
                 << statuses[node_id] << "\nstdout:\n"
                 << stdout_strs[node_id] << "\nstderr:\n"
                 << stderr_strs[node_id];
    }
  }

  return any_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
