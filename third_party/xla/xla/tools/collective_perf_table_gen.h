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

#ifndef XLA_TOOLS_COLLECTIVE_PERF_TABLE_GEN_H_
#define XLA_TOOLS_COLLECTIVE_PERF_TABLE_GEN_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/service/backend.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Generates performance table for collectives given a static specification. The
// goal is to produce a derating curve of collectives at HLO op level.
//
// This class is not thread safe.
class CollectivePerfTableGen {
 public:
  struct StepSpec {
    int64_t start = 1024;
    int64_t stop = 2ll * 1024 * 1024 * 1024;
    int64_t step = 0;
    int64_t factor = 2;
  };

  enum class CollectiveType {
    UNSPECIFIED,
    ALL_REDUCE,
    ALL_GATHER,
    REDUCE_SCATTER,
    ALL_TO_ALL,
  };

  struct Config {
    static constexpr absl::string_view kStdout = "stdout";

    // Search space.
    StepSpec tensor_size_bytes_spec;
    std::vector<CollectiveType> collective_types = {
        CollectiveType::ALL_REDUCE,
        CollectiveType::ALL_GATHER,
        CollectiveType::REDUCE_SCATTER,
        CollectiveType::ALL_TO_ALL,
    };
    std::vector<std::string> replica_groups_list;

    // Execution opts.
    bool dry_run = false;
    std::string output = std::string(kStdout);
    std::string coordinator_address = "";
    absl::Duration connection_timeout = absl::Seconds(600);
    uint16_t num_nodes = 1;
    uint16_t task_id = 0;
  };

  struct ProfilingData {
    absl::Duration runtime = absl::Nanoseconds(42);
  };

  // Factory method to create the perf table gen.
  static std::unique_ptr<CollectivePerfTableGen> Create(Config config);

  // Computes performance table for a given `config`.
  DeviceHloInstructionProfiles ComputeTable();

  // Dumps `table` to `config_`s `output`. If the output is set to "stdout" it
  // just prints the content to output stream. If it's a filepath ending with
  // .pbtx or .pb it will dump a proto to that file, merging the previous
  // content (but not deduplicating).
  absl::Status Dump(const DeviceHloInstructionProfiles& table);

  // Merges all of the profiled files under `merge_path`, deduplicates them
  // based on fingerprint and writes them to a single
  // `DeviceHloInstructionProfiles` proto.
  DeviceHloInstructionProfiles Merge(absl::string_view merge_path);

 private:
  explicit CollectivePerfTableGen(Config config, PjRtEnvironment&& pjrt_env)
      : config_(std::move(config)),
        backend_(std::move(Backend::CreateDefaultBackend().value())),
        pjrt_env_(std::move(pjrt_env)) {}

  ProfilingData Profile(std::unique_ptr<HloModule> module);

  std::unique_ptr<PjRtLoadedExecutable> Compile(
      std::unique_ptr<HloModule> module);

  std::vector<ExecutionProfile> Run(PjRtLoadedExecutable& executable);

  Config config_;
  std::unique_ptr<Backend> backend_;
  PjRtEnvironment pjrt_env_;
};

}  // namespace xla::gpu

#endif  // XLA_TOOLS_COLLECTIVE_PERF_TABLE_GEN_H_
