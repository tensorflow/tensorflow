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

#ifndef XLA_TOOLS_MATMUL_PERF_TABLE_GEN_H_
#define XLA_TOOLS_MATMUL_PERF_TABLE_GEN_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/platform_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

class MatmulPerfTableGen {
 public:
  struct DataTypeSpec {
    std::string lhs_dtype;
    std::string rhs_dtype;
    std::string out_dtype;
  };

  struct StepSpec {
    int start = 0;
    int stop = -1;
    int step = 0;
    int factor = 0;
  };

  struct Config {
    static constexpr absl::string_view kStdout = "stdout";

    // Search space.
    std::vector<DataTypeSpec> dtypes;
    StepSpec b_spec;
    StepSpec m_spec;
    StepSpec n_spec;
    StepSpec k_spec;
    std::string hlo_scan_path;

    // Execution opts.
    // Run matrix multiplications but do not trace.
    bool dry_run;
    std::string output = std::string(kStdout);
  };

  explicit MatmulPerfTableGen(Config config)
      : runner_(PlatformUtil::GetPlatform("gpu").value()),
        config_(std::move(config)) {};

  // Computes a performance table for a given `config`.
  DeviceHloInstructionProfiles ComputeTable();

  // Dumps a performance `table` to a given `output_file` from `Config`.
  absl::Status Dump(const DeviceHloInstructionProfiles& table);

  // Dumps a performance `table` to a given `output_file` from `Config`.
  absl::Status Dump(const GemmPerfTable& table);

  // Reads, deduplicates and merges multiple `xla.gpu.DeviceInstructionProfiles`
  // residing in a given `filepath`.
  absl::StatusOr<DeviceHloInstructionProfiles> Merge(
      absl::string_view filepath);

  static absl::StatusOr<GemmPerfTable> Compact(
      const DeviceHloInstructionProfiles& profiles);

 private:
  std::unique_ptr<OpaqueExecutable> Compile(std::unique_ptr<HloModule> module);

  absl::Duration Profile(std::unique_ptr<HloModule> module);

  HloRunner runner_;

  Config config_;
};

}  // namespace xla::gpu

#endif  // XLA_TOOLS_MATMUL_PERF_TABLE_GEN_H_
