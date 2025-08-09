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

#ifndef XLA_SERVICE_GPU_MODEL_COLLECTIVE_PTABLE_STATS_COLLECTION_H_
#define XLA_SERVICE_GPU_MODEL_COLLECTIVE_PTABLE_STATS_COLLECTION_H_

#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

class CollectivePerfTableStatsCollection : public HloModulePass {
 public:
  explicit CollectivePerfTableStatsCollection(
      absl::string_view perf_table_path,
      const se::DeviceDescription& device_info)
      : perf_table_path_(perf_table_path), device_info_(device_info) {}

  absl::string_view name() const override {
    return "collective-perf-table-stats-collection";
  }

  using HloPassInterface::Run;

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const std::string perf_table_path_;
  const se::DeviceDescription& device_info_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_COLLECTIVE_PTABLE_STATS_COLLECTION_H_
