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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_COST_MODEL_STATS_COLLECTION_H_
#define XLA_SERVICE_GPU_MODEL_GPU_COST_MODEL_STATS_COLLECTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Calculates costs for each fusion op and stores the result in backend
// config.
class GpuCostModelStatsCollection : public HloModulePass {
 public:
  explicit GpuCostModelStatsCollection(
      const se::DeviceDescription& d,
      const GpuHloCostAnalysis::Options& cost_analysis_options)
      : device_info_(d), cost_analysis_(cost_analysis_options, &device_info_) {}

  absl::string_view name() const override {
    return "gpu_cost_model_stats_collection";
  }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_info_;
  GpuHloCostAnalysis cost_analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_COST_MODEL_STATS_COLLECTION_H_
