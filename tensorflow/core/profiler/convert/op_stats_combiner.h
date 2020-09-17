/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_COMBINER_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_COMBINER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {

// Whether a host is a coordinator.
bool IsCoordinator(bool no_accelerator_in_system, HardwareType hardware_type);

// Translates the core id from single host to the one for multiple-host.
// We need this translation because the device_ordinal was assigned when a
// single host response was given. Now, we need a global core_id to distinguish
// it with multiple hosts.
uint32 GlobalCoreId(int host_id, uint32 device_ordinal);

// Combines the src map into the dst map.
// The src map keys are local core_ids. The src_host_id is used to convert them
// into global core_ids used as keys in the dst map.
// REQUIRED: cores from src_host_id are not already in dst.
template <typename CoreIdMap>
void CombineCoreIdMap(int src_host_id, const CoreIdMap& src, CoreIdMap* dst) {
  for (const auto& core_id_and_value : src) {
    uint32 global_core_id = GlobalCoreId(src_host_id, core_id_and_value.first);
    auto iter_and_inserted =
        dst->insert({global_core_id, core_id_and_value.second});
    DCHECK(iter_and_inserted.second)
        << "Duplicated core_id: " << iter_and_inserted.first->first;
  }
}

// Combine the src OpStats into the dst OpStats.
void CombineOpStats(
    bool no_accelerator_in_system, int src_host_id, HardwareType hardware_type,
    const StepIntersection& step_intersection, const OpStats& src, OpStats* dst,
    OpMetricsDbCombiner* host_op_metrics_db_combiner,
    OpMetricsDbCombiner* device_op_metrics_db_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_STATS_COMBINER_H_
