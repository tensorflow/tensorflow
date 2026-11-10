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

#ifndef XLA_SERVICE_SCHEDULER_CONFIG_H_
#define XLA_SERVICE_SCHEDULER_CONFIG_H_

#include <cstdint>
#include <string>

namespace xla {

struct SchedulerConfig {
  int64_t collective_broadcast_overlap_limit = 1;
  int64_t collective_permute_overlap_limit = 1;
  int64_t all_to_all_overlap_limit = 1;
  int64_t ragged_all_to_all_overlap_limit = 1;
  int64_t all_gather_overlap_limit = 1;
  int64_t all_reduce_overlap_limit = 1;
  int64_t reduce_scatter_overlap_limit = 1;
  int64_t send_recv_overlap_limit = 1;
  int64_t send_recv_host_overlap_limit = 1;
  int64_t copy_overlap_limit = 1;
  uint64_t memory_limit = UINT64_MAX;
  int64_t max_hops_to_closest_selective_overlap = 0;
  int64_t rerun = 0;
  int64_t parallel_collective_overlap_limit = 1;
  int64_t parallel_scale_up_collective_overlap_limit = 1;
  int64_t parallel_async_compute_limit = 2;
  bool schedule_send_recvs = false;
  bool deannotate_group_if_blocked = false;
  // Consider send recv as the same resource. Some platforms do not take well
  // overlapping the send/recv ops between themselves.
  bool force_send_recv_to_use_same_resource = false;
  bool use_real_cost_model = false;
  bool aggressive_scheduling_policies = false;
  bool prioritize_async_depth_over_stall = false;
  bool enable_release_start_policy = false;
  bool resource_sharing = false;
  bool resource_serializing = false;
  bool depth_based_memory_pressure_reduction = false;
  bool enable_selective_resources = false;

  // Freely schedule nodes at the start of a scheduling group, allowing the
  // scheduler to delay them to promote better overlap.
  bool flexible_scheduling_annotation_scheduling = false;
  // If the above flag is also set, force the scheduler to provide maximum delay
  // to nodes at the stat of a scheduling group.
  bool aggressive_flexible_annotation_scheduling = false;
  // Prioritize  flexible annotation scheduling over memory pressure; this is
  // useful when the memory pressure is high. Without this, under high memory
  // pressure, aggressive_flexible_annotation_scheduling is not respected.
  bool force_delay_over_memory_pressure = false;
  // If true, estimate the fragmentation size of the module by running the heap
  // simulator.
  bool estimate_fragmentation_size = false;
  // If true, track the resource usage of sync ops in latency hiding scheduler.
  bool track_sync_op_resource_usage = false;
  // If true, use top down scheduling.
  bool top_down_scheduling = false;
  // If true, enable schedule by structure.
  bool enable_schedule_by_structure = false;
  // If true (and enable_schedule_by_structure is true), synchronous
  // collectives are used as schedule anchors by the target-specific
  // schedule-by-structure (critical path depth) heuristic.
  bool enable_cpd_for_sync_collective = true;
  // If set, only log computations that match the given regular expression.
  std::string log_computation_re;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCHEDULER_CONFIG_H_
