/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
#define XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

inline std::string ToXStat(const KernelDetails& kernel_info,
                           double occupancy_pct) {
  uint32_t grid_x = kernel_info.workgroup_x != 0
                        ? kernel_info.grid_x / kernel_info.workgroup_x
                        : 0,
           grid_y = kernel_info.workgroup_y != 0
                        ? kernel_info.grid_y / kernel_info.workgroup_y
                        : 0,
           grid_z = kernel_info.workgroup_z != 0
                        ? kernel_info.grid_z / kernel_info.workgroup_z
                        : 0;

  return absl::StrCat(" grid:", grid_x, ",", grid_y, ",", grid_z,
                      " block:", kernel_info.workgroup_x, ",",
                      kernel_info.workgroup_y, ",", kernel_info.workgroup_z,
                      " private_mem:", kernel_info.private_segment_size,
                      " group_mem:", kernel_info.group_segment_size,
                      " occ_pct:", occupancy_pct);
}

struct RocmDeviceOccupancyParams {
  hipFuncAttributes attributes = {};
  int block_size = 0;
  size_t dynamic_smem_size = 0;
  void* func_ptr;

  friend bool operator==(const RocmDeviceOccupancyParams& a,
                         const RocmDeviceOccupancyParams& b) noexcept {
    // Compare only the fields that affect occupancy decisions.
    return std::tuple{a.attributes.binaryVersion,
                      a.attributes.cacheModeCA,
                      a.attributes.constSizeBytes,
                      a.attributes.localSizeBytes,
                      a.attributes.maxDynamicSharedSizeBytes,
                      a.attributes.maxThreadsPerBlock,
                      a.attributes.numRegs,
                      a.attributes.preferredShmemCarveout,
                      a.attributes.ptxVersion,
                      a.block_size,
                      a.dynamic_smem_size,
                      a.func_ptr} ==
           std::tuple{b.attributes.binaryVersion,
                      b.attributes.cacheModeCA,
                      b.attributes.constSizeBytes,
                      b.attributes.localSizeBytes,
                      b.attributes.maxDynamicSharedSizeBytes,
                      b.attributes.maxThreadsPerBlock,
                      b.attributes.numRegs,
                      b.attributes.preferredShmemCarveout,
                      b.attributes.ptxVersion,
                      b.block_size,
                      b.dynamic_smem_size,
                      b.func_ptr};
  }

  friend bool operator!=(const RocmDeviceOccupancyParams& a,
                         const RocmDeviceOccupancyParams& b) noexcept {
    return !(a == b);
  }

  template <typename H>
  friend H AbslHashValue(H hash_state,
                         const RocmDeviceOccupancyParams& params) {
    return H::combine(
        std::move(hash_state), params.attributes.maxThreadsPerBlock,
        params.attributes.numRegs, params.attributes.sharedSizeBytes,
        params.attributes.maxDynamicSharedSizeBytes, params.block_size,
        params.dynamic_smem_size, params.func_ptr);
  }
};

// FIXME: rocprofiler-sdk does not have this one yet
struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

class RocmTraceCollector {
 public:
  explicit RocmTraceCollector(const RocmTraceCollectorOptions& options)
      : options_(options) {}
  virtual ~RocmTraceCollector() {}

  virtual void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) = 0;
  virtual void OnEventsDropped(const std::string& reason,
                               uint32_t num_events) = 0;
  virtual void Flush() = 0;
  virtual void Export(tsl::profiler::XSpace* space) = 0;

 protected:
  RocmTraceCollectorOptions options_;

 public:
  // Disable copy and move.
  RocmTraceCollector(const RocmTraceCollector&) = delete;
  RocmTraceCollector& operator=(const RocmTraceCollector&) = delete;
};

class PerDeviceCollector {
 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns,
              tsl::profiler::XPlaneBuilder* device_plane,
              tsl::profiler::XPlaneBuilder* host_plane);

  PerDeviceCollector() = default;

  void AddEvent(RocmTracerEvent&& event);
  void GetDeviceCapabilities(int32_t device_ordinal,
                             tsl::profiler::XPlaneBuilder* device_plane);

 private:
  OccupancyStats GetOccupancy(const RocmDeviceOccupancyParams& params) const;
  void CreateXEvent(const RocmTracerEvent& event,
                    tsl::profiler::XPlaneBuilder* plane, uint64_t start_gpu_ns,
                    uint64_t end_gpu_ns, tsl::profiler::XLineBuilder* line);
  void SortByStartTime();
  bool IsHostEvent(const RocmTracerEvent& event, int64_t* line_id);

 private:
  absl::Mutex events_mutex_;
  std::vector<RocmTracerEvent> events_ ABSL_GUARDED_BY(events_mutex_);
  absl::flat_hash_map<RocmDeviceOccupancyParams, OccupancyStats>
      occupancy_cache_;
  hipDeviceProp_t device_properties_;
};  // PerDeviceCollector

class RocmTraceCollectorImpl : public RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        num_gpus_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override;
  void Flush() override;
  void Export(tsl::profiler::XSpace* space) override;

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    VLOG(2) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
            << ",) : " << reason << ".";
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;

  absl::Mutex event_maps_mutex_;
  absl::flat_hash_map<uint32_t, RocmTracerEvent> api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
   trigger multiple HIP ops domain activities. We keep them in a vector and
   merge them with api activities at flush time.
 */
  absl::flat_hash_map<uint32_t, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ ABSL_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint32_t, RocmTracerEvent> auxiliary_api_events_map_
      ABSL_GUARDED_BY(event_maps_mutex_);

  std::vector<RocmTracerEvent> ApiActivityInfoExchange()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(event_maps_mutex_);

  absl::node_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;
};  // RocmTraceCollectorImpl

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, uint64_t start_walltime_ns,
    uint64_t start_gputime_ns);

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_ROCM_COLLECTOR_H_
