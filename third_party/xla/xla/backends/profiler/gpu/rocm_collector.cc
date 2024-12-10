
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

#include "xla/backends/profiler/gpu/rocm_collector.h"

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "xla/stream_executor/rocm/roctracer_wrapper.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/parse_annotation.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/abi.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/status.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"

namespace xla {
namespace profiler {

namespace se = ::stream_executor;
using tensorflow::ProfileOptions;
using tsl::mutex;
using tsl::mutex_lock;
// using tsl::OkStatus;
using tsl::Status;
using tsl::profiler::Annotation;
using tsl::profiler::AnnotationStack;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kDeviceVendorAMD;
using tsl::profiler::kThreadIdOverhead;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::ProfilerInterface;
// using tsl::profiler::RegisterProfilerFactory;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

void AnnotationMap::Add(uint32_t correlation_id,
                        const std::string& annotation) {
  if (annotation.empty()) return;
  VLOG(3) << "Add annotation: " << " correlation_id=" << correlation_id
          << ", annotation: " << annotation;
  absl::MutexLock lock(&map_.mutex);
  if (map_.annotations.size() < max_size_) {
    absl::string_view annotation_str =
        *map_.annotations.insert(annotation).first;
    map_.correlation_map.emplace(correlation_id, annotation_str);
  }
}

absl::string_view AnnotationMap::LookUp(uint32_t correlation_id) {
  absl::MutexLock lock(&map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second : absl::string_view();
}

//==========
namespace {
// Set the all XLines of specified XPlane to starting walltime.
// Events time in both host and device planes are CUTPI timestamps.
// We set initial RocmTracer timestamp as start time for all lines to reflect
// this fact. Eventually we change line start time to corresponding
// start_walltime_ns to normalize with CPU wall time.
static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](tsl::profiler::XLineBuilder line) {
    line.SetTimestampNs(start_walltime_ns);
  });
}

std::string GetDeviceXLineName(
    int64_t stream_id, absl::flat_hash_set<RocmTracerEventType>& event_types) {
  std::string line_name = absl::StrCat("Stream #", stream_id);
  event_types.erase(RocmTracerEventType::Unsupported);
  if (event_types.empty()) return line_name;
  std::vector<const char*> type_names;
  for (const auto event_type : event_types) {
    type_names.emplace_back(GetRocmTracerEventTypeName(event_type));
  }
  return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
}

}  // namespace

static void DumpRocmTracerEvent(const RocmTracerEvent& event,
                                uint64_t start_walltime_ns,
                                uint64_t start_gputime_ns,
                                const std::string& message) {
  std::ostringstream oss;
  oss << "correlation_id=" << event.correlation_id;
  oss << ",type=" << GetRocmTracerEventTypeName(event.type);
  oss << ",source=" << GetRocmTracerEventSourceName(event.source);
  oss << ",domain=" << GetRocmTracerEventDomainName(event.domain);
  oss << ",name=" << event.name;
  oss << ",annotation=" << event.annotation;
  oss << ",start_time_us="
      << (start_walltime_ns + (start_gputime_ns - event.start_time_ns)) / 1000;
  oss << ",duration=" << (event.end_time_ns - event.start_time_ns) / 1000;
  oss << ",device_id=" << event.device_id;
  oss << ",thread_id=" << event.thread_id;
  oss << ",stream_id=" << event.stream_id;

  switch (event.type) {
    case RocmTracerEventType::Kernel:
      break;
    case RocmTracerEventType::MemcpyD2H:
    case RocmTracerEventType::MemcpyH2D:
    case RocmTracerEventType::MemcpyD2D:
    case RocmTracerEventType::MemcpyP2P:
      oss << ",num_bytes=" << event.memcpy_info.num_bytes;
      oss << ",destination=" << event.memcpy_info.destination;
      oss << ",async=" << event.memcpy_info.async;
      break;
    case RocmTracerEventType::MemoryAlloc:
      oss << ",num_bytes=" << event.memalloc_info.num_bytes;
      break;
    case RocmTracerEventType::MemcpyOther:
    case RocmTracerEventType::MemoryFree:
    case RocmTracerEventType::Memset:
    case RocmTracerEventType::Synchronization:
    case RocmTracerEventType::Generic:
      break;
    default:
      DCHECK(false);
      break;
  }
  oss << message;
  VLOG(3) << oss.str();
}

static uint64_t get_timestamp() {
  uint64_t ts;
  if (se::wrap::roctracer_get_timestamp(&ts) != ROCTRACER_STATUS_SUCCESS) {
    const char* errstr = se::wrap::roctracer_error_string();
    LOG(ERROR) << "function roctracer_get_timestamp failed with error "
               << errstr;
    // Return 0 on error.
    return 0;
  }
  return ts;
}

struct RocmDeviceOccupancyParams {
  hipFuncAttributes attributes = {};
  int block_size = 0;
  size_t dynamic_smem_size = 0;
  void* func_ptr;

  friend bool operator==(const RocmDeviceOccupancyParams& lhs,
                         const RocmDeviceOccupancyParams& rhs) {
    return 0 == memcmp(&lhs, &rhs, sizeof(lhs));
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

struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

struct CorrelationInfo {
  CorrelationInfo(uint32_t t, uint32_t e) : thread_id(t), enqueue_time_ns(e) {}
  uint32_t thread_id;
  uint64_t enqueue_time_ns;
};

class PerDeviceCollector {
 private:
  OccupancyStats GetOccupancy(const RocmDeviceOccupancyParams& params) const {
    // TODO(rocm-profiler): hipOccupancyMaxActiveBlocksPerMultiprocessor only
    // return hipSuccess for HIP_API_ID_hipLaunchKernel

    OccupancyStats stats;
    int number_of_active_blocks;
    hipError_t err = hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &number_of_active_blocks, params.func_ptr, params.block_size,
        params.dynamic_smem_size);

    if (err != hipError_t::hipSuccess) {
      return {};
    }

    stats.occupancy_pct = number_of_active_blocks * params.block_size * 100;
    stats.occupancy_pct /= device_properties_.maxThreadsPerMultiProcessor;

    err = hipOccupancyMaxPotentialBlockSize(
        &stats.min_grid_size, &stats.suggested_block_size,
        static_cast<const void*>(params.func_ptr), params.dynamic_smem_size, 0);

    if (err != hipError_t::hipSuccess) {
      return {};
    }

    return stats;
  }

  void CreateXEvent(const RocmTracerEvent& event, XPlaneBuilder* plane,
                    uint64_t start_gpu_ns, uint64_t end_gpu_ns,
                    XLineBuilder* line) {
    if (event.start_time_ns < start_gpu_ns || event.end_time_ns > end_gpu_ns ||
        event.start_time_ns > event.end_time_ns) {
      VLOG(2) << "events have abnormal timestamps:" << event.name
              << " start time(ns): " << event.start_time_ns
              << " end time(ns): " << event.end_time_ns
              << " start gpu(ns):" << start_gpu_ns
              << " end gpu(ns):" << end_gpu_ns
              << " corr. id:" << event.correlation_id;
      return;
    }
    std::string kernel_name = tsl::port::MaybeAbiDemangle(event.name.c_str());
    if (kernel_name.empty()) {
      kernel_name = GetRocmTracerEventTypeName(event.type);
    }
    XEventMetadata* event_metadata =
        plane->GetOrCreateEventMetadata(std::move(kernel_name));
    XEventBuilder xevent = line->AddEvent(*event_metadata);
    VLOG(7) << "Adding event to line=" << line->Id();
    xevent.SetTimestampNs(event.start_time_ns);
    xevent.SetEndTimestampNs(event.end_time_ns);
    if (event.source == RocmTracerEventSource::ApiCallback) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)),
          event.device_id);
    }
    if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kCorrelationId)),
                          event.correlation_id);
    }
    if (!event.roctx_range.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kNVTXRange)),
          *plane->GetOrCreateStatMetadata(event.roctx_range));
    }

    if (event.type == RocmTracerEventType::Kernel &&
        event.source == RocmTracerEventSource::Activity) {
      RocmDeviceOccupancyParams params{};
      params.attributes.maxThreadsPerBlock = INT_MAX;
      params.attributes.numRegs =
          static_cast<int>(event.kernel_info.registers_per_thread);
      params.attributes.sharedSizeBytes =
          event.kernel_info.static_shared_memory_usage;
      // params.attributes.partitionedGCConfig = PARTITIONED_GC_OFF;
      // params.attributes.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
      params.attributes.maxDynamicSharedSizeBytes = 0;
      params.block_size = static_cast<int>(event.kernel_info.block_x *
                                           event.kernel_info.block_y *
                                           event.kernel_info.block_z);

      params.dynamic_smem_size = event.kernel_info.dynamic_shared_memory_usage;
      params.func_ptr = event.kernel_info.func_ptr;

      OccupancyStats& occ_stats = occupancy_cache_[params];
      if (occ_stats.occupancy_pct == 0.0) {
        occ_stats = GetOccupancy(params);
      }
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kTheoreticalOccupancyPct)),
                          occ_stats.occupancy_pct);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kOccupancyMinGridSize)),
                          static_cast<tsl::int32>(occ_stats.min_grid_size));
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kOccupancySuggestedBlockSize)),
          static_cast<tsl::int32>(occ_stats.suggested_block_size));
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kKernelDetails)),
                          *plane->GetOrCreateStatMetadata(ToXStat(
                              event.kernel_info, occ_stats.occupancy_pct)));
    } else if (event.type == RocmTracerEventType::MemcpyH2D ||
               event.type == RocmTracerEventType::MemcpyD2H ||
               event.type == RocmTracerEventType::MemcpyD2D ||
               event.type == RocmTracerEventType::MemcpyP2P ||
               event.type == RocmTracerEventType::MemcpyOther) {
      VLOG(7) << "Add Memcpy stat";
      const auto& memcpy_info = event.memcpy_info;
      std::string memcpy_details = absl::StrCat(
          // TODO(rocm-profiler): we need to discover the memory kind similar
          // to CUDA
          "kind:", "Unknown", " size:", memcpy_info.num_bytes,
          " dest:", memcpy_info.destination, " async:", memcpy_info.async);
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kMemcpyDetails)),
          *plane->GetOrCreateStatMetadata(std::move(memcpy_details)));
    } else if (event.type == RocmTracerEventType::MemoryAlloc) {
      VLOG(7) << "Add MemAlloc stat";
      std::string value =
          // TODO(rocm-profiler): we need to discover the memory kind similar
          // to CUDA
          absl::StrCat("kind:", "Unknown",
                       " num_bytes:", event.memalloc_info.num_bytes);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemallocDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == RocmTracerEventType::MemoryFree) {
      VLOG(7) << "Add MemFree stat";
      std::string value =
          // TODO(rocm-profiler): we need to discover the memory kind similar
          // to CUDA
          absl::StrCat("kind:", "Unknown",
                       " num_bytes:", event.memalloc_info.num_bytes);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemFreeDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == RocmTracerEventType::Memset) {
      VLOG(7) << "Add Memset stat";
      auto value =
          // TODO(rocm-profiler): we need to discover the memory kind similar
          // to CUDA
          absl::StrCat("kind:", "Unknown",
                       " num_bytes:", event.memset_info.num_bytes,
                       " async:", event.memset_info.async);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemsetDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    }
    // TODO(rocm-profiler): we need to support the following event type
    /* else if (event.type == CuptiTracerEventType::MemoryResidency) {
      VLOG(7) << "Add MemoryResidency stat";
      std::string value = absl::StrCat(
          "kind:", GetMemoryKindName(event.memory_residency_info.kind),
          " num_bytes:", event.memory_residency_info.num_bytes,
          " addr:", event.memory_residency_info.address);
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kMemoryResidencyDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } */

    std::vector<Annotation> annotation_stack =
        ParseAnnotationStack(event.annotation);
    if (!annotation_stack.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
          *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
    }
    // If multiple metadata have the same key name, show the values from the
    // top of the stack (innermost annotation). Concatenate the values from
    // "hlo_op".
    absl::flat_hash_set<absl::string_view> key_set;

    for (auto annotation = annotation_stack.rbegin();
         annotation != annotation_stack.rend(); ++annotation) {
      for (const Annotation::Metadata& metadata : annotation->metadata) {
        if (key_set.insert(metadata.key).second) {
          xevent.ParseAndAddStatValue(
              *plane->GetOrCreateStatMetadata(metadata.key), metadata.value);
        }
      }
    }
  }

  void SortByStartTime() {
    mutex_lock lock(events_mutex);
    std::sort(events.begin(), events.end(),
              [](const RocmTracerEvent& event1, const RocmTracerEvent& event2) {
                return event1.start_time_ns < event2.start_time_ns;
              });
  }

  bool IsHostEvent(const RocmTracerEvent& event, tsl::int64* line_id) {
    // DriverCallback(i.e. kernel launching) events are host events.
    if (event.source == RocmTracerEventSource::ApiCallback) {
      *line_id = event.thread_id;
      return true;
    } else {  // activities
      *line_id = event.stream_id;
      return false;
    }

    // TODO(rocm-profiler): do we have such a report in rocm?
    // Non-overhead activity events are device events.
    /* if (event.type != CuptiTracerEventType::Overhead) {
      *line_id = event.stream_id;
      return false;
    } */
    // Overhead events can be associated with a thread or a stream, etc.
    // If a valid thread id is specified, we consider it as a host event.
    //

    if (event.stream_id != RocmTracerEvent::kInvalidStreamId) {
      *line_id = event.stream_id;
      return false;
    } else if (event.thread_id != RocmTracerEvent::kInvalidThreadId &&
               event.thread_id != 0) {
      *line_id = event.thread_id;
      return true;
    } else {
      *line_id = tsl::profiler::kThreadIdOverhead;
      return false;
    }
  }

 public:
  void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
              uint64_t end_gputime_ns, XPlaneBuilder* device_plane,
              XPlaneBuilder* host_plane) {
    int host_ev_cnt = 0, dev_ev_cnt = 0;
    mutex_lock l(events_mutex);
    // Tracking event types per line.
    absl::flat_hash_map<tsl::int64, absl::flat_hash_set<RocmTracerEventType>>
        events_types_per_line;
    for (const RocmTracerEvent& event : events) {
      int64_t line_id = RocmTracerEvent::kInvalidThreadId;
      bool is_host_event = IsHostEvent(event, &line_id);

      if (is_host_event) {
        host_ev_cnt++;
      } else {
        dev_ev_cnt++;
      }

      if (line_id == RocmTracerEvent::kInvalidThreadId ||
          line_id == RocmTracerEvent::kInvalidStreamId) {
        VLOG(3) << "Ignoring event, type=" << static_cast<int>(event.type);
        continue;
      }
      auto* plane = is_host_event ? host_plane : device_plane;
      VLOG(9) << "Event" << " type=" << static_cast<int>(event.type)
              << " line_id=" << line_id
              << (is_host_event ? " host plane=" : " device plane=")
              << plane->Name();
      XLineBuilder line = plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_gputime_ns);
      CreateXEvent(event, plane, start_gputime_ns, end_gputime_ns, &line);
      events_types_per_line[line_id].emplace(event.type);
    }
    device_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(
          GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
    });
    host_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(absl::StrCat("Host Threads/", line.Id()));
    });
    events.clear();
  }

  PerDeviceCollector() = default;

  void AddEvent(const RocmTracerEvent& event) {
    mutex_lock l(events_mutex);
    if (event.source == RocmTracerEventSource::ApiCallback) {
      // Cupti api callback events were used to populate launch times etc.
      if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
        correlation_info_.insert(
            {event.correlation_id,
             CorrelationInfo(event.thread_id, event.start_time_ns)});
      }
      events.emplace_back(std::move(event));
    } else {
      // Cupti activity events measure device times etc.
      events.emplace_back(std::move(event));
    }
  }

  void GetDeviceCapabilities(int32_t device_ordinal,
                             XPlaneBuilder* device_plane) {
    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kDevVendor)),
                               kDeviceVendorAMD);

    if (hipGetDeviceProperties(&device_properties_, device_ordinal) !=
        hipSuccess)
      return;

    auto clock_rate_in_khz =
        device_properties_.clockRate;  // this is also in Khz
    if (clock_rate_in_khz) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapClockRateKHz)),
          clock_rate_in_khz);
    }

    auto core_count = device_properties_.multiProcessorCount;
    if (core_count) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapCoreCount)),
          core_count);
    }

    auto mem_clock_khz = device_properties_.memoryClockRate;
    auto mem_bus_width_bits = device_properties_.memoryBusWidth;

    if (mem_clock_khz && mem_bus_width_bits) {
      // Times 2 because HBM is DDR memory; it gets two data bits per each
      // data lane.
      auto memory_bandwidth =
          uint64_t{2} * (mem_clock_khz) * 1000 * (mem_bus_width_bits) / 8;
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
          memory_bandwidth);
    }

    size_t total_memory = device_properties_.totalGlobalMem;
    if (total_memory) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemorySize)),
          static_cast<uint64_t>(total_memory));
    }

    auto compute_capability_major = device_properties_.major;
    if (compute_capability_major) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
          compute_capability_major);
    }
    auto compute_capability_minor = device_properties_.minor;
    if (compute_capability_minor) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
          compute_capability_minor);
    }
  }

 private:
  mutex events_mutex;
  std::vector<RocmTracerEvent> events TF_GUARDED_BY(events_mutex);
  absl::flat_hash_map<uint32_t, CorrelationInfo> correlation_info_
      TF_GUARDED_BY(events_mutex);
  absl::flat_hash_map<RocmDeviceOccupancyParams, OccupancyStats>
      occupancy_cache_;
  hipDeviceProp_t device_properties_;
};

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
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
  void Export(XSpace* space) override;

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    LOG(INFO) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
              << ",) : " << reason << ".";
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;
  int num_gpus_;

  mutex event_maps_mutex_;
  absl::flat_hash_map<uint32_t, RocmTracerEvent> api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
   trigger multiple HIP ops domain activities. We keep them in a vector and
   merge them with api activities at flush time.
 */
  absl::flat_hash_map<uint32_t, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ TF_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint32_t, RocmTracerEvent> auxiliary_api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  const std::vector<RocmTracerEvent> ApiActivityInfoExchange()
      TF_EXCLUSIVE_LOCKS_REQUIRED(event_maps_mutex_);

  absl::node_hash_map<uint32_t, PerDeviceCollector> per_device_collector_;
};
//==========

void RocmTraceCollectorImpl::AddEvent(RocmTracerEvent&& event,
                                      bool is_auxiliary) {
  mutex_lock lock(event_maps_mutex_);

  if (event.source == RocmTracerEventSource::ApiCallback && !is_auxiliary) {
    if (num_callback_events_ > options_.max_callback_api_events) {
      OnEventsDropped("max callback event capacity reached",
                      event.correlation_id);
      DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
      return;
    }
    num_callback_events_++;
  } else if (event.source == RocmTracerEventSource::Activity &&
             event.domain == RocmTracerEventDomain::HIP_API) {
    // we do not count HIP_OPS activities.
    if (num_activity_events_ > options_.max_activity_api_events) {
      OnEventsDropped("max activity event capacity reached",
                      event.correlation_id);
      DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
      return;
    }
    num_activity_events_++;
  }

  bool emplace_result = false;
  if (event.source == RocmTracerEventSource::ApiCallback) {
    auto& target_api_event_map =
        (is_auxiliary) ? auxiliary_api_events_map_ : api_events_map_;
    std::tie(std::ignore, emplace_result) =
        target_api_event_map.emplace(event.correlation_id, std::move(event));
  } else if (event.source == RocmTracerEventSource::Activity) {
    auto result = activity_ops_events_map_.emplace(
        event.correlation_id, std::vector<RocmTracerEvent>{});
    result.first->second.push_back(std::move(event));
    emplace_result = true;  // we always accept Hip-Ops events
  }
  if (!emplace_result) {
    OnEventsDropped("event with duplicate correlation_id was received.",
                    event.correlation_id);
    DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
  }
}

void RocmTraceCollectorImpl::Flush() {
  mutex_lock lock(event_maps_mutex_);
  auto& aggregated_events_ = ApiActivityInfoExchange();

  VLOG(3) << "RocmTraceCollector collected " << num_callback_events_
          << " callback events, " << num_activity_events_
          << " activity events, and aggregated them into "
          << aggregated_events_.size() << " events.";

  // device ids for GPUs filled in by roctracer are not zero indexed.
  // They are offset by number of CPUs on the machine
  tsl::uint32 min_device_id = INT32_MAX;
  ;
  for (auto& event : aggregated_events_) {
    if (event.device_id < min_device_id) {
      min_device_id = event.device_id;
    }
  }

  for (auto event : aggregated_events_) {
    event.device_id = event.device_id - min_device_id;
    if (event.device_id < num_gpus_) {
      per_device_collector_[event.device_id].AddEvent(event);
    } else {
      OnEventsDropped("Invalid device id for an event.", event.correlation_id);
      DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
    }
  }

  activity_ops_events_map_.clear();
  api_events_map_.clear();
  auxiliary_api_events_map_.clear();
}

void RocmTraceCollectorImpl::Export(XSpace* space) {
  uint64_t end_gputime_ns = get_timestamp();
  XPlaneBuilder host_plane(FindOrAddMutablePlaneWithName(
      space, tsl::profiler::kRoctracerApiPlaneName));

  for (int device_ordinal = 0; device_ordinal < num_gpus_; ++device_ordinal) {
    std::string name = GpuPlaneName(device_ordinal);
    XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
    device_plane.SetId(device_ordinal);
    // Calculate device capabilities before flushing, so that device
    // properties are available to the occupancy calculator in export().
    per_device_collector_[device_ordinal].GetDeviceCapabilities(device_ordinal,
                                                                &device_plane);
    per_device_collector_[device_ordinal].Export(
        start_walltime_ns_, start_gputime_ns_, end_gputime_ns, &device_plane,
        &host_plane);
    NormalizeTimeStamps(&device_plane, start_walltime_ns_);
  }
  NormalizeTimeStamps(&host_plane, start_walltime_ns_);
}

const std::vector<RocmTracerEvent>
RocmTraceCollectorImpl::ApiActivityInfoExchange() {
  /* Different from CUDA, roctracer activity records are not enough to fill a
    TF event. For most of the activities, we need to enable the corresponding
    API callsbacks (we call them auxiliary API callbacks) to capture the
    necessary fields from them using the correlation id. The purpose of this
    function is to let APIs and activities exchange information to reach a
    state very similar to TF CUDA and getting ready to dump the event.
  */

  std::vector<RocmTracerEvent> aggregated_events;

  // Copy info from activity events to API callback events
  for (auto& api_iter : api_events_map_) {
    RocmTracerEvent& api_event = api_iter.second;
    auto activity_event =
        activity_ops_events_map_.find(api_event.correlation_id);

    if (activity_event == activity_ops_events_map_.end()) {
      OnEventsDropped(
          "An event from HIP API discarded."
          "Could not find the counterpart activity.",
          api_event.correlation_id);
      DumpRocmTracerEvent(api_event, 0, 0, ". Dropped!");
    } else {
      api_event.device_id = activity_event->second.front().device_id;
      api_event.stream_id = activity_event->second.front().stream_id;
      switch (api_event.type) {
        case RocmTracerEventType::Kernel:
        case RocmTracerEventType::Memset:
        case RocmTracerEventType::MemoryAlloc:
        case RocmTracerEventType::MemoryFree:
        case RocmTracerEventType::Synchronization: {
          aggregated_events.push_back(api_event);
          break;
        }
        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyOther: {
          api_event.memcpy_info.destination =
              activity_event->second.front().device_id;
          aggregated_events.push_back(api_event);
          break;
        }
        default:
          OnEventsDropped("Missing API-Activity information exchange. Dropped!",
                          api_event.correlation_id);
          DumpRocmTracerEvent(api_event, 0, 0, ". Dropped!");
          LOG(WARNING) << "A ROCm API event type with unimplemented activity "
                          "merge dropped! "
                          "Type="
                       << GetRocmTracerEventTypeName(api_event.type);
      }
    }
  }

  // Make sure for all activity events we have API callback events
  for (auto& activity_iter : activity_ops_events_map_) {
    RocmTracerEvent& activity_event = activity_iter.second.front();
    auto api_event = api_events_map_.find(activity_event.correlation_id);

    if (api_event == api_events_map_.end()) {
      api_event = auxiliary_api_events_map_.find(activity_event.correlation_id);
    }

    if (api_event == auxiliary_api_events_map_.end()) {
      OnEventsDropped(
          "An event from activity was discarded."
          "Could not find the counterpart HIP API.",
          activity_event.correlation_id);
      DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
    } else {
      switch (activity_event.type) {
        // KERNEL ACTIVITY
        case RocmTracerEventType::Kernel: {
          activity_event.name = api_event->second.name;
          activity_event.kernel_info = api_event->second.kernel_info;
          aggregated_events.push_back(activity_event);
          break;
        }
        // MEMCPY ACTIVITY
        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyOther: {
          activity_event.memcpy_info = api_event->second.memcpy_info;
          aggregated_events.push_back(activity_event);
          break;
        }
        // MEMSET ACTIVITY
        case RocmTracerEventType::Memset: {
          activity_event.memset_info = api_event->second.memset_info;
          aggregated_events.push_back(activity_event);
          break;
        }
        // MALLOC ACTIVITY, FREE ACTIVITY
        case RocmTracerEventType::MemoryAlloc:
        case RocmTracerEventType::MemoryFree: {
          activity_event.device_id = api_event->second.device_id;
          aggregated_events.push_back(activity_event);
          break;
        }
        // SYNCHRONIZATION ACTIVITY
        case RocmTracerEventType::Synchronization: {
          activity_event.device_id = api_event->second.device_id;
          aggregated_events.push_back(activity_event);
          break;
        }
        default:
          OnEventsDropped("Missing API-Activity information exchange. Dropped!",
                          activity_event.correlation_id);
          DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          LOG(WARNING) << "A ROCm activity event with unimplemented API "
                          "callback merge dropped! "
                          "Type="
                       << GetRocmTracerEventTypeName(activity_event.type);
          break;
      }
    }
  }

  return aggregated_events;
}

std::unique_ptr<RocmTraceCollector> CreateRocmCollector(
    const RocmTraceCollectorOptions& options, const uint64_t start_walltime_ns,
    const uint64_t start_gputime_ns) {
  return std::make_unique<RocmTraceCollectorImpl>(options, start_walltime_ns,
                                                  start_gputime_ns);
}

}  // namespace profiler
}  // namespace xla
