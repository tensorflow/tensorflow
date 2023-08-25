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

#include "tensorflow/compiler/xla/backends/profiler/gpu/cupti_collector.h"

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_occupancy.h"
#include "tensorflow/tsl/platform/abi.h"
#include "tensorflow/tsl/platform/host_info.h"
#include "tensorflow/tsl/platform/mutex.h"
#include "tensorflow/tsl/profiler/utils/parse_annotation.h"
#include "tensorflow/tsl/profiler/utils/trace_utils.h"
#include "tensorflow/tsl/profiler/utils/xplane_builder.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"
#include "tensorflow/tsl/profiler/utils/xplane_utils.h"

namespace xla {
namespace profiler {

namespace {

using tensorflow::profiler::XEventMetadata;
using tensorflow::profiler::XSpace;
using tsl::mutex;
using tsl::mutex_lock;
using tsl::profiler::Annotation;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kCuptiDriverApiPlaneName;
using tsl::profiler::kDeviceVendorNvidia;
using tsl::profiler::kThreadIdOverhead;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;

bool IsHostEvent(const CuptiTracerEvent& event, int64_t* line_id) {
  // DriverCallback(i.e. kernel launching) events are host events.
  if (event.source == CuptiTracerEventSource::DriverCallback) {
    *line_id = event.thread_id;
    return true;
  }
  // Non-overhead activity events are device events.
  if (event.type != CuptiTracerEventType::Overhead) {
    *line_id = event.stream_id;
    return false;
  }
  // Overhead events can be associated with a thread or a stream, etc.
  // If a valid thread id is specified, we consider it as a host event.
  //
  if (event.stream_id != CuptiTracerEvent::kInvalidStreamId) {
    *line_id = event.stream_id;
    return false;
  } else if (event.thread_id != CuptiTracerEvent::kInvalidThreadId &&
             event.thread_id != 0) {
    *line_id = event.thread_id;
    return true;
  } else {
    *line_id = kThreadIdOverhead;
    return false;
  }
}

struct DeviceOccupancyParams {
  cudaOccFuncAttributes attributes = {};
  int block_size = 0;
  size_t dynamic_smem_size = 0;

  friend bool operator==(const DeviceOccupancyParams& lhs,
                         const DeviceOccupancyParams& rhs) {
    return 0 == memcmp(&lhs, &rhs, sizeof(lhs));
  }

  template <typename H>
  friend H AbslHashValue(H hash_state, const DeviceOccupancyParams& params) {
    return H::combine(
        std::move(hash_state), params.attributes.maxThreadsPerBlock,
        params.attributes.numRegs, params.attributes.sharedSizeBytes,
        static_cast<uint32_t>(params.attributes.partitionedGCConfig),
        static_cast<uint32_t>(params.attributes.shmemLimitConfig),
        params.attributes.maxDynamicSharedSizeBytes, params.block_size,
        params.dynamic_smem_size);
  }
};

struct OccupancyStats {
  double occupancy_pct = 0.0;
  int min_grid_size = 0;
  int suggested_block_size = 0;
};

class PerDeviceCollector {
 private:
  OccupancyStats GetOccupancy(const DeviceOccupancyParams& params) const {
    OccupancyStats stats;
    if (device_properties_.computeMajor == 0) {
      return {};
    }

    const cudaOccDeviceState state = {};
    cudaOccResult occ_result;
    cudaOccError status = cudaOccMaxActiveBlocksPerMultiprocessor(
        &occ_result, &device_properties_, &params.attributes, &state,
        params.block_size, params.dynamic_smem_size);
    if (status != CUDA_OCC_SUCCESS) {
      return {};
    }

    stats.occupancy_pct =
        occ_result.activeBlocksPerMultiprocessor * params.block_size * 100;
    stats.occupancy_pct /= device_properties_.maxThreadsPerMultiprocessor;

    status = cudaOccMaxPotentialOccupancyBlockSize(
        &stats.min_grid_size, &stats.suggested_block_size, &device_properties_,
        &params.attributes, &state, nullptr, params.dynamic_smem_size);
    if (status != CUDA_OCC_SUCCESS) {
      return {};
    }

    return stats;
  }

  void CreateXEvent(const CuptiTracerEvent& event, XPlaneBuilder* plane,
                    tsl::uint64 start_gpu_ns, tsl::uint64 end_gpu_ns,
                    XLineBuilder* line) {
    if (event.start_time_ns < start_gpu_ns || event.end_time_ns > end_gpu_ns ||
        event.start_time_ns > event.end_time_ns) {
      VLOG(2) << "events have abnormal timestamps:" << event.name
              << " start time(ns): " << event.start_time_ns
              << " end time(ns): " << event.end_time_ns;
      return;
    }
    std::string kernel_name = tsl::port::MaybeAbiDemangle(event.name.c_str());
    if (kernel_name.empty()) {
      kernel_name = GetTraceEventTypeName(event.type);
    }
    XEventMetadata* event_metadata =
        plane->GetOrCreateEventMetadata(std::move(kernel_name));
    XEventBuilder xevent = line->AddEvent(*event_metadata);
    VLOG(7) << "Adding event to line=" << line->Id();
    xevent.SetTimestampNs(event.start_time_ns);
    xevent.SetEndTimestampNs(event.end_time_ns);
    if (event.source == CuptiTracerEventSource::DriverCallback) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kDeviceId)),
          event.device_id);
    }
    if (event.correlation_id != CuptiTracerEvent::kInvalidCorrelationId) {
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kCorrelationId)),
                          event.correlation_id);
    }
    if (!event.nvtx_range.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kNVTXRange)),
          *plane->GetOrCreateStatMetadata(event.nvtx_range));
    }
    if (event.context_id != CuptiTracerEvent::kInvalidContextId) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kContextId)),
          absl::StrCat("$$", static_cast<tsl::uint64>(event.context_id)));
    }

    if (event.type == CuptiTracerEventType::Kernel &&
        event.source == CuptiTracerEventSource::Activity) {
      DeviceOccupancyParams params{};
      params.attributes.maxThreadsPerBlock = INT_MAX;
      params.attributes.numRegs =
          static_cast<int>(event.kernel_info.registers_per_thread);
      params.attributes.sharedSizeBytes =
          event.kernel_info.static_shared_memory_usage;
      params.attributes.partitionedGCConfig = PARTITIONED_GC_OFF;
      params.attributes.shmemLimitConfig = FUNC_SHMEM_LIMIT_DEFAULT;
      params.attributes.maxDynamicSharedSizeBytes = 0;
      params.block_size = static_cast<int>(event.kernel_info.block_x *
                                           event.kernel_info.block_y *
                                           event.kernel_info.block_z);

      params.dynamic_smem_size = event.kernel_info.dynamic_shared_memory_usage;

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
    } else if (event.type == CuptiTracerEventType::MemcpyH2D ||
               event.type == CuptiTracerEventType::MemcpyD2H ||
               event.type == CuptiTracerEventType::MemcpyD2D ||
               event.type == CuptiTracerEventType::MemcpyP2P ||
               event.type == CuptiTracerEventType::MemcpyOther) {
      const auto& memcpy_info = event.memcpy_info;
      std::string value = absl::StrCat(
          "kind_src:", GetMemoryKindName(event.memcpy_info.src_mem_kind),
          " kind_dst:", GetMemoryKindName(event.memcpy_info.dst_mem_kind),
          " size:", memcpy_info.num_bytes, " dest:", memcpy_info.destination,
          " async:", memcpy_info.async);
      VLOG(7) << "Add Memcpy stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemcpyDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryAlloc) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memalloc_info.mem_kind),
                       " num_bytes:", event.memalloc_info.num_bytes);
      VLOG(7) << "Add MemAlloc stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemallocDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryFree) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memfree_info.mem_kind),
                       " num_bytes:", event.memfree_info.num_bytes);
      VLOG(7) << "Add MemFree stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemFreeDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::Memset) {
      std::string value =
          absl::StrCat("kind:", GetMemoryKindName(event.memset_info.mem_kind),
                       " num_bytes:", event.memset_info.num_bytes,
                       " async:", event.memset_info.async);
      VLOG(7) << "Add Memset stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                              GetStatTypeStr(StatType::kMemsetDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    } else if (event.type == CuptiTracerEventType::MemoryResidency) {
      std::string value = absl::StrCat(
          "kind:", GetMemoryKindName(event.memory_residency_info.mem_kind),
          " num_bytes:", event.memory_residency_info.num_bytes, " addr:0x",
          absl::Hex(event.memory_residency_info.address, absl::kZeroPad16));
      VLOG(7) << "Add MemoryResidency stat. " << value;
      xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                              StatType::kMemoryResidencyDetails)),
                          *plane->GetOrCreateStatMetadata(std::move(value)));
    }

    std::vector<Annotation> annotation_stack =
        ParseAnnotationStack(event.annotation);
    if (!annotation_stack.empty()) {
      xevent.AddStatValue(
          *plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kTfOp)),
          *plane->GetOrCreateStatMetadata(annotation_stack.begin()->name));
    }
    // If multiple metadata have the same key name, show the values from the top
    // of the stack (innermost annotation). Concatenate the values from
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

  std::optional<int> GetDeviceAttribute(CUdevice device,
                                        CUdevice_attribute attrib) {
    int ret_val;
    CUresult err = cuDeviceGetAttribute(&ret_val, attrib, device);
    if (err != CUDA_SUCCESS) return std::nullopt;
    return ret_val;
  }

  std::string GetDeviceXLineName(
      int64_t stream_id,
      absl::flat_hash_set<CuptiTracerEventType>& event_types) {
    std::string line_name = absl::StrCat("Stream #", stream_id);
    event_types.erase(CuptiTracerEventType::Unsupported);
    if (event_types.empty()) return line_name;
    if (event_types.count(CuptiTracerEventType::Overhead))
      return "CUPTI overhead";
    std::vector<const char*> type_names;
    for (const auto event_type : event_types) {
      type_names.emplace_back(GetTraceEventTypeName(event_type));
    }
    return absl::StrCat(line_name, "(", absl::StrJoin(type_names, ","), ")");
  }

 public:
  PerDeviceCollector() = default;

  void AddEvent(CuptiTracerEvent&& event) {
    mutex_lock l(m_);
    events_.emplace_back(std::move(event));
  }

  size_t Flush(tsl::uint64 start_gpu_ns, tsl::uint64 end_gpu_ns,
               XPlaneBuilder* device_plane, XPlaneBuilder* host_plane) {
    mutex_lock l(m_);
    // Tracking event types per line.
    absl::flat_hash_map<int64_t, absl::flat_hash_set<CuptiTracerEventType>>
        events_types_per_line;
    for (auto& event : events_) {
      int64_t line_id = CuptiTracerEvent::kInvalidThreadId;
      bool is_host_event = IsHostEvent(event, &line_id);
      if (line_id == CuptiTracerEvent::kInvalidThreadId ||
          line_id == CuptiTracerEvent::kInvalidStreamId) {
        VLOG(9) << "Ignoring event, type=" << static_cast<int>(event.type);
        continue;
      }
      auto* plane = is_host_event ? host_plane : device_plane;
      VLOG(9) << "Event"
              << " type=" << static_cast<int>(event.type)
              << " line_id=" << line_id
              << (is_host_event ? " host plane=" : " device plane=")
              << plane->Name();
      XLineBuilder line = plane->GetOrCreateLine(line_id);
      line.SetTimestampNs(start_gpu_ns);
      CreateXEvent(event, plane, start_gpu_ns, end_gpu_ns, &line);
      events_types_per_line[line_id].emplace(event.type);
    }
    device_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(
          GetDeviceXLineName(line.Id(), events_types_per_line[line.Id()]));
    });
    host_plane->ForEachLine([&](XLineBuilder line) {
      line.SetName(absl::StrCat("Host Threads/", line.Id()));
    });
    size_t num_events = events_.size();
    events_.clear();
    return num_events;
  }

  void GetDeviceCapabilities(int32_t device_ordinal,
                             XPlaneBuilder* device_plane) {
    device_plane->AddStatValue(*device_plane->GetOrCreateStatMetadata(
                                   GetStatTypeStr(StatType::kDevVendor)),
                               kDeviceVendorNvidia);

    CUdevice device;
    if (cuDeviceGet(&device, device_ordinal) != CUDA_SUCCESS) return;

    auto clock_rate_in_khz =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
    if (clock_rate_in_khz) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapClockRateKHz)),
          *clock_rate_in_khz);
    }

    auto core_count =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
    if (core_count) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapCoreCount)),
          *core_count);
    }

    auto mem_clock_khz =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
    auto mem_bus_width_bits =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
    if (mem_clock_khz && mem_bus_width_bits) {
      // Times 2 because HBM is DDR memory; it gets two data bits per each
      // data lane.
      auto memory_bandwidth =
          tsl::uint64{2} * (*mem_clock_khz) * 1000 * (*mem_bus_width_bits) / 8;
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemoryBandwidth)),
          memory_bandwidth);
    }

    size_t total_memory = 0;
    if (cuDeviceTotalMem(&total_memory, device) == CUDA_SUCCESS) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapMemorySize)),
          static_cast<tsl::uint64>(total_memory));
    }

    auto compute_capability_major = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    if (compute_capability_major) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMajor)),
          *compute_capability_major);
    }
    auto compute_capability_minor = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    if (compute_capability_minor) {
      device_plane->AddStatValue(
          *device_plane->GetOrCreateStatMetadata(
              GetStatTypeStr(StatType::kDevCapComputeCapMinor)),
          *compute_capability_minor);
    }

    auto max_threads_per_block =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
    auto max_threads_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
    auto regs_per_block =
        GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK);
    auto regs_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
    auto warp_size = GetDeviceAttribute(device, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
    auto shared_mem_per_block = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
    auto shared_mem_per_sm = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
    auto shared_mem_per_block_optin = GetDeviceAttribute(
        device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);

    // Precondition for calculating GPU occupancy is to have all of these
    // inputs. Otherwise, GPU occupancy will be left unset as 0%.
    if (core_count && compute_capability_major && compute_capability_minor &&
        max_threads_per_block && max_threads_per_sm && regs_per_block &&
        regs_per_sm && warp_size && shared_mem_per_block && shared_mem_per_sm &&
        shared_mem_per_block_optin) {
      device_properties_.computeMajor = *compute_capability_major;
      device_properties_.computeMinor = *compute_capability_minor;
      device_properties_.numSms = *core_count;
      device_properties_.maxThreadsPerBlock = *max_threads_per_block;
      device_properties_.maxThreadsPerMultiprocessor = *max_threads_per_sm;
      device_properties_.regsPerBlock = *regs_per_block;
      device_properties_.regsPerMultiprocessor = *regs_per_sm;
      device_properties_.warpSize = *warp_size;
      device_properties_.sharedMemPerBlock = *shared_mem_per_block;
      device_properties_.sharedMemPerMultiprocessor = *shared_mem_per_sm;
      device_properties_.sharedMemPerBlockOptin = *shared_mem_per_block_optin;
    }
  }

 private:
  mutex m_;
  std::vector<CuptiTracerEvent> events_ TF_GUARDED_BY(m_);
  cudaOccDeviceProp device_properties_;
  absl::flat_hash_map<DeviceOccupancyParams, OccupancyStats> occupancy_cache_;
};

}  // namespace

void AnnotationMap::Add(tsl::uint32 device_id, tsl::uint32 correlation_id,
                        const absl::string_view annotation,
                        const absl::string_view nvtx_range) {
  if (annotation.empty() && nvtx_range.empty()) return;
  VLOG(3) << "Add annotation: device_id: " << device_id
          << " correlation_id: " << correlation_id
          << " annotation: " << annotation;
  if (device_id >= per_device_map_.size()) return;
  auto& per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  if (per_device_map.annotations.size() < max_size_) {
    AnnotationInfo info;
    info.annotation = *per_device_map.annotations.emplace(annotation).first;
    if (!nvtx_range.empty())
      info.nvtx_range = *per_device_map.nvtx_ranges.emplace(nvtx_range).first;
    per_device_map.correlation_map.emplace(correlation_id, info);
  }
}

AnnotationMap::AnnotationInfo AnnotationMap::LookUp(
    tsl::uint32 device_id, tsl::uint32 correlation_id) {
  if (device_id >= per_device_map_.size()) return AnnotationInfo();
  auto& per_device_map = per_device_map_[device_id];
  absl::MutexLock lock(&per_device_map.mutex);
  auto it = per_device_map.correlation_map.find(correlation_id);
  return it != per_device_map.correlation_map.end() ? it->second
                                                    : AnnotationInfo();
}

// CuptiTraceCollectorImpl store the CuptiTracerEvents from CuptiTracer and
// eventually convert and filter them to XSpace.
class CuptiTraceCollectorImpl : public CuptiTraceCollector {
 public:
  CuptiTraceCollectorImpl(const CuptiTracerCollectorOptions& option,
                          tsl::uint64 start_walltime_ns,
                          tsl::uint64 start_gpu_ns)
      : CuptiTraceCollector(option),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gpu_ns_(start_gpu_ns),
        num_gpus_(option.num_gpus),
        per_device_collector_(option.num_gpus) {}

  void AddEvent(CuptiTracerEvent&& event) override {
    if (event.device_id >= num_gpus_) return;
    if (event.source == CuptiTracerEventSource::DriverCallback) {
      if (num_callback_events_ > options_.max_callback_api_events) {
        OnEventsDropped("total driver(callback) events reaches max", 1);
        return;
      }
      num_callback_events_++;
    } else {
      if (num_activity_events_ > options_.max_activity_api_events) {
        OnEventsDropped("total device(activity) events reaches max", 1);
        return;
      }
      num_activity_events_++;
    }
    per_device_collector_[event.device_id].AddEvent(std::move(event));
  }
  void OnEventsDropped(const std::string& reason,
                       tsl::uint32 num_events) override {
    absl::MutexLock lock(&mutex_);
    dropped_events_[reason] += num_events;
  }
  void Flush() override {}
  // Returns true if some GPU events are captured.
  bool Export(XSpace* space, tsl::uint64 end_gpu_ns) override {
    LOG(INFO) << " GpuTracer has collected " << num_callback_events_
              << " callback api events and " << num_activity_events_
              << " activity events. " << ReportDroppedEvents();
    size_t num_events = 0;
    XPlaneBuilder host_plane(
        FindOrAddMutablePlaneWithName(space, kCuptiDriverApiPlaneName));
    for (int device_ordinal = 0; device_ordinal < num_gpus_; ++device_ordinal) {
      std::string name = GpuPlaneName(device_ordinal);
      XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
      device_plane.SetId(device_ordinal);
      VLOG(4) << "Creating plane for"
              << " name=" << name << " ordinal=" << device_ordinal;

      // Calculate device capabilities before flushing, so that device
      // properties are available to the occupancy calculator in Flush().
      per_device_collector_[device_ordinal].GetDeviceCapabilities(
          device_ordinal, &device_plane);
      num_events += per_device_collector_[device_ordinal].Flush(
          start_gpu_ns_, end_gpu_ns, &device_plane, &host_plane);
      NormalizeTimeStamps(&device_plane, start_walltime_ns_);
    }
    NormalizeTimeStamps(&host_plane, start_walltime_ns_);
    return num_events > 0;
  }

  std::string ReportDroppedEvents() {
    absl::MutexLock lock(&mutex_);
    std::string result;
    for (const auto& dropped : dropped_events_) {
      absl::StrAppend(&result, " ", dropped.second, " events dropped because ",
                      dropped.first, ";");
    }
    if (!result.empty()) result.back() = '.';
    return result;
  }
  std::string ReportNumEventsIfDropped() override {
    std::string events_dropped = ReportDroppedEvents();
    if (events_dropped.empty()) return "";
    return absl::StrCat("Detected GPU events dropped on ",
                        tsl::port::Hostname(), ": Profiler has collected ",
                        num_callback_events_.load(), " driver events and ",
                        num_activity_events_.load(), " device events.",
                        events_dropped);
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  absl::Mutex mutex_;
  absl::flat_hash_map<std::string, tsl::uint64> dropped_events_
      ABSL_GUARDED_BY(mutex_);
  tsl::uint64 start_walltime_ns_;
  tsl::uint64 start_gpu_ns_;
  int num_gpus_;

  // Set the all XLines of specified XPlane to starting walltime.
  // Events time in both host and device planes are CUTPI timestamps.
  // We set initial CUPTI timestamp as start time for all lines to reflect
  // this fact. Eventually we change line start time to corresponding
  // start_walltime_ns to normalize with CPU wall time.
  static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                  tsl::uint64 start_walltime_ns) {
    plane->ForEachLine(
        [&](XLineBuilder line) { line.SetTimestampNs(start_walltime_ns); });
  }

  absl::FixedArray<PerDeviceCollector> per_device_collector_;

  TF_DISALLOW_COPY_AND_ASSIGN(CuptiTraceCollectorImpl);
};

std::unique_ptr<CuptiTraceCollector> CreateCuptiCollector(
    const CuptiTracerCollectorOptions& options,
    const tsl::uint64 start_walltime_ns, const tsl::uint64 start_gputime_ns) {
  return std::make_unique<CuptiTraceCollectorImpl>(options, start_walltime_ns,
                                                   start_gputime_ns);
}

// The strings are parser friendly and have no whitespaces in them.
absl::string_view GetMemoryKindName(int8_t memory_kind) {
  switch (memory_kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "array";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "device";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
      return "device_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
      return "managed";
    case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
      return "managed_static";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
    default:
      return "unknown";
  }
}

}  // namespace profiler
}  // namespace xla
