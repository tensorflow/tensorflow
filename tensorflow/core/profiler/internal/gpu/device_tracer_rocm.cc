/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/internal/cpu/annotation_stack.h"
#include "tensorflow/core/profiler/internal/gpu/rocm_tracer.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/utils/parse_annotation.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

namespace {
// Set the all XLines of specified XPlane to starting walltime.
// Events time in both host and device planes are CUTPI timestamps.
// We set initial RocmTracer timestamp as start time for all lines to reflect
// this fact. Eventually we change line start time to corresponding
// start_walltime_ns to normalize with CPU wall time.
static void NormalizeTimeStamps(XPlaneBuilder* plane,
                                uint64_t start_walltime_ns) {
  plane->ForEachLine([&](tensorflow::profiler::XLineBuilder line) {
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

class RocmTraceCollectorImpl : public profiler::RocmTraceCollector {
 public:
  RocmTraceCollectorImpl(const RocmTraceCollectorOptions& options,
                         uint64_t start_walltime_ns, uint64_t start_gputime_ns)
      : RocmTraceCollector(options),
        num_callback_events_(0),
        num_activity_events_(0),
        start_walltime_ns_(start_walltime_ns),
        start_gputime_ns_(start_gputime_ns),
        per_device_collector_(options.num_gpus) {}

  void AddEvent(RocmTracerEvent&& event, bool is_auxiliary) override {
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
      if (event.domain == RocmTracerEventDomain::HIP_API) {
        std::tie(std::ignore, emplace_result) =
            activity_api_events_map_.emplace(event.correlation_id,
                                             std::move(event));
      } else if (event.domain == RocmTracerEventDomain::HCC_OPS) {
        auto result = activity_ops_events_map_.emplace(
            event.correlation_id, std::vector<RocmTracerEvent>{});
        result.first->second.push_back(std::move(event));
        emplace_result = true;  // we always accept Hip-Ops events
      }
    }
    if (!emplace_result) {
      OnEventsDropped("event with duplicate correlation_id was received.",
                      event.correlation_id);
      DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
    }
  }

  void OnEventsDropped(const std::string& reason,
                       uint32_t correlation_id) override {
    LOG(INFO) << "RocmTracerEvent dropped (correlation_id=" << correlation_id
              << ",) : " << reason << ".";
  }

  void Flush() override {
    mutex_lock lock(event_maps_mutex_);
    auto& aggregated_events_ = ApiActivityInfoExchange();

    VLOG(3) << "RocmTraceCollector collected " << num_callback_events_
            << " callback events, " << num_activity_events_
            << " activity events, and aggregated them into "
            << aggregated_events_.size() << " events.";

    for (auto& event : aggregated_events_) {
      if (event.device_id >= options_.num_gpus) {
        OnEventsDropped("device id >= num gpus", event.correlation_id);
        DumpRocmTracerEvent(event, 0, 0, ". Dropped!");
        LOG(WARNING) << "A ROCm profiler event record with wrong device ID "
                        "dropped! Type="
                     << GetRocmTracerEventTypeName(event.type);
        continue;
      }

      activity_api_events_map_.clear();
      activity_ops_events_map_.clear();
      api_events_map_.clear();
      auxiliary_api_events_map_.clear();

      per_device_collector_[event.device_id].AddEvent(event);
    }

    for (int i = 0; i < options_.num_gpus; ++i) {
      per_device_collector_[i].SortByStartTime();
    }
  }

  void Export(XSpace* space) {
    uint64_t end_gputime_ns = RocmTracer::GetTimestamp();
    XPlaneBuilder host_plane(
        FindOrAddMutablePlaneWithName(space, kRoctracerApiPlaneName));
    for (int i = 0; i < options_.num_gpus; ++i) {
      std::string name = GpuPlaneName(i);
      XPlaneBuilder device_plane(FindOrAddMutablePlaneWithName(space, name));
      device_plane.SetId(i);
      // Calculate device capabilities before flushing, so that device
      // properties are available to the occupancy calculator in export().
      per_device_collector_[i].GetDeviceCapabilities(i, &device_plane);
      per_device_collector_[i].Export(start_walltime_ns_, start_gputime_ns_,
                                      end_gputime_ns, &device_plane,
                                      &host_plane);

      NormalizeTimeStamps(&device_plane, start_walltime_ns_);
    }
    NormalizeTimeStamps(&host_plane, start_walltime_ns_);
  }

 private:
  std::atomic<int> num_callback_events_;
  std::atomic<int> num_activity_events_;
  uint64_t start_walltime_ns_;
  uint64_t start_gputime_ns_;

  mutex event_maps_mutex_;
  absl::flat_hash_map<uint32, RocmTracerEvent> api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);
  absl::flat_hash_map<uint32, RocmTracerEvent> activity_api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  /* Some apis such as MEMSETD32 (based on an observation with ResNet50),
    trigger multiple HIP ops domain activities. We keep them in a vector and
    merge them with api activities at flush time.
  */
  absl::flat_hash_map<uint32, std::vector<RocmTracerEvent>>
      activity_ops_events_map_ TF_GUARDED_BY(event_maps_mutex_);
  // This is for the APIs that we track because we need some information from
  // them to populate the corresponding activity that we actually track.
  absl::flat_hash_map<uint32, RocmTracerEvent> auxiliary_api_events_map_
      TF_GUARDED_BY(event_maps_mutex_);

  const std::vector<RocmTracerEvent> ApiActivityInfoExchange() {
    /* Different from CUDA, roctracer activity records are not enough to fill a
      TF event. For most of the activities, we need to enable the corresponding
      API callsbacks (we call them auxiliary API callbacks) to capture the
      necessary fields from them using the correlation id. The purpose of this
      function is to let APIs and activities exchange information to reach a
      state very similar to TF CUDA and getting ready to dump the event.
    */

    // Copying info from HIP-OPS activities to HIP-API activities
    /*HIP-API activities <<==== HIP-OPS activities*/
    auto activity_api_events_map_iter = activity_api_events_map_.begin();
    while (activity_api_events_map_iter != activity_api_events_map_.end()) {
      uint32_t activity_corr_id = activity_api_events_map_iter->first;
      RocmTracerEvent& activity_api_event =
          activity_api_events_map_iter->second;

      bool result = false;
      switch (activity_api_event.type) {
        case RocmTracerEventType::Kernel:
        case RocmTracerEventType::Memset: {
          // KERNEL & MEMSET
          auto iter =
              activity_ops_events_map_.find(activity_api_event.correlation_id);
          result = (iter != activity_ops_events_map_.end());
          if (result) {
            // since the key exist in the map, there should be at least one item
            // in the vector
            activity_api_event.device_id = iter->second.front().device_id;
            activity_api_event.stream_id = iter->second.front().stream_id;
            // we initialize the start time and end time based on the first
            // element
            activity_api_event.start_time_ns =
                iter->second.front().start_time_ns;
            activity_api_event.end_time_ns = iter->second.front().end_time_ns;
            for (auto& kernel_activity_op : iter->second) {
              activity_api_event.start_time_ns =
                  std::min(activity_api_event.start_time_ns,
                           kernel_activity_op.start_time_ns);
              activity_api_event.end_time_ns =
                  std::max(activity_api_event.end_time_ns,
                           kernel_activity_op.end_time_ns);
            }
          }
          break;
        }
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyOther: {
          // MEMCPY
          auto iter =
              activity_ops_events_map_.find(activity_api_event.correlation_id);
          result = (iter != activity_ops_events_map_.end());
          if (result) {
            // since the key exist in the map, there should be at least one item
            // in the vector
            activity_api_event.device_id = iter->second.front().device_id;
            activity_api_event.memcpy_info.destination =
                iter->second.front()
                    .memcpy_info.destination;  // similar to CUDA, it is the
                                               // same as device_id
            activity_api_event.stream_id = iter->second.front().stream_id;
            /* IMPORTANT: it seems that the HCC timing is only valid for
             * Synchronous memcpy activities*/
            if (!activity_api_event.memcpy_info.async) {
              activity_api_event.start_time_ns =
                  iter->second.front().start_time_ns;
              activity_api_event.end_time_ns = iter->second.front().end_time_ns;
              for (auto& kernel_activity_op : iter->second) {
                activity_api_event.start_time_ns =
                    std::min(activity_api_event.start_time_ns,
                             kernel_activity_op.start_time_ns);
                activity_api_event.end_time_ns =
                    std::max(activity_api_event.end_time_ns,
                             kernel_activity_op.end_time_ns);
              }
            }
          }
          break;
        }
        default:
          // nothing to do for the rest
          result = true;
          break;
      }
      if (!result) {
        OnEventsDropped(
            "A HIP-API activity with missing HIP-OPS activity was found",
            activity_api_event.correlation_id);
        DumpRocmTracerEvent(activity_api_event, 0, 0, ". Dropped!");
        activity_api_events_map_.erase(activity_api_events_map_iter++);
      } else {
        ++activity_api_events_map_iter;
      }
    }

    // the event vector to be returned
    std::vector<RocmTracerEvent> aggregated_events;

    // Copying info from HIP activities to HIP API callbacks
    /*HIP-API call backs <<==== HIP-API activities*/
    for (auto& api_iter : api_events_map_) {
      RocmTracerEvent& api_event = api_iter.second;
      auto iter = activity_api_events_map_.find(api_event.correlation_id);
      switch (api_event.type) {
        /*KERNEL API*/
        case RocmTracerEventType::Kernel: {
          aggregated_events.push_back(api_event);
          break;
        }
        /*MEMCPY API*/
        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyOther: {
          if (iter != activity_api_events_map_.end()) {
            api_event.device_id = iter->second.device_id;
            api_event.memcpy_info.destination =
                api_event.device_id;  // Similar to CUDA
            aggregated_events.push_back(api_event);
          } else {
            OnEventsDropped(
                "A Memcpy event from HIP API discarded."
                " Could not find the counterpart activity.",
                api_event.correlation_id);
            DumpRocmTracerEvent(api_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*MEMSET API*/
        case RocmTracerEventType::Memset: {
          if (iter != activity_api_events_map_.end()) {
            api_event.device_id = iter->second.device_id;

            aggregated_events.push_back(api_event);
          } else {
            OnEventsDropped(
                "A Memset event from HIP API discarded."
                " Could not find the counterpart activity.",
                api_event.correlation_id);
            DumpRocmTracerEvent(api_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*MALLOC API, FREE API*/
        case RocmTracerEventType::MemoryAlloc:
        case RocmTracerEventType::MemoryFree: {
          // no missing info
          aggregated_events.push_back(api_event);
          break;
        }
        /*SYNCHRONIZATION API*/
        case RocmTracerEventType::Synchronization: {
          // no missing info
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
          break;
      }  // end switch(api_event.type)
    }

    // Copying info from HIP API callbacks to HIP API activities
    //  API ACTIVITIES<<====API-CB
    for (auto& activity_iter : activity_api_events_map_) {
      RocmTracerEvent& activity_event = activity_iter.second;
      // finding the corresponding activity either in the api_call backs or the
      // axuilarities
      auto iter = api_events_map_.find(activity_event.correlation_id);

      iter = (iter == api_events_map_.end())
                 ? auxiliary_api_events_map_.find(activity_event.correlation_id)
                 : iter;
      switch (activity_event.type) {
        /*KERNEL ACTIVITY*/
        case RocmTracerEventType::Kernel: {
          if (iter != api_events_map_.end() ||
              iter != auxiliary_api_events_map_.end()) {
            activity_event.name = iter->second.name;
            activity_event.kernel_info = iter->second.kernel_info;
            aggregated_events.push_back(activity_event);
          } else {
            OnEventsDropped(
                "A Kernel event activity was discarded."
                " Could not find the counterpart API callback.",
                activity_event.correlation_id);
            DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*MEMCPY ACTIVITY*/
        case RocmTracerEventType::MemcpyD2H:
        case RocmTracerEventType::MemcpyH2D:
        case RocmTracerEventType::MemcpyD2D:
        case RocmTracerEventType::MemcpyOther: {
          if (iter != api_events_map_.end() ||
              iter != auxiliary_api_events_map_.end()) {
            activity_event.memcpy_info = iter->second.memcpy_info;
            aggregated_events.push_back(activity_event);
          } else {
            OnEventsDropped(
                "A Memcpy event activity was discarded."
                " Could not find the counterpart API callback.",
                activity_event.correlation_id);
            DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*MEMSET ACTIVITY*/
        case RocmTracerEventType::Memset: {
          if (iter != api_events_map_.end() ||
              iter != auxiliary_api_events_map_.end()) {
            activity_event.memset_info = iter->second.memset_info;
            aggregated_events.push_back(activity_event);

          } else {
            OnEventsDropped(
                "A Memset event activity was discarded."
                " Could not find the counterpart API callback.",
                activity_event.correlation_id);
            DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*MALLOC ACTIVITY, FREE ACTIVITY*/
        case RocmTracerEventType::MemoryAlloc:
        case RocmTracerEventType::MemoryFree: {
          if (iter != api_events_map_.end() ||
              iter != auxiliary_api_events_map_.end()) {
            activity_event.device_id = iter->second.device_id;
            aggregated_events.push_back(activity_event);
          } else {
            OnEventsDropped(
                "A Malloc/Free activity was discarded."
                " Could not find the counterpart API callback.",
                activity_event.correlation_id);
            DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          }
          break;
        }
        /*SYNCHRONIZATION ACTIVITY*/
        case RocmTracerEventType::Synchronization: {
          if (iter != api_events_map_.end() ||
              iter != auxiliary_api_events_map_.end()) {
            // CUDA does not provide device ID for these activities.
            // Interestingly, TF-profiler by default set the device id to 0 for
            // CuptiTracerEvent.
            // RocmTracerEvent type, set device by default to an unvalid
            // device-id value. To be consistent with CUDA (in terms of having a
            // logically valid value for device id) we update the device-id to
            // its correct value
            activity_event.device_id = iter->second.device_id;
            aggregated_events.push_back(activity_event);
          } else {
            OnEventsDropped(
                "A sync event activity was discarded."
                " Could not find the counterpart API callback.",
                activity_event.correlation_id);
            DumpRocmTracerEvent(activity_event, 0, 0, ". Dropped!");
          }
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
      }  // end switch(activity_event.type)
    }

    return aggregated_events;
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
    CorrelationInfo(uint32_t t, uint32_t e)
        : thread_id(t), enqueue_time_ns(e) {}
    uint32_t thread_id;
    uint64_t enqueue_time_ns;
  };

  struct PerDeviceCollector {
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
            uint64{2} * (mem_clock_khz)*1000 * (mem_bus_width_bits) / 8;
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
            static_cast<uint64>(total_memory));
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

    inline std::string ToXStat(const KernelDetails& kernel_info,
                               double occupancy_pct) {
      return absl::StrCat(
          "regs:", kernel_info.registers_per_thread,
          " static_shared:", kernel_info.static_shared_memory_usage,
          " dynamic_shared:", kernel_info.dynamic_shared_memory_usage,
          " grid:", kernel_info.grid_x, ",", kernel_info.grid_y, ",",
          kernel_info.grid_z, " block:", kernel_info.block_x, ",",
          kernel_info.block_y, ",", kernel_info.block_z,
          " occ_pct:", occupancy_pct);
    }
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
          &stats.min_grid_size, &stats.suggested_block_size, params.func_ptr,
          params.dynamic_smem_size, 0);

      if (err != hipError_t::hipSuccess) {
        return {};
      }

      return stats;
    }
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

    void SortByStartTime() {
      mutex_lock lock(events_mutex);
      std::sort(
          events.begin(), events.end(),
          [](const RocmTracerEvent& event1, const RocmTracerEvent& event2) {
            return event1.start_time_ns < event2.start_time_ns;
          });
    }

    void CreateXEvent(const RocmTracerEvent& event, XPlaneBuilder* plane,
                      uint64_t start_gpu_ns, uint64_t end_gpu_ns,
                      XLineBuilder* line) {
      if (event.start_time_ns < start_gpu_ns ||
          event.end_time_ns > end_gpu_ns ||
          event.start_time_ns > event.end_time_ns) {
        VLOG(2) << "events have abnormal timestamps:" << event.name
                << " start time(ns): " << event.start_time_ns
                << " end time(ns): " << event.end_time_ns
                << " start gpu(ns):" << start_gpu_ns
                << " end gpu(ns):" << end_gpu_ns
                << " corr. id:" << event.correlation_id;
        return;
      }
      std::string kernel_name = port::MaybeAbiDemangle(event.name.c_str());
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
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kDeviceId)),
                            event.device_id);
      }
      if (event.correlation_id != RocmTracerEvent::kInvalidCorrelationId) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kCorrelationId)),
                            event.correlation_id);
      }
      if (!event.roctx_range.empty()) {
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(
                                GetStatTypeStr(StatType::kNVTXRange)),
                            *plane->GetOrCreateStatMetadata(event.roctx_range));
      }
      // if (event.context_id != CuptiTracerEvent::kInvalidContextId) {
      //   xevent.AddStatValue(
      //       *plane->GetOrCreateStatMetadata(
      //           GetStatTypeStr(StatType::kContextId)),
      //       absl::StrCat("$$", static_cast<uint64>(event.context_id)));
      // }

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

        params.dynamic_smem_size =
            event.kernel_info.dynamic_shared_memory_usage;
        params.func_ptr = event.kernel_info.func_ptr;

        OccupancyStats& occ_stats = occupancy_cache_[params];
        if (occ_stats.occupancy_pct == 0.0) {
          occ_stats = GetOccupancy(params);
        }
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                                StatType::kTheoreticalOccupancyPct)),
                            occ_stats.occupancy_pct);
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                                StatType::kOccupancyMinGridSize)),
                            static_cast<int32>(occ_stats.min_grid_size));
        xevent.AddStatValue(*plane->GetOrCreateStatMetadata(GetStatTypeStr(
                                StatType::kOccupancySuggestedBlockSize)),
                            static_cast<int32>(occ_stats.suggested_block_size));
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
    bool IsHostEvent(const RocmTracerEvent& event, int64* line_id) {
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
        *line_id = kThreadIdOverhead;
        return false;
      }
    }
    void Export(uint64_t start_walltime_ns, uint64_t start_gputime_ns,
                uint64_t end_gputime_ns, XPlaneBuilder* device_plane,
                XPlaneBuilder* host_plane) {
      int host_ev_cnt = 0, dev_ev_cnt = 0;
      mutex_lock l(events_mutex);
      // Tracking event types per line.
      absl::flat_hash_map<int64, absl::flat_hash_set<RocmTracerEventType>>
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
        VLOG(9) << "Event"
                << " type=" << static_cast<int>(event.type)
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
      size_t num_events = events.size();
      events.clear();
    }

    mutex events_mutex;
    std::vector<RocmTracerEvent> events TF_GUARDED_BY(events_mutex);
    absl::flat_hash_map<uint32, CorrelationInfo> correlation_info_
        TF_GUARDED_BY(events_mutex);
    absl::flat_hash_map<RocmDeviceOccupancyParams, OccupancyStats>
        occupancy_cache_;
    hipDeviceProp_t device_properties_;
  };

  absl::FixedArray<PerDeviceCollector> per_device_collector_;
};

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
    LOG(INFO) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(XSpace* space) override;

 private:
  Status DoStart();
  Status DoStop();
  Status DoCollectData(XSpace* space);

  RocmTracerOptions GetRocmTracerOptions();

  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32_t num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollectorImpl> rocm_trace_collector_;
};

RocmTracerOptions GpuTracer::GetRocmTracerOptions() {
  // TODO(rocm-profiler): We need support for context similar to CUDA
  RocmTracerOptions options;
  std::vector<uint32_t> empty_vec;

  // clang formatting does not preserve one entry per line
  // clang-format off
  std::vector<uint32_t> hip_api_domain_ops{
      // KERNEL
      HIP_API_ID_hipExtModuleLaunchKernel,
      HIP_API_ID_hipModuleLaunchKernel,
      HIP_API_ID_hipHccModuleLaunchKernel,
      HIP_API_ID_hipLaunchKernel,
      // MEMCPY
      HIP_API_ID_hipMemcpy,
      HIP_API_ID_hipMemcpyAsync,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipMemcpyPeer,
      HIP_API_ID_hipMemcpyPeerAsync,

      // MEMSet
      HIP_API_ID_hipMemsetD32,
      HIP_API_ID_hipMemsetD32Async,
      HIP_API_ID_hipMemsetD16,
      HIP_API_ID_hipMemsetD16Async,
      HIP_API_ID_hipMemsetD8,
      HIP_API_ID_hipMemsetD8Async,
      HIP_API_ID_hipMemset,
      HIP_API_ID_hipMemsetAsync,

      // MEMAlloc
      HIP_API_ID_hipMalloc,
      HIP_API_ID_hipMallocPitch,
      // MEMFree
      HIP_API_ID_hipFree,
      // GENERIC
      HIP_API_ID_hipStreamSynchronize,
  };
  // clang-format on

  options.api_tracking_set =
      std::set<uint32_t>(hip_api_domain_ops.begin(), hip_api_domain_ops.end());

  // These are the list of APIs we track since roctracer activity
  // does not provide all the information necessary to fully populate the
  // TF events. We need to track the APIs for those activities in API domain but
  // we only use them for filling the missing items in their corresponding
  // activity (using correlation id).
  // clang-format off
  std::vector<uint32_t> hip_api_aux_ops{
    HIP_API_ID_hipStreamWaitEvent,
    // TODO(rocm-profiler): finding device ID from hipEventSynchronize need some
    // extra work, we ignore it for now.
    // HIP_API_ID_hipEventSynchronize,
    HIP_API_ID_hipHostFree,
    HIP_API_ID_hipHostMalloc,
    HIP_API_ID_hipSetDevice  //  added to track default device
  };
  // clang-format on

  hip_api_domain_ops.insert(hip_api_domain_ops.end(), hip_api_aux_ops.begin(),
                            hip_api_aux_ops.end());

  options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, hip_api_domain_ops);
  // options.api_callbacks.emplace(ACTIVITY_DOMAIN_ROCTX, empty_vec);
  // options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);

  // options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API,
  // hip_api_domain_ops);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);
  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HCC_OPS, empty_vec);

  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32_t num_gpus) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

Status GpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  AnnotationStack::Enable(true);

  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tensorflow::EnvTime::NowNanos();
  rocm_trace_collector_ = std::make_unique<RocmTraceCollectorImpl>(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  RocmTracerOptions tracer_options = GetRocmTracerOptions();
  rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());

  return Status::OK();
}

Status GpuTracer::Start() {
  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return Status::OK();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status GpuTracer::DoStop() {
  rocm_tracer_->Disable();
  AnnotationStack::Enable(false);
  return Status::OK();
}

Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return Status::OK();
}

Status GpuTracer::DoCollectData(XSpace* space) {
  if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
  return Status::OK();
}

Status GpuTracer::CollectData(XSpace* space) {
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(3) << "No trace data collected, session wasn't started";
      return Status::OK();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return Status::OK();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
      return Status::OK();
    case State::kStoppedOk: {
      DoCollectData(space);
      return Status::OK();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
    return nullptr;

  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer->IsAvailable()) return nullptr;

  return absl::make_unique<profiler::GpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_ROCM
