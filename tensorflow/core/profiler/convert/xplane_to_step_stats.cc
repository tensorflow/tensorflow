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

#include "tensorflow/core/profiler/convert/xplane_to_step_stats.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/profiler/utils/gpu_event_stats.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/gpu_event_stats.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

struct CorrelationInfo {
  uint32_t thread_id;
  uint64_t enqueue_time_ns;
};

enum GpuEventType {
  kUnknown,
  kKernel,
  kMemcpyH2D,
  kMemcpyD2H,
  kMemcpyD2D,
  kMemcpyP2P,
};

GpuEventType ParseMemcpyName(absl::string_view memcpy_name) {
  if (absl::ConsumePrefix(&memcpy_name, "Memcpy")) {
    if (memcpy_name == "H2D") return GpuEventType::kMemcpyH2D;
    if (memcpy_name == "D2H") return GpuEventType::kMemcpyD2H;
    if (memcpy_name == "D2D") return GpuEventType::kMemcpyD2D;
    if (memcpy_name == "P2P") return GpuEventType::kMemcpyP2P;
  }
  return GpuEventType::kUnknown;
}

void SetNodeTimes(uint64_t start_time, const XEventVisitor& event,
                  NodeExecStats* ns) {
  // Since XPlane uses relative times, we need to convert event.TimestampNs() to
  // absolute times
  ns->set_all_start_micros(
      tsl::profiler::NanoToMicro(start_time + event.TimestampNs()));
  ns->set_op_start_rel_micros(0);
  ns->set_op_end_rel_micros(tsl::profiler::NanoToMicro(event.DurationNs()));
  ns->set_all_end_rel_micros(tsl::profiler::NanoToMicro(event.DurationNs()));
}

}  // namespace

void ConvertGpuXSpaceToStepStats(const XSpace& xspace, StepStats* step_stats) {
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
  if (device_planes.empty()) {
    LOG(WARNING) << "GPU trace was not collected.";
    return;
  }

  const XPlane* env_plane = FindPlaneWithName(xspace, kTaskEnvPlaneName);
  XPlaneVisitor env_plane_visitor(env_plane, {}, {FindTaskEnvStatType});
  uint64_t start_time =
      env_plane_visitor.GetStat(TaskEnvStatType::kEnvProfileStartTime)
          ->IntOrUintValue();

  const XPlane* host_plane = FindPlaneWithName(xspace, kHostThreadsPlaneName);
  DCHECK_NE(host_plane, nullptr);

  absl::flat_hash_map<int64_t /*correlation_id*/, CorrelationInfo>
      correlation_info_map;

  absl::flat_hash_map<uint32_t /*device_id*/, DeviceStepStats*>
      sync_dev_stats_map;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(host_plane);
  plane.ForEachLine([&](const XLineVisitor& line) {
    uint32_t thread_id = line.Id();
    line.ForEachEvent([&](const XEventVisitor& event) {
      LaunchEventStats stats(&event);
      if (event.Name() == "cuStreamSynchronize") {
        if (stats.device_id.has_value()) {
          uint32_t device_ordinal = stats.device_id.value();
          DeviceStepStats* sync_dev_stats = sync_dev_stats_map[device_ordinal];
          if (sync_dev_stats == nullptr) {
            sync_dev_stats = step_stats->add_dev_stats();
            sync_dev_stats->set_device(
                absl::StrCat("/device:GPU:", device_ordinal, "/sync"));
          }
          NodeExecStats* ns = sync_dev_stats->add_node_stats();
          SetNodeTimes(start_time, event, ns);
          ns->set_node_name(std::string(event.Name()));
          ns->set_timeline_label(absl::StrCat("ThreadId ", thread_id));
          ns->set_thread_id(thread_id);
        }
      } else if (stats.correlation_id.has_value()) {
        int64_t correlation_id = stats.correlation_id.value();
        uint64_t enqueue_time_ns = event.TimestampNs();
        correlation_info_map[correlation_id] = {thread_id, enqueue_time_ns};
      }
    });
  });
  for (const XPlane* device_plane : device_planes) {
    absl::flat_hash_map<std::pair<int64_t /*stream_id*/, GpuEventType>,
                        DeviceStepStats*>
        stream_dev_stats_map;
    DeviceStepStats* unknown_stream_dev_stats = nullptr;
    DeviceStepStats* all_streams_dev_stats = nullptr;
    DeviceStepStats* memcpy_dev_stats = nullptr;
    XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(device_plane);
    uint32_t device_ordinal = plane.Id();
    plane.ForEachLine([&](const XLineVisitor& line) {
      uint32_t stream_id = line.Id();
      line.ForEachEvent([&](const XEventVisitor& event) {
        GpuEventStats stats(&event);

        auto ns = std::make_unique<NodeExecStats>();
        SetNodeTimes(start_time, event, ns.get());

        // Get launch information if available.
        if (stats.correlation_id.has_value()) {
          auto it = correlation_info_map.find(stats.correlation_id.value());
          if (it != correlation_info_map.end()) {
            const CorrelationInfo& correlation_info = it->second;
            ns->set_scheduled_micros(
                tsl::profiler::NanoToMicro(correlation_info.enqueue_time_ns));
            ns->set_thread_id(correlation_info.thread_id);
          }
        }

        absl::string_view node_name =
            stats.IsTfOp() ? stats.tf_op_fullname : event.Name();
        ns->set_node_name(std::string(node_name));

        if (stats.IsKernel()) {
          absl::string_view kernel_name = event.Name();
          ns->set_timeline_label(
              absl::StrCat(kernel_name, " ", stats.kernel_details));
          DeviceStepStats*& stream_dev_stats =
              stream_dev_stats_map[{stream_id, GpuEventType::kKernel}];
          if (stream_dev_stats == nullptr) {
            stream_dev_stats = step_stats->add_dev_stats();
            stream_dev_stats->set_device(absl::StrCat(
                "/device:GPU:", device_ordinal, "/stream:", stream_id));
          }
          *stream_dev_stats->add_node_stats() = *ns;
          if (all_streams_dev_stats == nullptr) {
            all_streams_dev_stats = step_stats->add_dev_stats();
            all_streams_dev_stats->set_device(
                absl::StrCat("/device:GPU:", device_ordinal, "/stream:all"));
          }
          all_streams_dev_stats->add_node_stats()->Swap(ns.get());

        } else if (stats.IsMemCpy()) {
          absl::string_view memcpy_name = event.Name();
          ns->set_timeline_label(
              absl::StrCat(memcpy_name, " ", stats.memcpy_details));
          GpuEventType gpu_event_type = ParseMemcpyName(memcpy_name);
          DCHECK_NE(gpu_event_type, GpuEventType::kUnknown);
          DeviceStepStats*& stream_dev_stats =
              stream_dev_stats_map[{stream_id, gpu_event_type}];
          if (stream_dev_stats == nullptr) {
            stream_dev_stats = step_stats->add_dev_stats();
            stream_dev_stats->set_device(
                absl::StrCat("/device:GPU:", device_ordinal,
                             "/stream:", stream_id, "<", memcpy_name, ">"));
          }
          *stream_dev_stats->add_node_stats() = *ns;
          if (memcpy_dev_stats == nullptr) {
            memcpy_dev_stats = step_stats->add_dev_stats();
            memcpy_dev_stats->set_device(
                absl::StrCat("/device:GPU:", device_ordinal, "/memcpy"));
          }
          memcpy_dev_stats->add_node_stats()->Swap(ns.get());

        } else {
          ns->set_timeline_label(std::string(node_name));
          if (unknown_stream_dev_stats == nullptr) {
            unknown_stream_dev_stats = step_stats->add_dev_stats();
            unknown_stream_dev_stats->set_device(
                absl::StrCat("/device:GPU:", device_ordinal, "/stream:"));
          }
          unknown_stream_dev_stats->add_node_stats()->Swap(ns.get());
        }
      });
    });
  }
}

}  // namespace profiler
}  // namespace tensorflow
