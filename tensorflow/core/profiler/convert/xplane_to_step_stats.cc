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
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

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

void SetNodeTimes(const XEventVisitor& event, NodeExecStats* ns) {
  ns->set_all_start_micros(NanosToMicros(event.TimestampNs()));
  ns->set_op_start_rel_micros(0);
  ns->set_op_end_rel_micros(NanosToMicros(event.DurationNs()));
  ns->set_all_end_rel_micros(NanosToMicros(event.DurationNs()));
}

}  // namespace

void ConvertGpuXSpaceToStepStats(const XSpace& xspace, StepStats* step_stats) {
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(xspace, kGpuPlanePrefix);
  if (device_planes.empty()) {
    LOG(WARNING) << "GPU trace was not collected.";
    return;
  }
  std::vector<const XPlane*> host_planes = FindPlanesWithNames(
      xspace, {kCuptiDriverApiPlaneName, kRoctracerApiPlaneName});
  DCHECK_LE(host_planes.size(), 1);

  absl::flat_hash_map<int64_t /*correlation_id*/, CorrelationInfo>
      correlation_info_map;
  for (const XPlane* host_plane : host_planes) {
    absl::flat_hash_map<uint32_t /*device_id*/, DeviceStepStats*>
        sync_dev_stats_map;
    XPlaneVisitor plane = CreateTfXPlaneVisitor(host_plane);
    plane.ForEachLine([&](const XLineVisitor& line) {
      uint32_t thread_id = line.Id();
      line.ForEachEvent([&](const XEventVisitor& event) {
        if (event.Name() == "cuStreamSynchronize") {
          auto device_id_stat = event.GetStat(StatType::kDeviceId);
          if (device_id_stat.has_value()) {
            uint32_t device_ordinal = device_id_stat->IntOrUintValue();
            DeviceStepStats* sync_dev_stats =
                sync_dev_stats_map[device_ordinal];
            if (sync_dev_stats == nullptr) {
              sync_dev_stats = step_stats->add_dev_stats();
              sync_dev_stats->set_device(
                  absl::StrCat("/device:GPU:", device_ordinal, "/sync"));
            }
            NodeExecStats* ns = sync_dev_stats->add_node_stats();
            SetNodeTimes(event, ns);
            ns->set_node_name(std::string(event.Name()));
            ns->set_timeline_label(absl::StrCat("ThreadId ", thread_id));
            ns->set_thread_id(thread_id);
          }
        } else {
          auto correlation_id_stat = event.GetStat(StatType::kCorrelationId);
          if (correlation_id_stat.has_value()) {
            int64_t correlation_id = correlation_id_stat->IntValue();
            uint64_t enqueue_time_ns = event.TimestampNs();
            correlation_info_map[correlation_id] = {thread_id, enqueue_time_ns};
          }
        }
      });
    });
  }
  for (const XPlane* device_plane : device_planes) {
    absl::flat_hash_map<std::pair<int64_t /*stream_id*/, GpuEventType>,
                        DeviceStepStats*>
        stream_dev_stats_map;
    DeviceStepStats* unknown_stream_dev_stats = nullptr;
    DeviceStepStats* all_streams_dev_stats = nullptr;
    DeviceStepStats* memcpy_dev_stats = nullptr;
    XPlaneVisitor plane = CreateTfXPlaneVisitor(device_plane);
    uint32_t device_ordinal = plane.Id();
    plane.ForEachLine([&](const XLineVisitor& line) {
      uint32_t stream_id = line.Id();
      line.ForEachEvent([&](const XEventVisitor& event) {
        int64_t correlation_id = -1;
        absl::string_view tf_op_fullname;
        absl::string_view kernel_details;
        absl::string_view memcpy_details;
        event.ForEachStat([&](const XStatVisitor& stat) {
          if (!stat.Type().has_value()) return;
          switch (stat.Type().value()) {
            case StatType::kCorrelationId:
              correlation_id = stat.IntValue();
              break;
            case StatType::kTfOp:
              tf_op_fullname = stat.StrOrRefValue();
              break;
            case StatType::kKernelDetails:
              kernel_details = stat.StrOrRefValue();
              break;
            case StatType::kMemcpyDetails:
              memcpy_details = stat.StrOrRefValue();
              break;
            default:
              break;
          }
        });

        auto ns = absl::make_unique<NodeExecStats>();
        SetNodeTimes(event, ns.get());

        // Get launch information if available.
        if (correlation_id > 0) {
          auto it = correlation_info_map.find(correlation_id);
          if (it != correlation_info_map.end()) {
            const CorrelationInfo& correlation_info = it->second;
            ns->set_scheduled_micros(
                NanosToMicros(correlation_info.enqueue_time_ns));
            ns->set_thread_id(correlation_info.thread_id);
          }
        }

        absl::string_view node_name =
            !tf_op_fullname.empty() ? tf_op_fullname : event.Name();
        ns->set_node_name(std::string(node_name));

        if (!kernel_details.empty()) {
          absl::string_view kernel_name = event.Name();
          ns->set_timeline_label(
              absl::StrCat(kernel_name, " ", kernel_details));
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

        } else if (!memcpy_details.empty()) {
          absl::string_view memcpy_name = event.Name();
          ns->set_timeline_label(
              absl::StrCat(memcpy_name, " ", memcpy_details));
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
