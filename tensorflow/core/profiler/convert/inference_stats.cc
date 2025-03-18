/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/inference_stats.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/macros.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/protobuf/inference_stats.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::EventType;
using ::tensorflow::profiler::EventTypeSpan;
using ::tensorflow::profiler::StepEvents;
using ::tensorflow::profiler::ToNonOverlappedEvents;
using ::tsl::profiler::CreateTfXPlaneVisitor;
using ::tsl::profiler::DeviceType;
using ::tsl::profiler::GroupMetadata;
using ::tsl::profiler::GroupMetadataMap;
using ::tsl::profiler::HostEventType;
using ::tsl::profiler::StatType;
using ::tsl::profiler::Timespan;
using ::tsl::profiler::XEventVisitor;
using ::tsl::profiler::XLineVisitor;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneVisitor;
using ::tsl::profiler::XSpace;
using ::tsl::profiler::XStatVisitor;

using EventsByType =
    absl::flat_hash_map<int64_t /*event_type*/, std::vector<XEventVisitor>>;

// Holds all the events within a user facing request.
// A user facing request can be a Session.Run without batching, or a
// BatchingSession.Run with Batching, or a Session.Run with
// BatchingFunctionOp.
struct RequestEvents {
  // Index to the model id.
  int32_t model_id_index;
  // The timespan of the entire request(including both host and device).
  Timespan request_timespan;
  // The latency between a request is scheduled and is processed in a batch.
  int64_t batching_request_delay_ps;
  // Size of a request in batching mode.
  int32_t batching_request_size;

  // Timestamps of the events used for the detailed execution time breakdown.
  struct EventTimestamps {
    std::optional<int64_t> ts_batch_schedule;
    std::optional<int64_t> ts_batch_concat_input;
    std::optional<int64_t> ts_tpu_execute;
    std::optional<int64_t> ts_tpu_program_launch;
    std::optional<int64_t> ts_tpu_complete_callback;
  };
  // Mapping from group ID to the timestamps, there can be multiple group IDs
  // in a single request, because if request splitting is enabled, one request
  // can be split to multiple batches for execution, and each batch has
  // different group ID.
  absl::flat_hash_map<int64_t, EventTimestamps> timestamps;

  // The events that record tensor details like shape, type and layout.
  std::vector<const XEventVisitor*> tensor_events;
  // The final tensor details in proto format.
  std::vector<tensorflow::profiler::TensorEventDetail>
      tensor_event_detail_protos;

  // The batch ids related to this request.
  std::vector<int64_t> related_batch_ids;
  // All the events.
  std::vector<EventTypeSpan> events;
};

// Helper functions to handle absl::optional
void MinOfOptional(std::optional<int64_t>& min, std::optional<int64_t> value) {
  if (!min.has_value())
    min = value;
  else
    min = std::min(min, value);
}
void MaxOfOptional(std::optional<int64_t>& max, std::optional<int64_t> value) {
  if (!max.has_value())
    max = value;
  else
    max = std::max(max, value);
}

// Helper functions to set timestamps in RequestEvents.
void UpdateTsBatchSchedule(int64_t group_id, int64_t value,
                           RequestEvents* events) {
  events->timestamps[group_id].ts_batch_schedule = value;
}
void UpdateTsBatchConcatInput(int64_t group_id, int64_t value,
                              RequestEvents* events) {
  events->timestamps[group_id].ts_batch_concat_input = value;
}
void UpdateTsTPUExecute(int64_t group_id, int64_t value,
                        RequestEvents* events) {
  events->timestamps[group_id].ts_tpu_execute = value;
}
void UpdateTsTPUProgramLaunch(int64_t group_id, int64_t value,
                              RequestEvents* events) {
  // There might be multiple TPUProgramLaunch events in a single request.
  // Set ts_tpu_program_launch to the earlist timestamp.
  MinOfOptional(events->timestamps[group_id].ts_tpu_program_launch, value);
}
void UpdateTsTPUCompleteCallback(int64_t group_id, int64_t value,
                                 RequestEvents* events) {
  events->timestamps[group_id].ts_tpu_complete_callback = value;
}

// Map from the ID of a request to its events.
using RequestEventsMap =
    absl::flat_hash_map<int64_t /*request_id*/, RequestEvents>;

// An internal data structure that holds all the events within a batch.
struct BatchEvents {
  // The events that record tensor details like shape, type and layout.
  std::vector<const XEventVisitor*> tensor_events;

  // The BatchDetail proto.
  tensorflow::profiler::BatchDetail batch_detail_proto;
};

// Map from the ID of a batch to its events.
using BatchEventsMap = absl::flat_hash_map<int64_t /*batch_id*/, BatchEvents>;

// Map from the ID of a request to its model ID.
using ModelIdMap = absl::flat_hash_map<int64_t, std::string>;

int32_t AssignIndexToModelId(
    const std::string& model_id,
    tensorflow::profiler::ModelIdDatabase* model_id_db) {
  if (model_id.empty()) return -1;
  auto [iter, inserted] = model_id_db->mutable_id_to_index()->insert(
      {model_id, model_id_db->ids_size()});
  if (inserted) {
    model_id_db->add_ids(model_id);
  }
  return iter->second;
}

// Updates timestamps in RequestEvents.
// <function> is the timestamp to update, <value> is the updated value.
void UpdateEventTimestamps(
    const GroupMetadataMap& group_metadata_map, int64_t group_id, int64_t value,
    std::function<void(int64_t, int64_t, RequestEvents*)> function,
    RequestEventsMap* request_events_map) {
  // Update RequestEvents that are directly associated with <group_id>.
  if (auto request_events = gtl::FindOrNull(*request_events_map, group_id)) {
    function(group_id, value, request_events);
  }

  // Update all the parent RequestEvents of <group_id>.
  const GroupMetadata* group_metadata =
      gtl::FindOrNull(group_metadata_map, group_id);
  if (!group_metadata) return;
  for (const int64_t parent_group_id : group_metadata->parents) {
    if (auto parent_request_events =
            gtl::FindOrNull(*request_events_map, parent_group_id)) {
      // Update parent events, but still use <group_id> instead of
      // <parent_group_id>, because xprof needs to track where these event
      // timestamps originally come from.
      function(group_id, value, parent_request_events);
    }
  }
}

// Updates RequestEvents using ReadFromDevice, WriteToDevice and DeviceRun.
void UpdateRequestEvents(const GroupMetadataMap& group_metadata_map,
                         absl::Span<const EventTypeSpan> events,
                         int64_t group_id,
                         RequestEventsMap* request_events_map) {
  // Update RequestEvents that are directly associated with <group_id>.
  if (auto request_events = gtl::FindOrNull(*request_events_map, group_id)) {
    request_events->events.insert(request_events->events.end(), events.begin(),
                                  events.end());
  }

  // Update all the parent RequestEvents of <group_id> with the same
  // <event_type> and <time_span>. Parent RequestEvents are all the requests
  // in a batch.
  const GroupMetadata* group_metadata =
      gtl::FindOrNull(group_metadata_map, group_id);
  if (!group_metadata) return;
  for (const int64_t parent_group_id : group_metadata->parents) {
    if (auto parent_request_events =
            gtl::FindOrNull(*request_events_map, parent_group_id)) {
      parent_request_events->events.insert(parent_request_events->events.end(),
                                           events.begin(), events.end());
    }
  }
}

// Initializes RequestEvents.
// <is_batching_request> determines whether this event is a
// BatchingSession.Run
void InitializeRequestEvents(
    const XEventVisitor& event, const GroupMetadataMap& group_metadata_map,
    const absl::flat_hash_set<int64_t>& process_batch_group_ids,
    const ModelIdMap& model_id_map, bool is_batching_request,
    bool is_user_defined_request,
    tensorflow::profiler::ModelIdDatabase* model_id_db,
    RequestEventsMap* request_events_map) {
  std::optional<XStatVisitor> optional_group_id =
      event.GetStat(StatType::kGroupId);
  if (!optional_group_id.has_value()) return;
  int64_t group_id = optional_group_id->IntValue();

  // If the event has ProcessBatch event as a parent, then do not consider
  // it as a request.
  if (process_batch_group_ids.contains(group_id)) return;

  RequestEvents& request_events = (*request_events_map)[group_id];
  const GroupMetadata* group_metadata =
      gtl::FindOrNull(group_metadata_map, group_id);
  if (!group_metadata) return;
  // The children group_ids of a request are the batches related to this
  // request.
  for (const int64_t child_group_id : group_metadata->children) {
    request_events.related_batch_ids.push_back(child_group_id);
  }
  // Sort related_batch_ids to get deterministic result.
  absl::c_sort(request_events.related_batch_ids);
  if (is_batching_request) {
    // The children events of BatchingSession.Run are multiple Session.Run,
    // use the first child event to initialize ModelId information, because
    // all the children events should have the same ModelId.
    if (group_metadata->children.empty()) return;
    int64_t children_group_id = *group_metadata->children.begin();
    const std::string* children_model_id =
        gtl::FindOrNull(model_id_map, children_group_id);
    request_events.model_id_index = AssignIndexToModelId(
        children_model_id ? *children_model_id : "", model_id_db);
  } else if (is_user_defined_request) {
    const std::string* model_id = gtl::FindOrNull(model_id_map, group_id);
    if (model_id) {
      request_events.model_id_index =
          AssignIndexToModelId(*model_id, model_id_db);
    } else {
      // In some cases (e.g., BrainServer::Estimate), a single request might
      // dispatch batches for multiple models. If all children events
      // have the same ModelId, we assign that ModelId to the request.
      if (group_metadata->children.empty()) return;
      int32_t model_id_index_for_all_children = -1;
      bool all_children_have_same_model_id = true;
      for (int64_t children_group_id : group_metadata->children) {
        const std::string* children_model_id =
            gtl::FindOrNull(model_id_map, children_group_id);
        int32_t child_model_id_index = AssignIndexToModelId(
            children_model_id ? *children_model_id : "", model_id_db);
        if (model_id_index_for_all_children == -1) {
          model_id_index_for_all_children = child_model_id_index;
        } else if (child_model_id_index != model_id_index_for_all_children) {
          all_children_have_same_model_id = false;
        }
      }
      request_events.model_id_index =
          all_children_have_same_model_id
              ? model_id_index_for_all_children
              : AssignIndexToModelId("", model_id_db);
    }
  } else {
    const std::string* model_id = gtl::FindOrNull(model_id_map, group_id);
    request_events.model_id_index =
        AssignIndexToModelId(model_id ? *model_id : "", model_id_db);
  }
}

// Set the begin and end timestamp of the request.
// The timespan of the request is marked by the earliest timestamp and latest
// timestamp of the events with the same group_id.
void UpdateRequestTimespan(const EventsByType& host_events_by_type,
                           RequestEventsMap* request_events_map) {
  for (const auto& [_, events] : host_events_by_type) {
    for (const auto& event : events) {
      auto optional_group_id = event.GetStat(StatType::kGroupId);
      if (optional_group_id.has_value()) {
        if (RequestEvents* request = gtl::FindOrNull(
                *request_events_map, optional_group_id->IntValue())) {
          auto begin_ps = request->request_timespan.begin_ps() == 0
                              ? event.GetTimespan().begin_ps()
                              : std::min(request->request_timespan.begin_ps(),
                                         event.GetTimespan().begin_ps());
          auto end_ps = std::max(request->request_timespan.end_ps(),
                                 event.GetTimespan().end_ps());
          request->request_timespan = Timespan::FromEndPoints(begin_ps, end_ps);
        }
      }
    }
  }
}

// Update RequestEventsMap using data transfer events in tpu::system.
// Each data transfer is associated with a start event, an end event, and a
// transfer type (H2D or D2H).
void UpdateTpuDataTransferEventsInTpuSystem(
    const EventsByType& host_events_by_type,
    const GroupMetadataMap& group_metadata_map,
    const HostEventType data_transfer_start_event,
    const HostEventType data_transfer_end_event,
    const EventType data_transfer_type, RequestEventsMap* request_events_map) {
  absl::flat_hash_map<uint64_t, std::array<const XEventVisitor*, 2>>
      events_per_transfer;

  auto build_events =
      [&](const HostEventType event_type,
          std::function<void(uint64_t, const XEventVisitor*)> func) {
        if (const auto* events =
                gtl::FindOrNull(host_events_by_type, event_type)) {
          for (const XEventVisitor& event : *events) {
            std::optional<XStatVisitor> optional_group_id =
                event.GetStat(StatType::kGroupId);
            if (!optional_group_id.has_value()) continue;
            std::optional<XStatVisitor> context_id =
                event.GetStat(StatType::kConsumerId);
            if (!context_id.has_value()) continue;
            func(context_id->IntValue(), &event);
          }
        }
      };

  // Build start event.
  build_events(data_transfer_start_event,
               [&](uint64_t id, const XEventVisitor* start_event) {
                 events_per_transfer[id] = {start_event, nullptr};
               });

  // Build end event.
  // This only happens when the start event exists, the end event has the same
  // group ID as the start event, and the end event timestamp is larger than
  // start event timestamp.
  build_events(data_transfer_end_event,
               [&](uint64_t id, const XEventVisitor* end_event) {
                 if (auto* value = gtl::FindOrNull(events_per_transfer, id)) {
                   const XEventVisitor* start_event = value->at(0);
                   if (start_event->TimestampPs() < end_event->TimestampPs()) {
                     value->at(1) = end_event;
                   }
                 }
               });

  std::vector<EventTypeSpan> event_to_update = {
      {data_transfer_type, Timespan(0, 0)}};
  for (const auto& [id, events] : events_per_transfer) {
    if (events[0] != nullptr && events[1] != nullptr) {
      // Duration of the data transfer is measured as the timespan between
      // start and end events.
      event_to_update[0].span =
          Timespan(events[0]->TimestampPs(),
                   events[1]->EndTimestampPs() - events[0]->TimestampPs());
      UpdateRequestEvents(group_metadata_map, event_to_update,
                          events[0]->GetStat(StatType::kGroupId)->IntValue(),
                          request_events_map);
    }
  }
}

// Initializes device side events for TPU.
void BuildTPUDeviceEvents(const std::vector<XPlane*>& device_traces,
                          const EventsByType& host_events_by_type,
                          const GroupMetadataMap& group_metadata_map,
                          RequestEventsMap* request_events_map) {
  static constexpr int64_t kDataTransferTypes[] = {
      HostEventType::kReadHbm, HostEventType::kTransferD2HRequest,
      HostEventType::kWriteHbm, HostEventType::kTransferH2DRequest,
      HostEventType::kTransferPreprocessedH2DRequest};
  auto data_transfer_type_to_enum = [](const int64_t type) {
    switch (type) {
      case HostEventType::kReadHbm:
      case HostEventType::kTransferD2HRequest:
        return EventType::DEVICE_TO_HOST;
      case HostEventType::kWriteHbm:
      case HostEventType::kTransferH2DRequest:
      case HostEventType::kTransferPreprocessedH2DRequest:
        return EventType::HOST_TO_DEVICE;
      default:
        return EventType::UNKNOWN_TIME;
    }
  };

  // Initialize a TPU device event for future updates.
  // In order to reuse the same UpdateRequestEvents function with GPU device
  // events, here we create a vector of size 1 for TPU event.
  std::vector<EventTypeSpan> event_to_update = {
      {EventType::UNKNOWN_TIME, Timespan(0, 0)}};

  // Update RequestEventsMap using data transfer events.
  for (const int64_t data_transfer_type : kDataTransferTypes) {
    if (const auto* data_transfer_events =
            gtl::FindOrNull(host_events_by_type, data_transfer_type)) {
      for (const XEventVisitor& data_transfer_event : *data_transfer_events) {
        std::optional<XStatVisitor> optional_group_id =
            data_transfer_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        event_to_update[0] = {data_transfer_type_to_enum(data_transfer_type),
                              data_transfer_event.GetTimespan()};
        UpdateRequestEvents(group_metadata_map, event_to_update, group_id,
                            request_events_map);
      }
    }
  }

  UpdateTpuDataTransferEventsInTpuSystem(
      host_events_by_type, group_metadata_map,
      HostEventType::kTransferToDeviceIssueEvent,
      HostEventType::kTransferToDeviceDone, EventType::HOST_TO_DEVICE,
      request_events_map);

  UpdateTpuDataTransferEventsInTpuSystem(
      host_events_by_type, group_metadata_map,
      HostEventType::kTransferFromDeviceIssueEvent,
      HostEventType::kTransferFromDeviceDone, EventType::DEVICE_TO_HOST,
      request_events_map);

  for (const XPlane* device_trace : device_traces) {
    XPlaneVisitor device_plane = CreateTfXPlaneVisitor(device_trace);
    device_plane.ForEachLine([request_events_map, &event_to_update,
                              &group_metadata_map](const XLineVisitor& line) {
      if (line.Name() != tsl::profiler::kXlaModuleLineName) return;
      line.ForEachEvent([request_events_map, &event_to_update,
                         &group_metadata_map](const XEventVisitor& event) {
        std::optional<XStatVisitor> group_id =
            event.GetStat(StatType::kGroupId);
        if (!group_id) return;
        // TPU compute does not specify 32bit or 16bit, use
        // DEVICE_COMPUTE_32 to annotate this is a compute event.
        event_to_update[0] = {EventType::DEVICE_COMPUTE_32,
                              event.GetTimespan()};
        UpdateRequestEvents(group_metadata_map, event_to_update,
                            group_id->IntValue(), request_events_map);
      });
    });
  }

  // Update timestamp for TPU execute event. It is used as the beginning of
  // TPU runtime. For old TPU runtime, it is the TPUPartitionedCall events,
  // for the new TPU runtime, it is the tpu::system::Execute event. There
  // might be multiple TPU execute events in the same request,
  // UpdateTsTPUExecute is implemented as getting the earlist timestamp of TPU
  // execute event.
  static constexpr int64_t kTPUExecuteTypes[] = {
      HostEventType::kTpuPartitionedCallOpExecuteLocal,
      HostEventType::kTpuPartitionedCallOpExecuteRemote,
      HostEventType::kTpuPartitionedCallOpInitializeVarOnTpu,
      HostEventType::kTpuSystemExecute};
  for (const int64_t tpu_execute_type : kTPUExecuteTypes) {
    if (const auto* tpu_execute_events =
            gtl::FindOrNull(host_events_by_type, tpu_execute_type)) {
      for (const XEventVisitor& tpu_execute_event : *tpu_execute_events) {
        std::optional<XStatVisitor> optional_group_id =
            tpu_execute_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        UpdateEventTimestamps(group_metadata_map, group_id,
                              tpu_execute_event.TimestampPs(),
                              UpdateTsTPUExecute, request_events_map);
      }
    }
  }

  // Update timestamp for TPU program launch events. This is used as the end
  // of TPU runtime. Only one of the following program launch events will
  // appear in a single profile.
  static constexpr int64_t kTPUProgramLaunchTypes[] = {
      HostEventType::kDoEnqueueProgram,
      HostEventType::kDoEnqueueContinuationProgram};
  for (const int64_t tpu_program_launch_type : kTPUProgramLaunchTypes) {
    if (const auto* tpu_program_launch_events =
            gtl::FindOrNull(host_events_by_type, tpu_program_launch_type)) {
      for (const XEventVisitor& tpu_program_launch_event :
           *tpu_program_launch_events) {
        std::optional<XStatVisitor> optional_group_id =
            tpu_program_launch_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        UpdateEventTimestamps(group_metadata_map, group_id,
                              tpu_program_launch_event.TimestampPs(),
                              UpdateTsTPUProgramLaunch, request_events_map);
      }
    }
  }

  // Update timestamp for TPU complete callbacks. This is used as the start of
  // host postprocessing.
  if (const auto* tpu_complete_callback_events = gtl::FindOrNull(
          host_events_by_type, HostEventType::kCompleteCallbacks)) {
    for (const XEventVisitor& tpu_complete_callback_event :
         *tpu_complete_callback_events) {
      std::optional<XStatVisitor> optional_group_id =
          tpu_complete_callback_event.GetStat(StatType::kGroupId);
      if (!optional_group_id.has_value()) continue;
      int64_t group_id = optional_group_id->IntValue();
      UpdateEventTimestamps(group_metadata_map, group_id,
                            tpu_complete_callback_event.TimestampPs(),
                            UpdateTsTPUCompleteCallback, request_events_map);
    }
  }
}

// Initializes device side events for GPU.
void BuildGPUDeviceEvents(const StepEvents& nonoverlapped_step_events,
                          const GroupMetadataMap& group_metadata_map,
                          RequestEventsMap* request_events_map) {
  for (const auto& [step_id, step_details] : nonoverlapped_step_events) {
    UpdateRequestEvents(group_metadata_map, step_details.Events(), step_id,
                        request_events_map);
  }
}

// Initialize the mapping from group_id to model_id. Skip the event if it
// doesn't have group_id or model_id.
ModelIdMap InitializeModelIdMap(
    const EventsByType& host_events_by_type,
    const std::vector<const XEventVisitor*>& user_defined_root_events) {
  ModelIdMap model_id_map;

  // Helper function to process model id.
  auto process_model_id = [&](const XEventVisitor& event) {
    auto group_id = event.GetStat(StatType::kGroupId);
    if (!group_id.has_value()) return;
    std::optional<XStatVisitor> model_id = event.GetStat(StatType::kModelId);
    if (!model_id.has_value()) return;
    model_id_map[group_id->IntValue()] = model_id->ToString();
  };

  static constexpr int64_t kModelIdRequestTypes[] = {
      HostEventType::kSessionRun, HostEventType::kTfrtModelRun,
      HostEventType::kServingModelRun};
  for (const int64_t event_type : kModelIdRequestTypes) {
    auto event_list = gtl::FindOrNull(host_events_by_type, event_type);
    if (!event_list) continue;
    for (const XEventVisitor& event : *event_list) {
      process_model_id(event);
    }
  }

  for (const XEventVisitor* event : user_defined_root_events) {
    process_model_id(*event);
  }

  return model_id_map;
}

// Builds a request_events_map from the given trace events.
void BuildRequestEventsMap(const std::vector<XPlane*>& device_traces,
                           const EventsByType& host_events_by_type,
                           const GroupMetadataMap& group_metadata_map,
                           const StepEvents& nonoverlapped_step_events,
                           DeviceType device_type,
                           tensorflow::profiler::ModelIdDatabase* model_id_db,
                           RequestEventsMap* request_events_map) {
  static constexpr int64_t kBatchingRequestTypes[] = {
      HostEventType::kBatchingSessionRun};
  static constexpr int64_t kNonBatchingRequestTypes[] = {
      HostEventType::kSessionRun, HostEventType::kRunGraph};
  // TODO(wffw): Merge them once go/pathways-tfrt-serving-unification is done.
  static constexpr int64_t kTfrtRequestTypes[] = {HostEventType::kTfrtModelRun};
  static constexpr int64_t kPathwayRequestTypes[] = {
      HostEventType::kServingModelRun};

  static constexpr int64_t kScheduleEventTypes[] = {
      HostEventType::kScheduleWithSplit, HostEventType::kScheduleWithoutSplit,
      HostEventType::kScheduleWithEagerSplit,
      HostEventType::kASBSQueueSchedule};

  // Events marked with "_r:-1" are user defined root events.
  std::vector<const XEventVisitor*> user_defined_root_events;
  for (const auto& [_, events] : host_events_by_type) {
    for (const auto& event : events) {
      std::optional<XStatVisitor> stat = event.GetStat(StatType::kIsRoot);
      if (stat.has_value() && stat->IntValue() == -1) {
        user_defined_root_events.push_back(&event);
      }
    }
  }

  // Group IDs of ProcessBatch events.
  absl::flat_hash_set<int64_t> process_batch_group_ids;
  if (const auto* process_batch_events =
          gtl::FindOrNull(host_events_by_type, HostEventType::kProcessBatch)) {
    for (const XEventVisitor& process_batch_event : *process_batch_events) {
      std::optional<XStatVisitor> optional_group_id =
          process_batch_event.GetStat(StatType::kGroupId);
      if (!optional_group_id.has_value()) continue;
      process_batch_group_ids.insert(optional_group_id->IntValue());
    }
  }

  ModelIdMap model_id_map =
      InitializeModelIdMap(host_events_by_type, user_defined_root_events);

  // Initialize RequestEventsMap.
  bool is_batching_request =
      host_events_by_type.contains(HostEventType::kBatchingSessionRun);
  bool is_tfrt_request =
      host_events_by_type.contains(HostEventType::kTfrtModelRun);
  // TODO(wffw): Merge them once go/pathways-tfrt-serving-unification is done.
  bool is_pathway_request =
      host_events_by_type.contains(HostEventType::kServingModelRun);
  absl::Span<const int64_t> request_types;
  if (is_batching_request) {
    request_types = absl::Span<const int64_t>(kBatchingRequestTypes);
  } else if (is_tfrt_request) {
    request_types = absl::Span<const int64_t>(kTfrtRequestTypes);
  } else if (is_pathway_request) {
    request_types = absl::Span<const int64_t>(kPathwayRequestTypes);
  } else {
    request_types = absl::Span<const int64_t>(kNonBatchingRequestTypes);
  }
  for (const int64_t request_type : request_types) {
    if (const auto* request_events =
            gtl::FindOrNull(host_events_by_type, request_type)) {
      for (const XEventVisitor& request_event : *request_events) {
        InitializeRequestEvents(request_event, group_metadata_map,
                                process_batch_group_ids, model_id_map,
                                is_batching_request,
                                /* is_user_defined_request=*/false, model_id_db,
                                request_events_map);
      }
    }
  }

  for (const XEventVisitor* event : user_defined_root_events) {
    InitializeRequestEvents(
        *event, group_metadata_map, process_batch_group_ids, model_id_map,
        /*is_batching_request=*/false,
        /* is_user_defined_request=*/true, model_id_db, request_events_map);
  }

  // Set the begin and end timestamp of the request.
  UpdateRequestTimespan(host_events_by_type, request_events_map);

  // Update RequestEventsMap using the request size in schedule event.
  for (const int64_t schedule_type : kScheduleEventTypes) {
    if (const auto* schedule_events =
            gtl::FindOrNull(host_events_by_type, schedule_type)) {
      for (const XEventVisitor& schedule_event : *schedule_events) {
        std::optional<XStatVisitor> optional_group_id =
            schedule_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        // Update timestamp for schedule events. It is used as the beginning
        // of batch formation.
        UpdateEventTimestamps(group_metadata_map, group_id,
                              schedule_event.TimestampPs(),
                              UpdateTsBatchSchedule, request_events_map);
        if (auto* request_events =
                gtl::FindOrNull(*request_events_map, group_id)) {
          std::optional<XStatVisitor> batching_request_size =
              schedule_event.GetStat(StatType::kBatchingInputTaskSize);
          if (!batching_request_size.has_value()) continue;
          request_events->batching_request_size =
              batching_request_size->IntValue();
        }
      }
    }
  }

  if (device_type == DeviceType::kTpu) {
    BuildTPUDeviceEvents(device_traces, host_events_by_type, group_metadata_map,
                         request_events_map);
  } else if (device_type == DeviceType::kGpu) {
    BuildGPUDeviceEvents(nonoverlapped_step_events, group_metadata_map,
                         request_events_map);
  }
}

// Extracts batch details from <event_forest>.
void BuildBatchEventsMap(const EventsByType& host_events_by_type,
                         const GroupMetadataMap& group_metadata_map,
                         RequestEventsMap* request_events_map,
                         BatchEventsMap* batch_events_map) {
  // Initialize BatchDetails from ProcessBatch events.
  if (const auto* process_batch_events =
          gtl::FindOrNull(host_events_by_type, HostEventType::kProcessBatch)) {
    for (const XEventVisitor& process_batch_event : *process_batch_events) {
      std::optional<XStatVisitor> optional_group_id =
          process_batch_event.GetStat(StatType::kGroupId);
      if (!optional_group_id.has_value()) continue;
      int64_t group_id = optional_group_id->IntValue();
      const GroupMetadata* group_metadata =
          gtl::FindOrNull(group_metadata_map, group_id);
      if (!group_metadata) continue;
      BatchEvents& batch_events = (*batch_events_map)[group_id];
      tensorflow::profiler::BatchDetail& batch_detail =
          batch_events.batch_detail_proto;
      batch_detail.set_batch_id(group_id);
      batch_detail.set_start_time_ps(process_batch_event.TimestampPs());
      batch_detail.set_end_time_ps(process_batch_event.EndTimestampPs());
      // The parent group_ids of a batch are the requests related to this
      // batch.
      for (const int64_t parent_group_id : group_metadata->parents) {
        batch_detail.add_related_request_ids(parent_group_id);
      }
      // Sort related_request_ids to get deterministic result.
      std::sort(batch_detail.mutable_related_request_ids()->begin(),
                batch_detail.mutable_related_request_ids()->end());
    }
  }

  // Update BatchDetailsMap with padding information. Only one of
  // ConcatInputTensors (for in-graph batching) or MergeInputTensors (for
  // BatchingSession), or BrainSessionRun will appear in the
  // same profile.
  static constexpr int64_t kPaddingEventTypes[] = {
      HostEventType::kConcatInputTensors,
      HostEventType::kMergeInputTensors,
      HostEventType::kBrainSessionRun,
  };
  for (const int64_t padding_event_type : kPaddingEventTypes) {
    if (const auto* padding_events =
            gtl::FindOrNull(host_events_by_type, padding_event_type)) {
      for (const XEventVisitor& padding_event : *padding_events) {
        // Update timestamp for padding events. They are used as the
        // beginning of batch processing.
        std::optional<XStatVisitor> optional_group_id =
            padding_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        UpdateEventTimestamps(group_metadata_map, group_id,
                              padding_event.TimestampPs(),
                              UpdateTsBatchConcatInput, request_events_map);
        BatchEvents* batch_events =
            gtl::FindOrNull(*batch_events_map, group_id);
        if (!batch_events) continue;
        std::optional<XStatVisitor> padding_amount =
            padding_event.GetStat(StatType::kPaddingAmount);
        if (!padding_amount.has_value()) continue;
        std::optional<XStatVisitor> batch_size_after_padding =
            padding_event.GetStat(StatType::kBatchSizeAfterPadding);
        if (!batch_size_after_padding.has_value()) continue;
        tensorflow::profiler::BatchDetail* batch_detail =
            &batch_events->batch_detail_proto;
        batch_detail->set_batch_size_after_padding(
            batch_size_after_padding->IntValue());
        batch_detail->set_padding_amount(padding_amount->IntValue());
      }
    }
  }

  // Populate BatchDetailsMap with model_id information from the corresponding
  // requests in RequestEventsMap.
  for (auto& [batch_id, batch_events] : *batch_events_map) {
    tensorflow::profiler::BatchDetail& batch_detail =
        batch_events.batch_detail_proto;
    if (!batch_detail.related_request_ids().empty()) {
      // Set the model_id of a batch using the model_id of the corresponding
      // request. All requests in the same batch must share the same model_id,
      // so we can pick any request in the batch here.
      int32_t first_request_id = batch_detail.related_request_ids(0);
      const RequestEvents* request_events =
          gtl::FindOrNull(*request_events_map, first_request_id);
      if (request_events) {
        batch_detail.set_model_id_index(request_events->model_id_index);
      }
    }
  }
}

// Calculates the delay between request and batch.
void GenerateRequestAndBatchDelay(RequestEventsMap* request_events_map,
                                  BatchEventsMap* batch_events_map) {
  for (auto& [request_id, request_event] : *request_events_map) {
    const tensorflow::profiler::BatchDetail* first_batch_detail = nullptr;
    const tensorflow::profiler::BatchDetail* last_batch_detail = nullptr;
    // For each request, measure the latency between the request and the first
    // batch that processes this request.
    for (const int64_t batch_id : request_event.related_batch_ids) {
      const auto* batch_events = gtl::FindOrNull(*batch_events_map, batch_id);
      if (!batch_events) continue;
      const tensorflow::profiler::BatchDetail* batch_detail =
          &batch_events->batch_detail_proto;
      if (!first_batch_detail || (first_batch_detail->has_start_time_ps() >
                                  batch_detail->has_start_time_ps())) {
        first_batch_detail = batch_detail;
      }
      if (!last_batch_detail || (last_batch_detail->has_end_time_ps() <
                                 batch_detail->has_end_time_ps())) {
        last_batch_detail = batch_detail;
      }
    }
    if (first_batch_detail) {
      request_event.batching_request_delay_ps =
          first_batch_detail->start_time_ps() -
          request_event.request_timespan.begin_ps();
    }
    if (last_batch_detail && request_event.request_timespan.end_ps() <
                                 last_batch_detail->end_time_ps()) {
      request_event.request_timespan =
          Timespan::FromEndPoints(request_event.request_timespan.begin_ps(),
                                  last_batch_detail->end_time_ps());
    }
  }

  for (auto& [batch_id, batch_events] : *batch_events_map) {
    const RequestEvents* first_request_events = nullptr;
    tensorflow::profiler::BatchDetail& batch_detail =
        batch_events.batch_detail_proto;
    // For each batch, measure the latency between the first request in this
    // batch and the start time of this batch.
    for (const int64_t request_id : batch_detail.related_request_ids()) {
      const auto* request_events =
          gtl::FindOrNull(*request_events_map, request_id);
      if (!request_events) continue;
      if (!first_request_events ||
          (first_request_events->request_timespan.begin_ps() >
           request_events->request_timespan.begin_ps())) {
        first_request_events = request_events;
      }
    }
    if (first_request_events) {
      batch_detail.set_batch_delay_ps(
          batch_detail.start_time_ps() -
          first_request_events->request_timespan.begin_ps());
    }
  }
}

// Generates detailed breakdown for a request by generating events using the
// timestamps in RequestEvents.
void GenerateRequestDetailedBreakdown(RequestEventsMap* request_events_map) {
  for (auto& [_, request] : *request_events_map) {
    std::optional<int64_t> first_tpu_execute;
    std::optional<int64_t> first_batch_concat_input;
    std::optional<int64_t> last_tpu_complete_callback;
    std::optional<int64_t> only_batch_schedule;
    for (const auto& [group_id, timestamps] : request.timestamps) {
      if (timestamps.ts_tpu_execute.has_value()) {
        MinOfOptional(first_tpu_execute, timestamps.ts_tpu_execute);

        // Host runtime: From the start of TPU execute event to the start of
        // TPU program launch. Because of request splitting, there can be
        // multiple host runtime in a single request, one for each batch.
        if (timestamps.ts_tpu_program_launch.has_value()) {
          request.events.push_back(
              {EventType::HOST_RUNTIME,
               Timespan::FromEndPoints(
                   timestamps.ts_tpu_execute.value(),
                   timestamps.ts_tpu_program_launch.value())});
        }
      }

      if (timestamps.ts_batch_concat_input.has_value()) {
        MinOfOptional(first_batch_concat_input,
                      timestamps.ts_batch_concat_input);
      }

      if (timestamps.ts_tpu_complete_callback.has_value()) {
        MaxOfOptional(last_tpu_complete_callback,
                      timestamps.ts_tpu_complete_callback);
      }

      if (timestamps.ts_batch_schedule.has_value()) {
        if (only_batch_schedule.has_value()) {
          LOG(ERROR) << "Found multiple batch schedule events in a single "
                     << "request.";
        } else {
          only_batch_schedule = timestamps.ts_batch_schedule;
        }
      }
    }

    // Host preprocessing: From the start of the request to the start of the
    // first execute event. There is only one host preprocess even if there
    // are multiple batches caused by request splitting.
    if (first_tpu_execute.has_value()) {
      request.events.push_back(
          {EventType::HOST_PREPROCESS,
           Timespan::FromEndPoints(request.request_timespan.begin_ps(),
                                   first_tpu_execute.value())});
    }

    // Host postprocessing: If there are CompleteCallback events for this
    // request, use the last CompleteCallback event as the beginning of host
    // postprocessing. Else, use the end time of the last TPU device compute
    // events. There is only one host postprocessing even if there are
    // multiple batches caused by request splitting.
    if (last_tpu_complete_callback.has_value()) {
      request.events.push_back(
          {EventType::HOST_POSTPROCESS,
           Timespan::FromEndPoints(last_tpu_complete_callback.value(),
                                   request.request_timespan.end_ps())});
    } else {
      // Get the latest end time of TPU device compute events.
      // These events are annotated with type DEVICE_COMPUTE_32.
      // TODO(tianrun): Deprecate this code path after CompleteCallback is
      // enabled in all Tensorflow binaries.
      uint64_t device_compute_end = 0;
      for (const auto& event : request.events) {
        if (event.type == EventType::DEVICE_COMPUTE_32) {
          device_compute_end =
              std::max(device_compute_end, event.span.end_ps());
        }
      }
      if (device_compute_end != 0) {
        request.events.push_back(
            {EventType::HOST_POSTPROCESS,
             Timespan::FromEndPoints(device_compute_end,
                                     request.request_timespan.end_ps())});
      }
    }

    // Batch formation: From the start of batch schedule, to the start of the
    // first concat input. This is only applicable when batching is enabled,
    // and it overlaps with host preprocessing.
    if (only_batch_schedule.has_value() &&
        first_batch_concat_input.has_value()) {
      request.events.push_back(
          {EventType::HOST_BATCH_FORMATION,
           Timespan::FromEndPoints(only_batch_schedule.value(),
                                   first_batch_concat_input.value())});
    }
  }
}

// Generates tensor patterns from tensor related EventNodes.
// If there is any error during the generation, return an empty string.
std::string GenerateTensorPattern(
    const std::vector<const XEventVisitor*>& tensor_events) {
  // Generate one sub pattern for each tensor event, the sub pattern records
  // the tensor shape, type, and layout.
  std::vector<std::string> sub_patterns;
  sub_patterns.reserve(tensor_events.size());
  for (const XEventVisitor* tensor_event : tensor_events) {
    std::optional<XStatVisitor> shape =
        tensor_event->GetStat(StatType::kTensorShapes);
    if (!shape.has_value()) return "";
    std::optional<XStatVisitor> layout =
        tensor_event->GetStat(StatType::kTensorLayout);
    if (!layout.has_value()) return "";
    sub_patterns.push_back(absl::StrCat(tensor_event->Name(), " ",
                                        shape->StrOrRefValue(), " ",
                                        layout->StrOrRefValue()));
  }
  // Sort the sub patterns to get a deterministic result.
  std::sort(sub_patterns.begin(), sub_patterns.end());
  // The final tensor pattern is generated as the concatenation of all sub
  // patterns. Use <br> as separator so it can be displayed properly in
  // frontend.
  return absl::StrJoin(sub_patterns, "<br>");
}

// Generates the total time spent on linearize and delinearize tensors.
uint64_t GenerateTensorLinearizeDelinearizeTime(
    const std::vector<const XEventVisitor*>& tensor_events) {
  uint64_t result = 0;
  for (const XEventVisitor* tensor_event : tensor_events) {
    result += tensor_event->DurationPs();
  }
  return result;
}

// Generates the details related to tensor shape, type, and layout.
void GenerateTensorDetails(
    const EventsByType& host_events_by_type,
    RequestEventsMap* request_events_map, BatchEventsMap* batch_events_map,
    tensorflow::profiler::InferenceStats* inference_stats) {
  static constexpr int64_t kTensorDetailEventTypes[] = {
      HostEventType::kLinearize, HostEventType::kDelinearize,
      HostEventType::kTransferBufferFromDeviceFastPath};

  for (const int64_t tensor_detail_event_type : kTensorDetailEventTypes) {
    if (const auto* tensor_detail_events =
            gtl::FindOrNull(host_events_by_type, tensor_detail_event_type)) {
      for (const XEventVisitor& tensor_detail_event : *tensor_detail_events) {
        std::optional<XStatVisitor> optional_group_id =
            tensor_detail_event.GetStat(StatType::kGroupId);
        if (!optional_group_id.has_value()) continue;
        int64_t group_id = optional_group_id->IntValue();
        // Add events to corresponding requests and batches.
        if (auto* request_events =
                gtl::FindOrNull(*request_events_map, group_id)) {
          request_events->tensor_events.push_back(&tensor_detail_event);
        } else if (auto* batch_events =
                       gtl::FindOrNull(*batch_events_map, group_id)) {
          batch_events->tensor_events.push_back(&tensor_detail_event);
        }
      }
    }
  }

  absl::flat_hash_map<std::string, int> tensor_patterns;
  auto get_tensor_pattern_index =
      [&tensor_patterns](const std::string& tensor_pattern) {
        if (int* index = gtl::FindOrNull(tensor_patterns, tensor_pattern)) {
          return *index;
        }
        int index = tensor_patterns.size();
        tensor_patterns.insert(std::make_pair(tensor_pattern, index));
        return index;
      };

  // Generates the tensor details that are owned by request.
  for (auto& [group_id, request_events] : *request_events_map) {
    if (request_events.tensor_events.empty()) continue;
    std::string tensor_pattern =
        GenerateTensorPattern(request_events.tensor_events);
    if (tensor_pattern.empty()) continue;
    int index = get_tensor_pattern_index(tensor_pattern);
    tensorflow::profiler::TensorEventDetail tensor_event_detail;
    tensor_event_detail.set_tensor_pattern_index(index);
    tensor_event_detail.set_owner(
        tensorflow::profiler::TensorEventDetail::REQUEST);
    tensor_event_detail.set_linearize_delinearize_time_ps(
        GenerateTensorLinearizeDelinearizeTime(request_events.tensor_events));
    request_events.tensor_event_detail_protos.push_back(
        std::move(tensor_event_detail));
  }

  // Generates the tensor details that are owned by batch.
  for (auto& [group_id, batch_events] : *batch_events_map) {
    if (batch_events.tensor_events.empty()) continue;
    std::string tensor_pattern =
        GenerateTensorPattern(batch_events.tensor_events);
    if (tensor_pattern.empty()) continue;
    int index = get_tensor_pattern_index(tensor_pattern);
    auto* tensor_event_detail =
        batch_events.batch_detail_proto.mutable_tensor_event_detail();
    tensor_event_detail->set_tensor_pattern_index(index);
    tensor_event_detail->set_owner(
        tensorflow::profiler::TensorEventDetail::BATCH);
    tensor_event_detail->set_linearize_delinearize_time_ps(
        GenerateTensorLinearizeDelinearizeTime(batch_events.tensor_events));
  }

  // Populates the tensor details from batch to the related requests. These
  // tensor details are still owned by the batches and will not be used to
  // calculate statistics like the number of occurrence of each tensor
  // pattern.
  for (const auto& [group_id, batch_events] : *batch_events_map) {
    if (!batch_events.batch_detail_proto.has_tensor_event_detail()) continue;
    for (const int64_t request_id :
         batch_events.batch_detail_proto.related_request_ids()) {
      if (auto* request_events =
              gtl::FindOrNull(*request_events_map, request_id)) {
        request_events->tensor_event_detail_protos.push_back(
            batch_events.batch_detail_proto.tensor_event_detail());
      }
    }
  }

  // Generates TensorPatternDatabase.
  if (tensor_patterns.empty()) {
    return;
  }
  absl::flat_hash_map<int, const std::string*> reversed_tensor_patterns;
  for (const auto& tensor_pattern : tensor_patterns) {
    reversed_tensor_patterns[tensor_pattern.second] = &tensor_pattern.first;
  }
  for (int i = 0; i < static_cast<int>(tensor_patterns.size()); i++) {
    inference_stats->mutable_tensor_pattern_db()->add_tensor_pattern(
        *reversed_tensor_patterns.at(i));
  }
}

// Generates the request details proto from its events.
void RequestEventsToDetails(
    DeviceType device_type, int64_t group_id,
    const RequestEvents& request_events,
    tensorflow::profiler::RequestDetail* request_detail) {
  request_detail->set_request_id(group_id);
  request_detail->set_model_id_index(request_events.model_id_index);
  request_detail->set_start_time_ps(request_events.request_timespan.begin_ps());
  request_detail->set_end_time_ps(request_events.request_timespan.end_ps());
  request_detail->set_batching_request_delay_ps(
      request_events.batching_request_delay_ps);
  request_detail->set_batching_request_size(
      request_events.batching_request_size);
  for (const auto& tensor_event_detail :
       request_events.tensor_event_detail_protos) {
    *request_detail->add_tensor_event_details() = tensor_event_detail;
  }
  for (const int64_t batch_id : request_events.related_batch_ids) {
    request_detail->add_related_batch_ids(batch_id);
  }

  std::vector<EventTypeSpan> tpu_non_overlapped_events;
  const std::vector<EventTypeSpan>* non_overlapped_events =
      &tpu_non_overlapped_events;
  if (device_type == DeviceType::kTpu) {
    // For TPU device events, request_events.events may be overlapped in the
    // timeline. So first converts it to non-overlapped events in the timeline
    // before the breakdown.
    tpu_non_overlapped_events = ToNonOverlappedEvents(request_events.events);
  } else if (device_type == DeviceType::kGpu) {
    // For GPU device events, request_events.events come from non overlapped
    // StepEvents, so there is no need to convert to non overlapping events
    // again.
    non_overlapped_events = &(request_events.events);
  }

  int64_t device_time_ps = 0;
  int64_t write_time_ps = 0;
  int64_t read_time_ps = 0;
  int64_t host_preprocess_ps = 0;
  int64_t host_postprocess_ps = 0;
  int64_t host_runtime_ps = 0;
  int64_t host_batch_formation_ps = 0;
  int64_t idle_time_ps = 0;
  for (const auto& event : *non_overlapped_events) {
    const auto& duration_ps = event.span.duration_ps();
    switch (event.type) {
      case EventType::DEVICE_COMPUTE_16:
      case EventType::DEVICE_COMPUTE_32:
        device_time_ps += duration_ps;
        break;
      case EventType::HOST_TO_DEVICE:
        write_time_ps += duration_ps;
        break;
      case EventType::DEVICE_TO_HOST:
        read_time_ps += duration_ps;
        break;
      case EventType::HOST_PREPROCESS:
        host_preprocess_ps += duration_ps;
        break;
      case EventType::HOST_POSTPROCESS:
        host_postprocess_ps += duration_ps;
        break;
      case EventType::HOST_RUNTIME:
        host_runtime_ps += duration_ps;
        break;
      case EventType::HOST_BATCH_FORMATION:
        host_batch_formation_ps += duration_ps;
        break;
      case EventType::UNKNOWN_TIME:
        idle_time_ps += duration_ps;
        break;
      default:
        break;
    }
  }
  request_detail->set_device_time_ps(device_time_ps);
  request_detail->set_write_to_device_time_ps(write_time_ps);
  request_detail->set_read_from_device_time_ps(read_time_ps);
  request_detail->set_host_preprocessing_ps(host_preprocess_ps);
  request_detail->set_host_postprocessing_ps(host_postprocess_ps);
  request_detail->set_host_runtime_ps(host_runtime_ps);
  request_detail->set_host_batch_formation_ps(host_batch_formation_ps);
  request_detail->set_idle_time_ps(idle_time_ps);
}

// Compares two data points by duration.
// DataType can be either RequestDetail or BatchDetail.
template <typename DataType>
bool CompareByDuration(const DataType& a, const DataType& b) {
  return Timespan::ByDuration(
      Timespan::FromEndPoints(a.start_time_ps(), a.end_time_ps()),
      Timespan::FromEndPoints(b.start_time_ps(), b.end_time_ps()));
}

void BuildRequestDetails(
    const RequestEventsMap& request_events_map, DeviceType device_type,
    const int32_t host_id,
    tsl::protobuf::RepeatedPtrField<tensorflow::profiler::RequestDetail>*
        request_details) {
  for (auto& [group_id, request_events] : request_events_map) {
    if (request_events.request_timespan.duration_ps() == 0) continue;
    tensorflow::profiler::RequestDetail* request_detail =
        request_details->Add();
    request_detail->set_host_id(host_id);
    RequestEventsToDetails(device_type, group_id, request_events,
                           request_detail);
  }
  std::sort(request_details->begin(), request_details->end(),
            CompareByDuration<tensorflow::profiler::RequestDetail>);
}

void BuildBatchDetails(
    BatchEventsMap batch_events_map, const int32_t host_id,
    tsl::protobuf::RepeatedPtrField<tensorflow::profiler::BatchDetail>*
        batch_details) {
  for (auto& [group_id, batch_events] : batch_events_map) {
    batch_events.batch_detail_proto.set_host_id(host_id);
    *batch_details->Add() = std::move(batch_events.batch_detail_proto);
  }
  std::sort(batch_details->begin(), batch_details->end(),
            CompareByDuration<tensorflow::profiler::BatchDetail>);
}

// Parses TFstreamz xplane to get batching parameters, and stores the
// parameters to <model_id_db>.
void ParseTfstreamzForBatchingParameter(
    const XSpace& xspace, tensorflow::profiler::ModelIdDatabase* model_id_db) {
  const XPlane* tfstreamz_plane = ::tsl::profiler::FindPlaneWithName(
      xspace, tsl::profiler::kTFStreamzPlaneName);
  // There are two TFStreamz events per profile, one at the beginning, one at
  // the end of the profile, each represents a snapshot of the TFstreamz.
  // Use the last one as the source to get batching parameters because the
  // first snapshot might be taken before Tensorflow setting up the batching
  // parameters.
  if (tfstreamz_plane == nullptr || tfstreamz_plane->lines().empty() ||
      tfstreamz_plane->lines(0).events_size() != 2) {
    return;
  }
  XPlaneVisitor plane(tfstreamz_plane);
  XEventVisitor event(&plane, &tfstreamz_plane->lines(0),
                      &tfstreamz_plane->lines(0).events(1));

  static constexpr char kBatchingParamPrefix[] =
      "/tensorflow/serving/batching/";
  static constexpr char kBatchingParamNumBatchThreads[] = "num_batch_threads";
  static constexpr char kBatchingParamBatchTimeoutMicros[] =
      "batch_timeout_micros";
  static constexpr char kBatchingParamMaxBatchSize[] = "max_batch_size";
  static constexpr char kBatchingParamMaxEnqueuedBatches[] =
      "max_enqueued_batches";
  static constexpr char kBatchingParamAllowedBatchSizes[] =
      "allowed_batch_sizes";

  // Parse the batching parameters from TFstreamz and associate it them with
  // model IDs.
  absl::flat_hash_map<absl::string_view,
                      tensorflow::profiler::BatchingParameters>
      model_params;
  event.ForEachStat([&](const XStatVisitor& stat) {
    if (!absl::StartsWith(stat.Name(), kBatchingParamPrefix)) return;

    absl::string_view param_detail =
        stat.Name().substr(ABSL_ARRAYSIZE(kBatchingParamPrefix) - 1);
    auto [parse_success, model_id_tfstreamz] = ParseModelName(param_detail);
    if (!parse_success) {
      return;
    }

    if (absl::StartsWith(param_detail, kBatchingParamNumBatchThreads)) {
      model_params[model_id_tfstreamz].set_num_batch_threads(stat.IntValue());
    } else if (absl::StartsWith(param_detail,
                                kBatchingParamBatchTimeoutMicros)) {
      model_params[model_id_tfstreamz].set_batch_timeout_micros(
          stat.IntValue());
    } else if (absl::StartsWith(param_detail, kBatchingParamMaxBatchSize)) {
      model_params[model_id_tfstreamz].set_max_batch_size(stat.IntValue());
    } else if (absl::StartsWith(param_detail,
                                kBatchingParamMaxEnqueuedBatches)) {
      model_params[model_id_tfstreamz].set_max_enqueued_batches(
          stat.IntValue());
    } else if (absl::StartsWith(param_detail,
                                kBatchingParamAllowedBatchSizes)) {
      model_params[model_id_tfstreamz].set_allowed_batch_sizes(
          std::string(stat.StrOrRefValue()));
    }
  });

  // It is possible that the model IDs from Session.Run is in the format of
  // <model_id>:<version>, while the model IDs in TFstreamz is in the format
  // of <model_id> (without the version number). Build a map to connect the
  // model IDs in TFstreamz and Session.Run.
  absl::flat_hash_map<absl::string_view, std::vector<absl::string_view>>
      model_id_map;
  for (const auto& model_id_and_version : model_id_db->ids()) {
    size_t i = model_id_and_version.find_last_of(':');
    if (i == std::string::npos) {
      model_id_map[model_id_and_version].push_back(model_id_and_version);
    } else {
      // If there is a version number at the end of model_id, remove the
      // version number.
      absl::string_view version_str(model_id_and_version.data() + i + 1);
      int64_t version;
      bool success = absl::SimpleAtoi(version_str, &version);
      if (success) {
        absl::string_view model_id_only(model_id_and_version.data(), i);
        model_id_map[model_id_only].push_back(model_id_and_version);
      } else {
        LOG(ERROR) << "Can not parse model version number: " << version_str;
      }
    }
  }

  // One model ID from TFstreamz might map to multiple model IDs in
  // Session.Run, update the batching parameters of all the model IDs in
  // Session.Run.
  for (const auto& [model_id_tfstreamz, params] : model_params) {
    if (const std::vector<absl::string_view>* model_ids_session_run =
            gtl::FindOrNull(model_id_map, model_id_tfstreamz)) {
      for (const absl::string_view model_id_session_run :
           *model_ids_session_run) {
        (*model_id_db->mutable_id_to_batching_params())[model_id_session_run] =
            params;
      }
    }
  }
}

}  // namespace

std::pair<bool, absl::string_view> ParseModelName(absl::string_view param) {
  // Param can be in one of the two following formats:
  // batching_param{model_name=<model_name>}
  // batching_param{model_name=<model_name>, op_name=<op_name>}
  size_t label_begin = param.find_first_of('{');
  size_t label_end = param.find_last_of('}');
  if (label_begin == absl::string_view::npos ||
      label_end == absl::string_view::npos || label_end <= label_begin) {
    return {false, ""};
  }
  // Go over all the labels to look for model name.
  std::vector<absl::string_view> labels = absl::StrSplit(
      param.substr(label_begin + 1, label_end - label_begin - 1), ", ");
  for (const absl::string_view label : labels) {
    std::vector<absl::string_view> key_value = absl::StrSplit(label, '=');
    if (key_value.size() != 2) continue;
    if (key_value[0] == "model_name") {
      return {true, key_value[1]};
    }
  }
  // Unable to find model name.
  return {false, ""};
}

void GenerateInferenceStats(
    const std::vector<XPlane*>& device_traces,
    const StepEvents& nonoverlapped_step_events,
    const GroupMetadataMap& group_metadata_map, const XSpace& xspace,
    DeviceType device_type, int32_t host_id,
    tensorflow::profiler::InferenceStats* inference_stats) {
  tensorflow::profiler::PerHostInferenceStats* per_host_inference_stats =
      &(*inference_stats->mutable_inference_stats_per_host())[host_id];
  RequestEventsMap request_events_map;

  // Build the mapping from host event type to events.
  EventsByType host_events_by_type;
  const XPlane* host = tsl::profiler::FindPlaneWithName(
      xspace, tsl::profiler::kHostThreadsPlaneName);
  if (!host) return;
  XPlaneVisitor host_plane = CreateTfXPlaneVisitor(host);
  for (const auto& line : host->lines()) {
    for (const auto& event : line.events()) {
      XEventVisitor event_visitor(&host_plane, &line, &event);
      auto type = event_visitor.Type();
      if (!type.has_value()) {
        type = HostEventType::kUnknownHostEventType;
      }
      host_events_by_type[type.value()].push_back(event_visitor);
    }
  }

  BuildRequestEventsMap(device_traces, host_events_by_type, group_metadata_map,
                        nonoverlapped_step_events, device_type,
                        inference_stats->mutable_model_id_db(),
                        &request_events_map);
  BatchEventsMap batch_events_map;
  BuildBatchEventsMap(host_events_by_type, group_metadata_map,
                      &request_events_map, &batch_events_map);

  GenerateRequestAndBatchDelay(&request_events_map, &batch_events_map);
  GenerateRequestDetailedBreakdown(&request_events_map);

  GenerateTensorDetails(host_events_by_type, &request_events_map,
                        &batch_events_map, inference_stats);

  auto* request_details = per_host_inference_stats->mutable_request_details();
  BuildRequestDetails(request_events_map, device_type, host_id,
                      request_details);
  auto* batch_details = per_host_inference_stats->mutable_batch_details();
  BuildBatchDetails(std::move(batch_events_map), host_id, batch_details);

  ParseTfstreamzForBatchingParameter(xspace,
                                     inference_stats->mutable_model_id_db());
}

}  // namespace profiler
}  // namespace tensorflow
