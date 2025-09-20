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

#include "xla/tsl/profiler/utils/preprocess_xplane.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_mutators.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::profiler::HostEventType;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XEventMetadata;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;
using ::tsl::profiler::XStat;

// Generates a new HLO module ID based on the event metadata and plane id.
// Returns 0 if a new ID cannot be generated.
uint64_t GenerateNewHloModuleId(const XEventMetadataVisitor& event_metadata,
                                uint64_t plane_id) {
  auto hlo_proto_stat = event_metadata.GetStat(StatType::kHloProto);
  if (!hlo_proto_stat || hlo_proto_stat->ValueCase() != XStat::kBytesValue) {
    return 0;
  }

  auto fingerprint_stat = event_metadata.GetStat(StatType::kFingerprint);
  uint64_t fingerprint =
      fingerprint_stat ? fingerprint_stat->IntOrUintValue() : 0;

  absl::string_view hlo_proto_bytes = hlo_proto_stat->BytesValue();
  std::string fingerprint_str =
      absl::StrCat(hlo_proto_bytes, plane_id, fingerprint);
  return absl::HashOf(fingerprint_str);
}

// Adds an event to the new line builder, potentially updating its metadata id.
void AddEventWithNormalizedId(
    XLineBuilder& new_line_builder, const XEventVisitor& event,
    const absl::flat_hash_map<int64_t, int64_t>& old_to_new_metadata_id) {
  int64_t original_metadata_id = event.metadata()->id();
  auto it = old_to_new_metadata_id.find(original_metadata_id);
  int64_t new_metadata_id =
      (it != old_to_new_metadata_id.end()) ? it->second : original_metadata_id;

  XEvent new_event(event.RawEvent());
  new_event.set_metadata_id(new_metadata_id);
  new_line_builder.AddEvent(new_event);
}

void NormalizeGpuHloModuleIdsForPlane(XPlane* plane, uint64_t plane_id) {
  absl::flat_hash_map<int64_t, int64_t> old_to_new_metadata_id;
  XPlaneVisitor plane_visitor = CreateTfXPlaneVisitor(plane);
  const XStatMetadata* hlo_proto_stat_metadata =
      plane_visitor.GetStatMetadataByType(StatType::kHloProto);
  if (hlo_proto_stat_metadata == nullptr) {
    return;
  }

  plane_visitor.ForEachEventMetadata(
      [&](const XEventMetadataVisitor& event_metadata) {
        uint64_t new_program_id =
            GenerateNewHloModuleId(event_metadata, plane_id);
        if (new_program_id != 0) {
          old_to_new_metadata_id[event_metadata.Id()] = new_program_id;
        }
      });

  if (old_to_new_metadata_id.empty()) {
    return;
  }

  XPlane new_plane;
  new_plane.set_id(plane->id());
  new_plane.set_name(plane->name());

  for (auto const& [id, stat_metadata] : plane->stat_metadata()) {
    (*new_plane.mutable_stat_metadata())[id] = stat_metadata;
  }

  for (auto const& [id, event_metadata] : plane->event_metadata()) {
    if (old_to_new_metadata_id.contains(id)) {
      XEventMetadata new_event_metadata = event_metadata;
      new_event_metadata.set_id(old_to_new_metadata_id.at(id));
      (*new_plane.mutable_event_metadata())[new_event_metadata.id()] =
          new_event_metadata;
    } else {
      (*new_plane.mutable_event_metadata())[id] = event_metadata;
    }
  }

  XPlaneVisitor old_plane_visitor(plane);
  XPlaneBuilder new_plane_builder(&new_plane);
  old_plane_visitor.ForEachLine([&](const XLineVisitor& line) {
    XLineBuilder new_line_builder =
        new_plane_builder.GetOrCreateLine(line.Id());
    new_line_builder.SetName(line.Name());
    new_line_builder.SetTimestampNs(line.TimestampNs());
    line.ForEachEvent([&](const XEventVisitor& event) {
      AddEventWithNormalizedId(new_line_builder, event, old_to_new_metadata_id);
    });
  });

  plane->Swap(&new_plane);
}

void NormalizeGpuHloModuleIds(XSpace* space) {
  uint64_t plane_id = 0;
  for (XPlane& plane : *space->mutable_planes()) {
    NormalizeGpuHloModuleIdsForPlane(&plane, plane_id++);
  }
}

// Categorizes mutators created from factories into line mutators and
// event-metadata-based mutators.
void CategorizeMutators(
    const std::vector<std::unique_ptr<XplaneEventMutatorFactory>>&
        mutator_factories,
    XPlaneBuilder& plane_builder,
    absl::flat_hash_map<int64_t,
                        std::vector<std::unique_ptr<XplaneEventMutator>>>&
        mutators_from_event_metadata_id,
    std::vector<std::unique_ptr<XplaneEventMutator>>& line_mutators) {
  for (const auto& mutator_factory : mutator_factories) {
    auto mutators = mutator_factory->CreateMutators(plane_builder);
    for (auto& mutator : mutators) {
      if (mutator->event_metadata()) {
        mutators_from_event_metadata_id[mutator->event_metadata()->id()]
            .push_back(std::move(mutator));
      } else {
        line_mutators.push_back(std::move(mutator));
      }
    }
  }
}

// Applies line and event mutators to a single XLineBuilder.
void ApplyMutatorsToLine(
    XLineBuilder& line_builder,
    const absl::flat_hash_map<int64_t,
                              std::vector<std::unique_ptr<XplaneEventMutator>>>&
        mutators_from_event_metadata_id,
    const std::vector<std::unique_ptr<XplaneEventMutator>>& line_mutators) {
  for (const auto& mutator : line_mutators) {
    mutator->MutateEventsInLine(line_builder);
  }
  if (mutators_from_event_metadata_id.empty()) {
    return;
  }
  line_builder.ForEachEvent([&](XEventBuilder event_builder) {
    if (auto it =
            mutators_from_event_metadata_id.find(event_builder.MetadataId());
        it != mutators_from_event_metadata_id.end()) {
      for (const auto& mutator : it->second) {
        mutator->Mutate(event_builder);
      }
    }
  });
}

void MutateXPlane(XPlane& plane,
                  const std::vector<std::unique_ptr<XplaneEventMutatorFactory>>&
                      mutator_factories) {
  if (mutator_factories.empty()) {
    return;
  }

  XPlaneBuilder plane_builder(&plane);
  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<XplaneEventMutator>>>
      mutators_from_event_metadata_id;
  std::vector<std::unique_ptr<XplaneEventMutator>> line_mutators;

  CategorizeMutators(mutator_factories, plane_builder,
                     mutators_from_event_metadata_id, line_mutators);

  if (mutators_from_event_metadata_id.empty() && line_mutators.empty()) {
    return;
  }

  plane_builder.ForEachLine([&](XLineBuilder line_builder) {
    ApplyMutatorsToLine(line_builder, mutators_from_event_metadata_id,
                        line_mutators);
  });
}

std::vector<std::unique_ptr<XplaneEventMutatorFactory>>
CreateMutatorFactories() {
  std::vector<std::unique_ptr<XplaneEventMutatorFactory>> mutator_factories;
  mutator_factories.push_back(ThreadpoolLineMutatorFactory::CreateFactory());
  mutator_factories.push_back(XplaneRootEventMutatorFactory::CreateFactory(
      HostEventType::kProcessBatch, 2));
  mutator_factories.push_back(XplaneRootEventMutatorFactory::CreateFactory(
      HostEventType::kBatchingSessionRun, 1));
  // Legacy asynchronous TPU execution dispatcher
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          /*producer_event=*/HostEventType::kExecutorStateProcess,
          /*consumer_event=*/HostEventType::kTpuExecuteOp, ContextType::kLegacy,
          /*unique_stats=*/false,
          XContextStatsAccessor<uint64_t, StatType::kStepId>,
          XContextStatsAccessor<uint64_t,
                                StatType::kIterNum>>::CreateFactory());

// Queue : enque => deque.
#define ADD_QUEUE_CONNECTION(__enque_event__, __deque_event__)            \
  mutator_factories.push_back(                                            \
      XplaneConnectedEventMutatorFactory<                                 \
          HostEventType::__enque_event__, HostEventType::__deque_event__, \
          ContextType::kTpuStream, /*unique_stats=*/true,                 \
          XContextStatsAccessor<uint64, StatType::kRequestId>,            \
          XContextStatsAccessor<uint64,                                   \
                                StatType::kQueueAddr>>::CreateFactory())
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kRunProgramRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kHostCallbackRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferH2DRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferPreprocessedH2DRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kTransferD2HRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceRecvRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRecvLocalRequest);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kCustomWait);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceSendRequestMulti);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kOnDeviceRecvRequestMulti);
  ADD_QUEUE_CONNECTION(kEnqueueRequestLocked, kPjrtAsyncWait);
#undef ADD_QUEUE_CONNECTION

  // Fixup run_id from Host TraceMe to 28 LSB
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kDoEnqueueProgram>::CreateFactory());
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kCompleteCallbacks>::CreateFactory());
  mutator_factories.push_back(
      HostRunIdMutatorFactory<
          HostEventType::kDoEnqueueContinuationProgram>::CreateFactory());

  // TPU program execution launch related
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          /*producer_event=*/HostEventType::kDoEnqueueProgram,
          /*consumer_event=*/HostEventType::kCompleteCallbacks,
          ContextType::kTpuLaunch,
          /*unique_stats=*/true,
          XContextStatsAccessor<uint64_t, StatType::kDeviceOrdinal>,
          XContextStatsAccessor<uint64_t, StatType::kQueueId>,
          XContextStatsAccessor<uint64_t, StatType::kRunId>,
          XContextStatsAccessorWithDefault<uint64_t, StatType::kCoreType,
                                           0ULL>>::CreateFactory());

  mutator_factories.push_back(TpuModuleLineMutatorFactory::CreateFactory());
  return mutator_factories;
}

}  // namespace

void PreprocessXPlane(XPlane* plane) {
  if (plane == nullptr) {
    return;
  }

  auto mutator_factories = CreateMutatorFactories();
  MutateXPlane(*plane, mutator_factories);
}

void PreprocessXSpace(XSpace* space) {
  if (space == nullptr) {
    return;
  }

  NormalizeGpuHloModuleIds(space);
  auto mutator_factories = CreateMutatorFactories();
  for (XPlane& plane : *space->mutable_planes()) {
    MutateXPlane(plane, mutator_factories);
  }
}

}  // namespace profiler
}  // namespace tsl
