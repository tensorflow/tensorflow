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

#include "tensorflow/tsl/profiler/utils/preprocess_xplane.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/tsl/profiler/lib/connected_traceme.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"
#include "tensorflow/tsl/profiler/utils/xplane_builder.h"
#include "tensorflow/tsl/profiler/utils/xplane_schema.h"

namespace tsl {
namespace profiler {
namespace {

using ::tsl::profiler::HostEventType;
using ::tsl::profiler::StatType;
using ::tsl::profiler::XEventBuilder;
using ::tsl::profiler::XLineBuilder;
using ::tsl::profiler::XPlane;
using ::tsl::profiler::XPlaneBuilder;
using ::tsl::profiler::XSpace;

void MutateXPlane(XPlane* plane,
                  const std::vector<std::unique_ptr<XplaneEventMutatorFactory>>&
                      mutator_factories) {
  XPlaneBuilder plane_builder(plane);

  absl::flat_hash_map<int64_t, std::vector<std::unique_ptr<XplaneEventMutator>>>
      mutators_from_event_metadata_id;
  std::vector<std::unique_ptr<XplaneEventMutator>> line_mutators;
  for (const auto& mutator_factory : mutator_factories) {
    auto mutators = mutator_factory->CreateMutators(&plane_builder);
    for (auto& mutator : mutators) {
      if (mutator->event_metadata()) {
        auto id = mutator->event_metadata()->id();
        mutators_from_event_metadata_id[id].push_back(std::move(mutator));
      } else {
        line_mutators.push_back(std::move(mutator));
      }
    }
  }
  if (mutators_from_event_metadata_id.empty() && line_mutators.empty()) {
    return;
  }

  plane_builder.ForEachLine([&](XLineBuilder line_builder) {
    for (const auto& mutator : line_mutators) {
      mutator->MutateEventsInLine(&line_builder);
    }
    if (mutators_from_event_metadata_id.empty()) return;
    line_builder.ForEachEvent([&](XEventBuilder event_builder) {
      auto event_mutators =
          mutators_from_event_metadata_id.find(event_builder.MetadataId());
      if (event_mutators != mutators_from_event_metadata_id.end()) {
        for (const auto& mutator : event_mutators->second) {
          mutator->Mutate(&event_builder);
        }
      }
    });
  });
}

std::vector<std::unique_ptr<XplaneEventMutatorFactory>>
CreateMutatorFactories() {
  std::vector<std::unique_ptr<XplaneEventMutatorFactory>> mutator_factories;
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

  // TPU program execution launch related
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          /*producer_event=*/HostEventType::kDoEnqueueProgram,
          /*consumer_event=*/HostEventType::kCompleteCallbacks,
          ContextType::kTpuLaunch,
          /*unique_stats=*/true,
          XContextStatsAccessor<uint64_t, StatType::kDeviceOrdinal>,
          XContextStatsAccessor<uint64_t, StatType::kQueueId>,
          XContextStatsAccessor<uint64_t, StatType::kRunId>>::CreateFactory());
  // TODO(jiesun): remove kDoEnqueueContinuationProgram after 04/21/2023
  // see cl/443548431.
  mutator_factories.push_back(
      XplaneConnectedEventMutatorFactory<
          /*producer_event=*/HostEventType::kDoEnqueueContinuationProgram,
          /*consumer_event=*/HostEventType::kCompleteCallbacks,
          ContextType::kTpuLaunch,
          /*unique_stats=*/true,
          XContextStatsAccessor<uint64_t, StatType::kDeviceOrdinal>,
          XContextStatsAccessor<uint64_t, StatType::kQueueId>,
          XContextStatsAccessor<uint64_t, StatType::kRunId>>::CreateFactory());

  mutator_factories.push_back(TpuModuleLineMutatorFactory::CreateFactory());
  return mutator_factories;
}

}  // namespace

void PreprocessXPlane(XPlane* plane) {
  auto mutator_factories = CreateMutatorFactories();
  MutateXPlane(plane, mutator_factories);
}

void PreprocessXSpace(XSpace* space) {
  auto mutator_factories = CreateMutatorFactories();
  for (XPlane& plane : *space->mutable_planes()) {
    MutateXPlane(&plane, mutator_factories);
  }
}

}  // namespace profiler
}  // namespace tsl
