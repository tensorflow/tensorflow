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
#include "tsl/profiler/backends/cpu/threadpool_listener.h"

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/tracing.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/backends/cpu/threadpool_listener_state.h"
#include "tsl/profiler/backends/cpu/traceme_recorder.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme_encode.h"
#include "tsl/profiler/utils/time_utils.h"
#include "tsl/profiler/utils/xplane_schema.h"

namespace tsl {
namespace profiler {
namespace {

void RegisterThreadpoolEventCollector(ThreadpoolEventCollector* collector) {
  tracing::SetEventCollector(tracing::EventCategory::kScheduleClosure,
                             collector);
  tracing::SetEventCollector(tracing::EventCategory::kRunClosure, collector);
}

void UnregisterThreadpoolEventCollector() {
  tracing::SetEventCollector(tracing::EventCategory::kScheduleClosure, nullptr);
  tracing::SetEventCollector(tracing::EventCategory::kRunClosure, nullptr);
}

}  // namespace

void ThreadpoolEventCollector::RecordEvent(uint64 arg) const {
  int64_t now = GetCurrentTimeNanos();
  TraceMeRecorder::Record(
      {TraceMeEncode(kThreadpoolListenerRecord,
                     {{"_pt", ContextType::kThreadpoolEvent}, {"_p", arg}}),
       now, now});
}
void ThreadpoolEventCollector::StartRegion(uint64 arg) const {
  int64_t now = GetCurrentTimeNanos();
  TraceMeRecorder::Record(
      {TraceMeEncode(kThreadpoolListenerStartRegion,
                     {{"_ct", ContextType::kThreadpoolEvent}, {"_c", arg}}),
       now, now});
}
void ThreadpoolEventCollector::StopRegion() const {
  int64_t now = GetCurrentTimeNanos();
  TraceMeRecorder::Record(
      {TraceMeEncode(kThreadpoolListenerStopRegion, {}), now, now});
}

absl::Status ThreadpoolProfilerInterface::Start() {
  if (tracing::EventCollector::IsEnabled()) {
    LOG(WARNING) << "[ThreadpoolEventCollector] EventCollector is enabled, Not "
                    "collecting events from ThreadPool.";
    status_ = absl::FailedPreconditionError(
        "ThreadpoolEventCollector is enabled, Not collecting events from "
        "ThreadPool.");
    return absl::OkStatus();
  }
  event_collector_ = std::make_unique<ThreadpoolEventCollector>();
  RegisterThreadpoolEventCollector(event_collector_.get());
  threadpool_listener::Activate();
  return absl::OkStatus();
}

absl::Status ThreadpoolProfilerInterface::Stop() {
  threadpool_listener::Deactivate();
  UnregisterThreadpoolEventCollector();
  return absl::OkStatus();
}

absl::Status ThreadpoolProfilerInterface::CollectData(
    tensorflow::profiler::XSpace* space) {
  if (!status_.ok()) {
    *space->add_errors() = status_.ToString();
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tsl
