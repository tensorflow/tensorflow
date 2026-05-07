/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tsl/profiler/lib/profiler_orchestrator.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/profiler/convert/post_process_single_host_xplane.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/host_info.h"
#include "tsl/profiler/lib/profiler_session.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"

namespace tsl {
namespace profiler {

ProfilerSessionOrchestrator::ProfilerSessionOrchestrator(
    const tensorflow::ProfileOptions& options)
    : options_(options) {}

ProfilerSessionOrchestrator::~ProfilerSessionOrchestrator() {
  Stop().IgnoreError();
}

absl::Status ProfilerSessionOrchestrator::Start() {
  if (session_ != nullptr) {
    return absl::FailedPreconditionError("Session already started.");
  }
  global_start_time_ns_ = tsl::Env::Default()->NowNanos();
  session_ = tsl::ProfilerSession::Create(options_);
  if (session_ == nullptr) {
    return absl::InternalError("Failed to create ProfilerSession.");
  }
  return session_->Status();
}

absl::Status ProfilerSessionOrchestrator::Stop() {
  if (session_ == nullptr) {
    return absl::OkStatus();  // Already stopped or not started.
  }
  session_.reset();
  return absl::OkStatus();
}

absl::StatusOr<int> ProfilerSessionOrchestrator::Consume() {
  if (session_ == nullptr) {
    return absl::FailedPreconditionError("Session not started.");
  }

  consume_buffers_.emplace_back(sizeof(std::vector<char>));
  auto& buffer = consume_buffers_.back();
  TF_RETURN_IF_ERROR(session_->Consume(buffer.data()));

  uint64_t now = tsl::Env::Default()->NowNanos();
  consume_stop_times_.push_back(now);

  return consume_buffers_.size() - 1;
}

absl::Status ProfilerSessionOrchestrator::Serialize(int buffer_index) {
  if (session_ == nullptr) {
    return absl::FailedPreconditionError("Session not started.");
  }

  if (buffer_index < 0 || buffer_index >= consume_buffers_.size()) {
    return absl::InvalidArgumentError("Invalid buffer index.");
  }
  serialize_space_.Clear();
  auto& buffer = consume_buffers_[buffer_index];
  TF_RETURN_IF_ERROR(session_->Serialize(buffer.data(), &serialize_space_));

  serialize_space_.add_hostnames(tsl::port::Hostname());
  profiler::SetXSpacePidIfNotSet(serialize_space_,
                                 tsl::Env::Default()->GetProcessId());

  uint64_t stop_time = 0;
  if (buffer_index < consume_stop_times_.size()) {
    stop_time = consume_stop_times_[buffer_index];
  }
  profiler::PostProcessSingleHostXSpace(&serialize_space_,
                                        global_start_time_ns_, stop_time);

  {
    profiler::XPlaneBuilder xplane(profiler::FindOrAddMutablePlaneWithName(
        &serialize_space_, tsl::profiler::kTaskEnvPlaneName));
    xplane.AddStatValue(
        *xplane.GetOrCreateStatMetadata(tsl::profiler::GetTaskEnvStatTypeStr(
            tsl::profiler::kEnvProfileOptions)),
        options_);
  }

  return absl::OkStatus();
}

void ProfilerSessionOrchestrator::ClearConsumeBuffers() {
  std::vector<std::vector<uint8_t>>().swap(consume_buffers_);
  consume_stop_times_.clear();
}

}  // namespace profiler
}  // namespace tsl
