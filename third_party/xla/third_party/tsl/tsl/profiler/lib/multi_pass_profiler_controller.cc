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

#include "tsl/profiler/lib/multi_pass_profiler_controller.h"

#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

MultiPassProfilerController::MultiPassProfilerController(
    std::unique_ptr<MultiPassProfilerInterface> profiler)
    : profiler_(std::move(profiler)) {}

MultiPassProfilerController::~MultiPassProfilerController() {
  // Ensure an active pass is stopped.
  if (state_ == MultiPassProfilerState::kPassStarted) {
    StopPass().IgnoreError();
  }
  if (state_ == MultiPassProfilerState::kStarted ||
      state_ == MultiPassProfilerState::kPassStopped) {
    profiler_->Stop().IgnoreError();
    state_ = MultiPassProfilerState::kStopped;
  }
}

bool MultiPassProfilerController::NeedMorePasses() {
  if (state_ == MultiPassProfilerState::kPassStopped ||
      state_ == MultiPassProfilerState::kStarted ||
      state_ == MultiPassProfilerState::kInit) {
    if (!status_.ok()) return false;
    return profiler_->NeedMorePasses();
  }
  return false;
}

absl::Status MultiPassProfilerController::StartPass() {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kStarted ||
      state_ == MultiPassProfilerState::kPassStopped) {
    state_ = MultiPassProfilerState::kPassStarted;
    if (status_.ok()) {
      status = status_ = profiler_->StartPass();
    } else {
      status = absl::AbortedError("Previous call returned an error.");
    }
  } else {
    status = absl::AbortedError("StartPass called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::PushRange(absl::string_view name) {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kPassStarted) {
    if (status_.ok()) {
      status = status_ = profiler_->PushRange(name);
      if (status_.ok()) {
        ++active_range_count_;
      }
    } else {
      status = absl::AbortedError("Previous call returned an error.");
    }
  } else {
    status = absl::AbortedError("PushRange called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::PopRange() {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kPassStarted) {
    if (active_range_count_ <= 0) {
      if (status_.ok()) {
        status = status_ =
            absl::InternalError("PopRange called with no active ranges.");
      } else {
        status = absl::AbortedError("Previous call returned an error.");
      }
    } else {
      --active_range_count_;
      if (status_.ok()) {
        status = status_ = profiler_->PopRange();
      } else {
        profiler_->PopRange().IgnoreError();
        status = absl::AbortedError("Previous call returned an error.");
      }
    }
  } else {
    status = absl::AbortedError("PopRange called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::PopAllRanges() {
  absl::Status status;
  while (active_range_count_ > 0) {
    status.Update(PopRange());
  }
  return status;
}

absl::Status MultiPassProfilerController::StopPass() {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kPassStarted) {
    status = PopAllRanges();
    state_ = MultiPassProfilerState::kPassStopped;
    if (status_.ok()) {
      status = status_ = profiler_->StopPass();
    } else {
      profiler_->StopPass().IgnoreError();
      if (status.ok()) {
        status = absl::AbortedError("Previous call returned an error.");
      }
    }
  } else {
    status = absl::AbortedError("StopPass called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::CollectData(
    tensorflow::profiler::XSpace* space) {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kInit) {
    status = absl::OkStatus();
    state_ = MultiPassProfilerState::kCollectData;
  } else if (state_ == MultiPassProfilerState::kStopped) {
    state_ = MultiPassProfilerState::kCollectData;
    if (status_.ok()) {
      status = status_ = profiler_->CollectData(space);
    } else {
      status = absl::AbortedError("Previous call returned an error.");
    }
  } else {
    status = absl::AbortedError("CollectData called in the wrong order.");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::Start() {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kInit) {
    state_ = MultiPassProfilerState::kStarted;
    if (status_.ok()) {
      status = status_ = profiler_->Start();
    } else {
      status = absl::AbortedError("Previous call returned an error.");
    }
  } else {
    status = absl::AbortedError("Start called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

absl::Status MultiPassProfilerController::Stop() {
  absl::Status status;
  if (state_ == MultiPassProfilerState::kPassStarted) {
    status = StopPass();
  }
  if (state_ == MultiPassProfilerState::kPassStopped ||
      state_ == MultiPassProfilerState::kStarted) {
    state_ = MultiPassProfilerState::kStopped;
    if (status_.ok()) {
      status = status_ = profiler_->Stop();
    } else if (status.ok()) {
      status = absl::AbortedError("Previous call returned an error.");
    }
  } else if (status.ok()) {
    status = absl::AbortedError("Stop called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

}  // namespace profiler

}  // namespace tsl
