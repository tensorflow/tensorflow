/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/tsl/profiler/lib/profiler_controller.h"

#include <memory>
#include <utility>

#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/profiler/lib/profiler_interface.h"
#include "tensorflow/tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {

ProfilerController::ProfilerController(
    std::unique_ptr<ProfilerInterface> profiler)
    : profiler_(std::move(profiler)) {}

ProfilerController::~ProfilerController() {
  // Ensure a successfully started profiler is stopped.
  if (state_ == ProfilerState::kStart && status_.ok()) {
    profiler_->Stop().IgnoreError();
  }
}

Status ProfilerController::Start() {
  Status status;
  if (state_ == ProfilerState::kInit) {
    state_ = ProfilerState::kStart;
    if (status_.ok()) {
      status = status_ = profiler_->Start();
    } else {
      status = errors::Aborted("Previous call returned an error.");
    }
  } else {
    status = errors::Aborted("Start called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

Status ProfilerController::Stop() {
  Status status;
  if (state_ == ProfilerState::kStart) {
    state_ = ProfilerState::kStop;
    if (status_.ok()) {
      status = status_ = profiler_->Stop();
    } else {
      status = errors::Aborted("Previous call returned an error.");
    }
  } else {
    status = errors::Aborted("Stop called in the wrong order");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

Status ProfilerController::CollectData(tensorflow::profiler::XSpace* space) {
  Status status;
  if (state_ == ProfilerState::kStop) {
    state_ = ProfilerState::kCollectData;
    if (status_.ok()) {
      status = status_ = profiler_->CollectData(space);
    } else {
      status = errors::Aborted("Previous call returned an error.");
    }
  } else {
    status = errors::Aborted("CollectData called in the wrong order.");
  }
  if (!status.ok()) LOG(ERROR) << status;
  return status;
}

}  // namespace profiler
}  // namespace tsl
