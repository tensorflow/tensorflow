/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/stream_executor/timer.h"

#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace perftools {
namespace gputools {

static internal::TimerInterface *CreateTimerImplementation(
    StreamExecutor *parent) {
  PlatformKind platform_kind = parent->platform_kind();
  if (platform_kind == PlatformKind::kCuda) {
    return (*internal::MakeCUDATimerImplementation())(parent);
  } else if (platform_kind == PlatformKind::kOpenCL ||
             platform_kind == PlatformKind::kOpenCLAltera) {
    return (*internal::MakeOpenCLTimerImplementation())(parent);
  } else if (platform_kind == PlatformKind::kHost) {
    return internal::MakeHostTimerImplementation(parent);
  } else if (platform_kind == PlatformKind::kMock) {
    return nullptr;
  } else {
    LOG(FATAL) << "cannot create timer implementation for platform kind: "
               << PlatformKindString(platform_kind);
  }
}

Timer::Timer(StreamExecutor *parent)
    : implementation_(CreateTimerImplementation(parent)), parent_(parent) {}

Timer::~Timer() { parent_->DeallocateTimer(this); }

uint64 Timer::Microseconds() const { return implementation_->Microseconds(); }

uint64 Timer::Nanoseconds() const { return implementation_->Nanoseconds(); }

}  // namespace gputools
}  // namespace perftools
