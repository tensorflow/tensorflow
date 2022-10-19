/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/cl_event.h"

#include <string>
#include <utility>

#include "tensorflow/lite/delegates/gpu/cl/opencl_wrapper.h"

namespace tflite {
namespace gpu {
namespace cl {

CLEvent::CLEvent(cl_event event) : event_(event) {}

CLEvent::CLEvent(CLEvent&& event)
    : event_(event.event_), name_(std::move(event.name_)) {
  event.event_ = nullptr;
}

CLEvent& CLEvent::operator=(CLEvent&& event) {
  if (this != &event) {
    Release();
    std::swap(event_, event.event_);
    name_ = std::move(event.name_);
  }
  return *this;
}

uint64_t CLEvent::GetStartedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                          &time_ns, nullptr);
  return time_ns;
}

uint64_t CLEvent::GetFinishedTimeNs() const {
  cl_ulong time_ns;
  clGetEventProfilingInfo(event_, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                          &time_ns, nullptr);
  return time_ns;
}

double CLEvent::GetEventTimeMs() const {
  const uint64_t start = GetStartedTimeNs();
  const uint64_t end = GetFinishedTimeNs();
  const uint64_t time_ns = (end - start);

  return static_cast<double>(time_ns) * 1e-6;
}

uint64_t CLEvent::GetEventTimeNs() const {
  return GetFinishedTimeNs() - GetStartedTimeNs();
}

void CLEvent::SetName(const std::string& name) { name_ = name; }

void CLEvent::Wait() const { clWaitForEvents(1, &event_); }

CLEvent::~CLEvent() { Release(); }

void CLEvent::Release() {
  if (event_) {
    clReleaseEvent(event_);
    event_ = nullptr;
  }
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
