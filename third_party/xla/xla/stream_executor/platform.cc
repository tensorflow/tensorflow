/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/platform.h"

#include <map>
#include <string>

#include "absl/status/status.h"

namespace stream_executor {

std::string StreamPriorityToString(StreamPriority priority) {
  switch (priority) {
    case StreamPriority::Lowest:
      return "Lowest priority";
    case StreamPriority::Highest:
      return "Highest priority";
    default:
      return "Default Priority";
  }
}

StreamExecutorConfig::StreamExecutorConfig() : ordinal(-1) {}

StreamExecutorConfig::StreamExecutorConfig(int ordinal_in)
    : ordinal(ordinal_in) {}

Platform::~Platform() {}

bool Platform::Initialized() const { return true; }

absl::Status Platform::Initialize(
    const std::map<std::string, std::string> &platform_options) {
  if (!platform_options.empty()) {
    return absl::UnimplementedError(
        "this platform does not support custom initialization");
  }
  return absl::OkStatus();
}

}  // namespace stream_executor
