/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/gpu/scoped_command_buffer_annotation.h"

#include <cstddef>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"

namespace stream_executor {
namespace {

struct ThreadState {
  // The number of enabled roots open on the thread.
  int depth = 0;
  std::string joined;
  // The size of `joined` before each pushed frame.
  std::vector<size_t> prev_sizes;
};

ThreadState& state() {
  thread_local ThreadState s;
  return s;
}

}  // namespace

ScopedCommandBufferAnnotation ScopedCommandBufferAnnotation::Root(
    bool enabled, absl::string_view annotation) {
  return ScopedCommandBufferAnnotation(RootTag{}, enabled, annotation);
}

ScopedCommandBufferAnnotation::ScopedCommandBufferAnnotation(
    RootTag, bool enabled, absl::string_view annotation) {
  if (!enabled) {
    return;
  }
  ++state().depth;
  is_root_ = true;
  Push(annotation);
}

ScopedCommandBufferAnnotation::ScopedCommandBufferAnnotation(
    absl::string_view annotation) {
  if (IsActive()) {
    Push(annotation);
  }
}

ScopedCommandBufferAnnotation::~ScopedCommandBufferAnnotation() {
  ThreadState& s = state();
  if (pushed_ && !s.prev_sizes.empty()) {
    s.joined.resize(s.prev_sizes.back());
    s.prev_sizes.pop_back();
  }
  if (is_root_ && --s.depth == 0) {
    // Recording is off again: drop any leftover text.
    s.joined.clear();
    s.prev_sizes.clear();
  }
}

void ScopedCommandBufferAnnotation::Push(absl::string_view annotation) {
  if (annotation.empty()) {
    return;
  }
  ThreadState& s = state();
  s.prev_sizes.push_back(s.joined.size());
  if (!s.joined.empty()) {
    s.joined.append("::");
  }
  s.joined.append(annotation.data(), annotation.size());
  pushed_ = true;
}

bool ScopedCommandBufferAnnotation::IsActive() { return state().depth > 0; }

absl::string_view ScopedCommandBufferAnnotation::GetCurrentAnnotation() {
  return state().joined;
}

}  // namespace stream_executor
