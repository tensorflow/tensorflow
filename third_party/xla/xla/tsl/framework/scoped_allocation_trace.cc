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

#include "xla/tsl/framework/scoped_allocation_trace.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/logging.h"

namespace tsl {
namespace {

struct AnnotationState {
  std::vector<ScopedAllocationTrace::Frame> frames;
};

AnnotationState& ThreadAnnotationState() {
  static thread_local AnnotationState state;  // NOLINT
  return state;
}

}  // namespace

ScopedAllocationTrace::Arg::Arg(absl::string_view key,
                                const absl::AlphaNum& value
                                    ABSL_ATTRIBUTE_LIFETIME_BOUND)
    : key(key), value(value.Piece()) {}

ScopedAllocationTrace::Frame::Frame(absl::string_view name) : name(name) {}

ScopedAllocationTrace::Frame& ScopedAllocationTrace::Frame::Add(
    absl::string_view key, const absl::AlphaNum& value) {
  args.emplace_back(std::string(key), std::string(value.Piece()));
  return *this;
}

ScopedAllocationTrace::Snapshot::Snapshot(std::vector<Frame> frames)
    : frames(std::move(frames)) {}

ScopedAllocationTrace::ScopedAllocationTrace(absl::string_view name,
                                             std::initializer_list<Arg> args) {
  Frame frame(name);
  frame.args.reserve(args.size());
  for (const Arg& arg : args) {
    frame.args.emplace_back(std::string(arg.key), std::string(arg.value));
  }
  AnnotationState& state = ThreadAnnotationState();
  state.frames.push_back(std::move(frame));
}

ScopedAllocationTrace::ScopedAllocationTrace(Frame frame) {
  AnnotationState& state = ThreadAnnotationState();
  state.frames.push_back(std::move(frame));
}

ScopedAllocationTrace::~ScopedAllocationTrace() {
  AnnotationState& state = ThreadAnnotationState();
  DCHECK(!state.frames.empty());
  state.frames.pop_back();
}

ScopedAllocationTrace::Snapshot ScopedAllocationTrace::Current() {
  return Snapshot(ThreadAnnotationState().frames);
}

}  // namespace tsl
