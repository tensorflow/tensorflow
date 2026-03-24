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

#include "xla/backends/gpu/runtime/while_loop.h"

#include <cstddef>
#include <list>
#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"

namespace xla::gpu {

static thread_local std::list<WhileLoopState> while_loop_stack;

static const WhileLoopState* EnterWhileLoop(absl::string_view loop_name,
                                            std::optional<size_t> trip_count) {
  size_t depth = while_loop_stack.size();
  while_loop_stack.push_back(
      WhileLoopState{std::string(loop_name), trip_count, depth, 0});
  return &while_loop_stack.back();
}

static const WhileLoopState* IncWhileLoopIteration() {
  if (while_loop_stack.empty()) return nullptr;
  ++while_loop_stack.back().loop_iteration;
  return &while_loop_stack.back();
}

static WhileLoopState ExitWhileLoop() {
  WhileLoopState state = std::move(while_loop_stack.back());
  while_loop_stack.pop_back();
  return state;
}

const WhileLoopState* IsInsideWhileLoop() {
  if (while_loop_stack.empty()) return nullptr;
  return &while_loop_stack.back();
}

ScopedWhileLoop::ScopedWhileLoop(absl::string_view loop_name,
                                 std::optional<size_t> trip_count)
    : state(EnterWhileLoop(loop_name, trip_count)) {}

ScopedWhileLoop::~ScopedWhileLoop() { ExitWhileLoop(); }

absl::string_view ScopedWhileLoop::loop_name() const {
  return state->loop_name;
}

std::optional<size_t> ScopedWhileLoop::trip_count() const {
  return state->loop_trip_count;
}

size_t ScopedWhileLoop::loop_depth() const { return state->loop_depth; }

size_t ScopedWhileLoop::loop_iteration() const { return state->loop_iteration; }

void ScopedWhileLoop::IncLoopIteration() { IncWhileLoopIteration(); }

}  // namespace xla::gpu
