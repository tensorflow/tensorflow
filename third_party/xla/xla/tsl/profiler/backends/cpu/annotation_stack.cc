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

#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace profiler {

// Returns the annotation data for the given generation.
static auto GetAnnotationData(const std::atomic<int>& atomic) {
  static thread_local struct {
    int generation = 0;
    std::vector<size_t> stack;
    std::string string;
    std::vector<int64_t> scope_range_id_stack;
  } data;
  int generation = atomic.load(std::memory_order_acquire);
  if (generation != data.generation) {
    data = {generation};
  }
  return std::make_tuple(&data.stack, &data.string, &data.scope_range_id_stack);
};

void AnnotationStack::PushAnnotation(absl::string_view name) {
  static std::atomic<int64_t> scope_range_counter = 0;

  auto [stack, string, scope_range_id_stack] = GetAnnotationData(generation_);
  stack->push_back(string->size());
  if (!string->empty()) {
    absl::StrAppend(
        string, "::", absl::string_view(name.data(), name.size())  // NOLINT
    );
  } else {
    string->assign(name);
  }
  int64_t scope_range_id =
      scope_range_counter.fetch_add(1, std::memory_order_relaxed) + 1;
  if (TF_PREDICT_FALSE(scope_range_id == 0)) {
    scope_range_id =
        scope_range_counter.fetch_add(1, std::memory_order_relaxed) + 1;
  }
  scope_range_id_stack->push_back(scope_range_id);
}

void AnnotationStack::PopAnnotation() {
  auto [stack, string, scope_range_id_stack] = GetAnnotationData(generation_);
  if (stack->empty()) {
    string->clear();
    scope_range_id_stack->clear();
    return;
  }
  string->resize(stack->back());
  stack->pop_back();
  scope_range_id_stack->pop_back();
}

const string& AnnotationStack::Get() {
  return *std::get<1>(GetAnnotationData(generation_));
}

absl::Span<const int64_t> AnnotationStack::GetScopeRangeIds() {
  return absl::MakeConstSpan(*std::get<2>(GetAnnotationData(generation_)));
}

void AnnotationStack::Enable(bool enable) {
  int generation = generation_.load(std::memory_order_relaxed);
  while (!generation_.compare_exchange_weak(
      generation, enable ? generation | 1 : generation + 1 & ~1,
      std::memory_order_release)) {
  }
}

// AnnotationStack::generation_ implementation must be lock-free for faster
// execution of the ScopedAnnotation API.
std::atomic<int> AnnotationStack::generation_{0};
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");

}  // namespace profiler
}  // namespace tsl
