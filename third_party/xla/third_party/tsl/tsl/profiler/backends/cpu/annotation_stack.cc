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

#include "tsl/profiler/backends/cpu/annotation_stack.h"

#include <atomic>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace profiler {

// Returns the annotation data for the given generation.
static auto GetAnnotationData(const std::atomic<int>& atomic) {
  static thread_local struct {
    int generation = 0;
    std::vector<size_t> stack;
    std::string string;
  } data;
  int generation = atomic.load(std::memory_order_acquire);
  if (generation != data.generation) {
    data = {generation};
  }
  return std::make_pair(&data.stack, &data.string);
};

void AnnotationStack::PushAnnotation(std::string_view name) {
  auto [stack, string] = GetAnnotationData(generation_);
  stack->push_back(string->size());
  if (!string->empty()) {
    return absl::StrAppend(
        string, "::", absl::string_view(name.data(), name.size())  // NOLINT
    );
  }
  string->assign(name);
}

void AnnotationStack::PopAnnotation() {
  auto [stack, string] = GetAnnotationData(generation_);
  if (stack->empty()) {
    return string->clear();
  }
  string->resize(stack->back());
  stack->pop_back();
}

const string& AnnotationStack::Get() {
  return *std::get<std::string*>(GetAnnotationData(generation_));
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
