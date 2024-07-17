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
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace profiler {

namespace {

constexpr std::string_view kAnnotationDelimiter = "::";

struct StringGeneratorAndView {
  AnnotationStack::Generator generator;
  std::optional<std::string_view> view;

  explicit StringGeneratorAndView(std::function<std::string_view()>&& gen)
      : generator(std::move(gen)) {}

  std::string_view GetStringView() {
    if (!view.has_value()) {
      view = generator();
    }
    return *view;
  }
};

struct AnnotationStackData {
  int generation = 0;
  std::vector<StringGeneratorAndView> stack;
  std::string string;

  void RebuildString() {
    string.clear();
    for (auto& level : stack) {
      auto level_str = level.GetStringView();
      if (level_str.empty()) continue;
      if (!string.empty()) {
        absl::StrAppend(&string, kAnnotationDelimiter, level_str);
      } else {
        string.assign(level_str);
      }
    }
  }

  void AppendStringWithStackTop() {
    if (TF_PREDICT_FALSE(stack.empty())) return;
    std::string_view back_str = stack.back().GetStringView();
    if (TF_PREDICT_FALSE(back_str.empty())) return;
    if (!string.empty()) {
      absl::StrAppend(&string, kAnnotationDelimiter, back_str);
    } else {
      string.assign(back_str);
    }
  }

  void PopStringByStackTop() {
    if (TF_PREDICT_FALSE(stack.empty())) return;
    std::string_view back_str = stack.back().GetStringView();
    if (TF_PREDICT_FALSE(back_str.empty())) return;
    size_t shrink_size = kAnnotationDelimiter.size() + back_str.size();
    size_t remained_size =
        (shrink_size <= string.size() ? (string.size() - shrink_size) : 0);
    string.resize(remained_size);
  }
};

// Returns the annotation data for the given generation.
AnnotationStackData& GetStackData() {
  static thread_local AnnotationStackData data;
  return data;
}

};  // namespace

// Note the life cycle of the name, it must be always valid before the
// annotation is popped.
void AnnotationStack::PushAnnotationGenerator(
    std::function<std::string_view()> generator) {
  auto& data = GetStackData();
  data.stack.emplace_back(std::move(generator));

  int target_generation = generation_.load(std::memory_order_acquire);
  if (target_generation != data.generation) {
    if (target_generation & 1) {
      data.RebuildString();
    }
    data.generation = target_generation;
  } else {
    if (target_generation & 1) {
      data.AppendStringWithStackTop();
    }
  }
}

void AnnotationStack::PopAnnotation() {
  auto& data = GetStackData();
  if (data.stack.empty()) return;
  int target_generation = generation_.load(std::memory_order_acquire);
  if ((target_generation & 1) && target_generation == data.generation) {
    data.PopStringByStackTop();
  }
  data.stack.pop_back();
}

const string& AnnotationStack::Get() {
  auto& data = GetStackData();
  int target_generation = generation_.load(std::memory_order_acquire);
  if ((target_generation & 1) && target_generation != data.generation) {
    data.RebuildString();
    data.generation = target_generation;
  }
  return data.string;
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
