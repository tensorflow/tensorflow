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

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace profiler {

namespace {

static constexpr uint64_t kRangeIdMask = (0x1ull << 48) - 1;
static constexpr uint64_t kTidMask = ~kRangeIdMask;
static constexpr uint64_t kSharedRangeIdMask = (0x1ull << 62) - 1;
static constexpr uint64_t kSharedTid = 0x2ull << 62;
static constexpr uint64_t kUninitializedTid = 3ull << 62;

// All id with most significant bit value 0..... are id with normal thread
// id, the real thread id is in the following 15 bits, and least significant
// 48 bits are the range id. So the range of normal thread id is [0, 2^15 - 1].
inline bool IsIdWithNormalTId(uint64_t id) { return (id >> 63) == 0; }
inline bool IsNormalTId(uint16_t tid) { return (tid >> 15) == 0; }

// All id with most significant bits value 10..... are id with shared thread
// id. And the least significant 62 bits are the range id shared by all
// threads using this shared thread id prefix.
// All id with most significant bits value 11...... are uninitialized id.
inline bool IsUninitializedId(uint64_t id) { return id >= kUninitializedTid; }

// This will only be called by the threads which do call
// annotation_stack::PushAnnotation(). The TID part in the thread local
// storage will be keeped along with the thread lifetime.
inline uint64_t InitializeUniqueId() {
  // For backward compatibility, we start with thread id 1 instead of 0.
  static std::atomic<uint16_t> next_free_thread_id = 1;
  uint16_t tid = next_free_thread_id.load(std::memory_order_relaxed);
  while (IsNormalTId(tid) && !next_free_thread_id.compare_exchange_weak(
                                 tid, tid + 1, std::memory_order_release)) {
  }
  if (!IsNormalTId(tid)) {
    LOG(WARNING) << "Thread is using shared thread id for scope range id "
                    "generation. Performance may be degraded.";
    return kSharedTid;
  }
  return static_cast<uint64_t>(tid) << 48;
}

inline uint64_t NextUniqueId(uint64_t id) {
  static std::atomic<uint64_t> global_shared_range_id = 0;
  if (IsUninitializedId(id)) {
    id = InitializeUniqueId();
  }
  return IsIdWithNormalTId(id)
             ? ((id + 1) & kRangeIdMask) | (id & kTidMask)
             : ((kSharedRangeIdMask & global_shared_range_id.fetch_add(
                                          1, std::memory_order_release)) |
                kSharedTid);
};

// Returns the annotation data for the given generation.
auto GetAnnotationData(const std::atomic<int>& atomic) {
  static thread_local struct {
    int generation = 0;
    std::vector<size_t> stack;
    std::string string;
    std::vector<int64_t> scope_range_id_stack;
    uint64_t scope_id_last = kUninitializedTid;
  } data{};
  int generation = atomic.load(std::memory_order_acquire);
  if (generation != data.generation) {
    data.generation = generation;
    data.stack.clear();
    data.string.clear();
    data.scope_range_id_stack.clear();
    // Note that data.scope_id_last should be keep here.
  }
  return std::make_tuple(&data.stack, &data.string, &data.scope_range_id_stack,
                         &data.scope_id_last);
};

}  // namespace

void AnnotationStack::PushAnnotation(absl::string_view name) {
  auto [stack, string, scope_range_id_stack, id_last] =
      GetAnnotationData(generation_);
  stack->push_back(string->size());
  if (!string->empty()) {
    absl::StrAppend(
        string, "::", absl::string_view(name.data(), name.size())  // NOLINT
    );
  } else {
    string->assign(name);
  }
  *id_last = NextUniqueId(*id_last);
  scope_range_id_stack->push_back(*id_last);
}

void AnnotationStack::PopAnnotation() {
  auto [stack, string, scope_range_id_stack, _] =
      GetAnnotationData(generation_);
  if (stack->empty()) {
    string->clear();
    scope_range_id_stack->clear();
    return;
  }
  string->resize(stack->back());
  stack->pop_back();
  scope_range_id_stack->pop_back();
}

const std::string& AnnotationStack::Get() {
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
