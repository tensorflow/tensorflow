/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_OBJECT_POOL_H_
#define XLA_RUNTIME_OBJECT_POOL_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

namespace xla {

// A non-blocking pool of objects of type `T`. Objects in the pool are created
// lazily when needed by calling the user-provided `builder` function.
//
// This object pool is intended to be used on a critical path and optimized for
// zero-allocation in steady state.
template <typename T, typename... Args>
class ObjectPool {
  struct Entry {
    // Keep `object` as optional to allow using object pool for objects that
    // cannot be default-constructed.
    std::optional<T> object;
    Entry* next = nullptr;
  };

  // We use pointer tagging for the logical deletion of entries from the linked
  // list to avoid data races on the `entry->next` pointer and to avoid ABA
  // problem. A thread that tries to pop an entry from the pool first tags the
  // entry pointer to get an exclusive access to the entry, concurrent pop
  // operations will wait in the spin loop.
  static constexpr uintptr_t kMask = 0x1;

  static bool IsMarked(Entry* entry) {
    return (tsl::safe_reinterpret_cast<uintptr_t>(entry) & kMask) == kMask;
  }

  static Entry* Mark(Entry* entry) {
    return tsl::safe_reinterpret_cast<Entry*>(
        tsl::safe_reinterpret_cast<uintptr_t>(entry) | kMask);
  }

 public:
  explicit ObjectPool(absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder);
  ~ObjectPool();

  class BorrowedObject {
   public:
    ~BorrowedObject();

    T& operator*() { return *entry_->object; }
    T* operator->() { return &*entry_->object; }

    BorrowedObject(BorrowedObject&&) = default;
    BorrowedObject& operator=(BorrowedObject&&) = default;

   private:
    friend class ObjectPool;

    BorrowedObject(ObjectPool* parent, std::unique_ptr<Entry> entry);

    ObjectPool* parent_;
    std::unique_ptr<Entry> entry_;
  };

  absl::StatusOr<BorrowedObject> GetOrCreate(Args... args);

  size_t num_created() const {
    return num_created_.load(std::memory_order_relaxed);
  }

 private:
  absl::StatusOr<std::unique_ptr<Entry>> CreateEntry(Args... args);

  std::unique_ptr<Entry> PopEntry();
  void PushEntry(std::unique_ptr<Entry> entry);

  absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder_;
  std::atomic<Entry*> head_;
  std::atomic<size_t> num_created_;
};

template <typename T, typename... Args>
ObjectPool<T, Args...>::ObjectPool(
    absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder)
    : builder_(std::move(builder)), head_(nullptr), num_created_(0) {}

template <typename T, typename... Args>
ObjectPool<T, Args...>::~ObjectPool() {
  while (Entry* entry = head_.load(std::memory_order_acquire)) {
    head_.store(entry->next, std::memory_order_relaxed);
    delete entry;
  }
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::CreateEntry(Args... args)
    -> absl::StatusOr<std::unique_ptr<Entry>> {
  DCHECK(builder_) << "ObjectPool builder is not initialized";
  auto entry = std::make_unique<Entry>();
  TF_ASSIGN_OR_RETURN(entry->object, builder_(std::forward<Args>(args)...));
  num_created_.fetch_add(1, std::memory_order_relaxed);
  return entry;
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::PopEntry() -> std::unique_ptr<Entry> {
  Entry* head = head_.load(std::memory_order_relaxed);

  // Try to mark the entry at head for deletion with a CAS operation.
  while (head &&
         (IsMarked(head) || !head_.compare_exchange_weak(
                                head, Mark(head), std::memory_order_acquire,
                                std::memory_order_relaxed))) {
    if (ABSL_PREDICT_FALSE(IsMarked(head))) {
      head = head_.load(std::memory_order_relaxed);
    }
  }

  // Object pool is empty.
  if (ABSL_PREDICT_FALSE(head == nullptr)) {
    return nullptr;
  }

  // Update head pointer to the next entry.
  head_.store(head->next, std::memory_order_relaxed);

  return std::unique_ptr<Entry>(head);
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::PushEntry(std::unique_ptr<Entry> entry) {
  Entry* new_head = entry.release();
  new_head->next = head_.load(std::memory_order_relaxed);
  while (IsMarked(new_head->next) ||
         !head_.compare_exchange_weak(new_head->next, new_head,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
    if (ABSL_PREDICT_FALSE(IsMarked(new_head->next))) {
      new_head->next = head_.load(std::memory_order_relaxed);
    }
  }
}

template <typename T, typename... Args>
ObjectPool<T, Args...>::BorrowedObject::BorrowedObject(
    ObjectPool<T, Args...>* parent, std::unique_ptr<Entry> entry)
    : parent_(parent), entry_(std::move(entry)) {}

template <typename T, typename... Args>
ObjectPool<T, Args...>::BorrowedObject::~BorrowedObject() {
  if (ABSL_PREDICT_TRUE(parent_ && entry_)) {
    parent_->PushEntry(std::move(entry_));
  }
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::GetOrCreate(Args... args)
    -> absl::StatusOr<BorrowedObject> {
  if (std::unique_ptr<Entry> entry = PopEntry(); ABSL_PREDICT_TRUE(entry)) {
    return BorrowedObject(this, std::move(entry));
  }
  TF_ASSIGN_OR_RETURN(auto entry, CreateEntry(std::forward<Args>(args)...));
  return BorrowedObject(this, std::move(entry));
}

}  // namespace xla

#endif  // XLA_RUNTIME_OBJECT_POOL_H_
