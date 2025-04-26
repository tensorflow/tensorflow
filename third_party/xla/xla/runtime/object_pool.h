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
#include <memory>
#include <optional>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/statusor.h"

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
    std::atomic<Entry*> next;
  };

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

  size_t num_created() const { return num_created_.load(); }

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
  while (Entry* entry = head_.load()) {
    head_.store(entry->next);
    delete entry;
  }
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::CreateEntry(Args... args)
    -> absl::StatusOr<std::unique_ptr<Entry>> {
  auto entry = std::make_unique<Entry>();
  TF_ASSIGN_OR_RETURN(entry->object, builder_(std::forward<Args>(args)...));
  entry->next = nullptr;
  num_created_.fetch_add(1);
  return entry;
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::PopEntry() -> std::unique_ptr<Entry> {
  Entry* head = head_.load();
  while (head && !head_.compare_exchange_weak(head, head->next)) {
  }
  return std::unique_ptr<Entry>(head);
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::PushEntry(std::unique_ptr<Entry> entry) {
  Entry* head = head_.load();
  Entry* new_head = entry.release();
  do {
    new_head->next = head;
  } while (!head_.compare_exchange_weak(head, new_head));
}

template <typename T, typename... Args>
ObjectPool<T, Args...>::BorrowedObject::BorrowedObject(
    ObjectPool<T, Args...>* parent, std::unique_ptr<Entry> entry)
    : parent_(parent), entry_(std::move(entry)) {}

template <typename T, typename... Args>
ObjectPool<T, Args...>::BorrowedObject::~BorrowedObject() {
  if (parent_ && entry_) {
    parent_->PushEntry(std::move(entry_));
  }
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::GetOrCreate(Args... args)
    -> absl::StatusOr<BorrowedObject> {
  if (std::unique_ptr<Entry> entry = PopEntry()) {
    return BorrowedObject(this, std::move(entry));
  }
  TF_ASSIGN_OR_RETURN(auto entry, CreateEntry(std::forward<Args>(args)...));
  return BorrowedObject(this, std::move(entry));
}

}  // namespace xla

#endif  // XLA_RUNTIME_OBJECT_POOL_H_
