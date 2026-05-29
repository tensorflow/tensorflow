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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

// A pool of objects of type `T`. Objects in the pool are created
// lazily when needed by calling the user-provided `builder` function.
template <typename T, typename... Args>
class ObjectPool {
  struct Entry {
    // Keep `object` as optional to allow using object pool for objects that
    // cannot be default-constructed.
    std::optional<T> object;
    Entry* next = nullptr;
  };

 public:
  explicit ObjectPool(absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder);
  ~ObjectPool();

  class BorrowedObject final {
   public:
    ~BorrowedObject();

    T& operator*() const { return *entry_->object; }
    T* operator->() const { return &*entry_->object; }

    BorrowedObject(BorrowedObject&) = delete;
    BorrowedObject& operator=(BorrowedObject&) = delete;
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
    absl::MutexLock lock(m_);
    return num_created_;
  }

 private:
  absl::StatusOr<std::unique_ptr<Entry>> CreateEntry(Args... args);

  std::unique_ptr<Entry> PopEntry();
  void PushEntry(std::unique_ptr<Entry> entry);

  absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder_;

  Entry* head_ ABSL_GUARDED_BY(m_);
  size_t num_created_ ABSL_GUARDED_BY(m_);
  mutable absl::Mutex m_;
};

template <typename T, typename... Args>
ObjectPool<T, Args...>::ObjectPool(
    absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder)
    : builder_(std::move(builder)), head_(nullptr), num_created_(0) {}

template <typename T, typename... Args>
ObjectPool<T, Args...>::~ObjectPool() {
  int deleted = 0;
  while (Entry* entry = head_) {
    deleted++;
    head_ = entry->next;
    delete entry;
  }
  CHECK_EQ(deleted, num_created_);
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::CreateEntry(Args... args)
    -> absl::StatusOr<std::unique_ptr<Entry>> {
  DCHECK(builder_) << "ObjectPool builder is not initialized";
  auto entry = std::make_unique<Entry>();
  ASSIGN_OR_RETURN(entry->object, builder_(std::forward<Args>(args)...));
  return entry;
}

template <typename T, typename... Args>
auto ObjectPool<T, Args...>::PopEntry() -> std::unique_ptr<Entry> {
  absl::MutexLock lock(m_);
  Entry* old_head = head_;
  if (old_head) {
    head_ = old_head->next;
    old_head->next = nullptr;
  }
  return std::unique_ptr<Entry>(old_head);
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::PushEntry(std::unique_ptr<Entry> entry) {
  absl::MutexLock lock(m_);
  Entry* e = entry.release();
  e->next = head_;
  head_ = e;
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
  ASSIGN_OR_RETURN(auto entry, CreateEntry(std::forward<Args>(args)...));

  absl::MutexLock lock(m_);
  num_created_++;
  return BorrowedObject(this, std::move(entry));
}

}  // namespace xla

#endif  // XLA_RUNTIME_OBJECT_POOL_H_
