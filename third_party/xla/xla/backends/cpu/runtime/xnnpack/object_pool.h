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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_OBJECT_POOL_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_OBJECT_POOL_H_

#include <atomic>
#include <cstddef>
#include <memory>

#include "absl/functional/any_invocable.h"

namespace xla::cpu {

// A non-blocking pool of objects of type `T`. Objects in the pool are created
// lazily when needed by calling the user-provided `builder` function.
//
// This object pool is intended to be used on a critical path and optimized for
// zero-allocation in steady state.
template <typename T>
class ObjectPool {
  struct Entry {
    T object;
    std::atomic<Entry*> next;
  };

 public:
  explicit ObjectPool(absl::AnyInvocable<T()> builder, size_t initial_size = 0);
  ~ObjectPool();

  class BorrowedObject {
   public:
    ~BorrowedObject();
    T& operator*() { return entry_->object; }

    BorrowedObject(BorrowedObject&&) = default;
    BorrowedObject& operator=(BorrowedObject&&) = default;

   private:
    friend class ObjectPool;

    BorrowedObject(ObjectPool<T>* parent, std::unique_ptr<Entry> entry);

    ObjectPool<T>* parent_;
    std::unique_ptr<Entry> entry_;
  };

  BorrowedObject GetOrCreate();

 private:
  std::unique_ptr<Entry> CreateEntry();
  std::unique_ptr<Entry> PopEntry();
  void PushEntry(std::unique_ptr<Entry> entry);

  absl::AnyInvocable<T()> builder_;
  std::atomic<Entry*> head_;
};

template <typename T>
ObjectPool<T>::ObjectPool(absl::AnyInvocable<T()> builder, size_t initial_size)
    : builder_(std::move(builder)), head_(nullptr) {
  for (size_t i = 0; i < initial_size; ++i) PushEntry(CreateEntry());
}

template <typename T>
ObjectPool<T>::~ObjectPool() {
  while (Entry* entry = head_.load()) {
    head_.store(entry->next);
    delete entry;
  }
}

template <typename T>
auto ObjectPool<T>::CreateEntry() -> std::unique_ptr<Entry> {
  auto entry = std::make_unique<Entry>();
  entry->object = builder_();
  entry->next = nullptr;
  return entry;
}

template <typename T>
auto ObjectPool<T>::PopEntry() -> std::unique_ptr<Entry> {
  Entry* head = head_.load();
  while (head && !head_.compare_exchange_weak(head, head->next)) {
  }
  return std::unique_ptr<Entry>(head);
}

template <typename T>
void ObjectPool<T>::PushEntry(std::unique_ptr<Entry> entry) {
  Entry* head = head_.load();
  Entry* new_head = entry.release();
  do {
    new_head->next = head;
  } while (!head_.compare_exchange_weak(head, new_head));
}

template <typename T>
ObjectPool<T>::BorrowedObject::BorrowedObject(ObjectPool<T>* parent,
                                              std::unique_ptr<Entry> entry)
    : parent_(parent), entry_(std::move(entry)) {}

template <typename T>
ObjectPool<T>::BorrowedObject::~BorrowedObject() {
  if (parent_ && entry_) parent_->PushEntry(std::move(entry_));
}

template <typename T>
auto ObjectPool<T>::GetOrCreate() -> BorrowedObject {
  if (std::unique_ptr<Entry> entry = PopEntry()) {
    return BorrowedObject(this, std::move(entry));
  }
  return BorrowedObject(this, CreateEntry());
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_OBJECT_POOL_H_
