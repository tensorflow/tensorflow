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

// A lock-free pool of objects of type `T`. Objects in the pool are created
// lazily when needed by calling the user-provided `builder` function.
//
// This object pool is intended to be used on a critical path and optimized for
// zero-allocation in steady state.
//
// ABA problem is avoided by packing a monotonically increasing version counter
// into the upper bits of the head pointer (above the usable virtual address
// space). The version counter increments on every CAS, ensuring that even if a
// pointer value is recycled, the tagged pointer will differ.
template <typename T, typename... Args>
class ObjectPool {
 protected:
  struct alignas(64) Entry {
    // Keep `object` as optional to allow using object pool for objects that
    // cannot be default-constructed.
    std::optional<T> object;
    std::atomic<Entry*> next = nullptr;
  };

 public:
  explicit ObjectPool(absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder);
  ~ObjectPool();

  class BorrowedObject {
   public:
    ~BorrowedObject();

    T& operator*() const { return *entry_->object; }
    T* operator->() const { return &*entry_->object; }

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

 protected:
  // Tagged pointer layout: [hi tag bits | pointer | lo tag bits]
  //
  // We pack a monotonic version counter into bits that are unused in Entry
  // pointers. The tag is split across two regions:
  //
  //  - High bits: above the usable virtual address space. On x86-64 with
  //    5-level paging (LA57), userspace uses up to 57-bit virtual addresses.
  //    This leaves 7 high bits.
  //
  //  - Low bits: Entry is alignas(64) (cache-line aligned), so the low 6
  //    bits of any Entry* are always zero.
  //
  // Combined: 7 + 6 = 13 tag bits (8192 versions). ABA requires the same
  // pointer to be popped and re-pushed 8192 times while a single CAS is in
  // flight. This is a pragmatic choice for XLA where pooled objects (e.g.
  // stream executor events or XNNPACK executors) used sparingly during a single
  // execution and we don't expect millions of push/pop events per second.
  static constexpr int32_t kPtrBits = 57;
  static constexpr int32_t kLowTagBits = 6;

  static constexpr uintptr_t kLowTagMask = (uintptr_t{1} << kLowTagBits) - 1;
  static constexpr uintptr_t kPtrMask =
      ((uintptr_t{1} << kPtrBits) - 1) & ~kLowTagMask;
  static constexpr uintptr_t kTagIncr = uintptr_t{1} << kPtrBits;

  static Entry* GetPtr(uintptr_t tagged) {
    return tsl::safe_reinterpret_cast<Entry*>(tagged & kPtrMask);
  }

  static size_t GetTag(uintptr_t tagged) {
    uintptr_t lo = tagged & kLowTagMask;
    uintptr_t hi = (tagged >> kPtrBits);
    return (hi << kLowTagBits) | lo;
  }

  static uintptr_t PackTagBits(size_t tag) {
    uintptr_t lo = tag & kLowTagMask;
    uintptr_t hi = tag >> kLowTagBits;
    return (hi << kPtrBits) | lo;
  }

  static uintptr_t MakeTagged(Entry* ptr, uintptr_t old_tagged) {
    uintptr_t raw = tsl::safe_reinterpret_cast<uintptr_t>(ptr);
    DCHECK_EQ(raw & ~kPtrMask, 0) << "pointer has non-zero tag bits";
    return raw | PackTagBits(GetTag(old_tagged) + 1);
  }

 private:
  absl::StatusOr<std::unique_ptr<Entry>> CreateEntry(Args... args);

  std::unique_ptr<Entry> PopEntry();
  void PushEntry(std::unique_ptr<Entry> entry);

  absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder_;
  std::atomic<uintptr_t> head_;
  std::atomic<size_t> num_created_;
};

template <typename T, typename... Args>
ObjectPool<T, Args...>::ObjectPool(
    absl::AnyInvocable<absl::StatusOr<T>(Args...)> builder)
    : builder_(std::move(builder)), head_(0), num_created_(0) {
  static_assert(sizeof(uintptr_t) == 8,
                "ObjectPool tagged pointer requires a 64-bit platform");
  static_assert(alignof(Entry) >= (1 << kLowTagBits),
                "Entry alignment must provide enough low tag bits");
}

template <typename T, typename... Args>
ObjectPool<T, Args...>::~ObjectPool() {
  Entry* entry = GetPtr(head_.load(std::memory_order_acquire));
  while (entry) {
    Entry* next = entry->next.load(std::memory_order_relaxed);
    delete entry;
    entry = next;
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
  uintptr_t head = head_.load(std::memory_order_acquire);

  while (Entry* ptr = GetPtr(head)) {
    // Single CAS: swing head to ptr->next with an incremented version counter.
    // If another thread modified head between our load and CAS (push or pop),
    // the version bits will differ and CAS fails — no ABA possible.
    uintptr_t desired =
        MakeTagged(ptr->next.load(std::memory_order_relaxed), head);
    if (ABSL_PREDICT_TRUE(head_.compare_exchange_weak(
            head, desired, std::memory_order_acquire,
            std::memory_order_acquire))) {
      return std::unique_ptr<Entry>(ptr);
    }
    // head is reloaded by compare_exchange_weak on failure.
  }

  return nullptr;
}

template <typename T, typename... Args>
void ObjectPool<T, Args...>::PushEntry(std::unique_ptr<Entry> entry) {
  Entry* new_head = entry.release();
  uintptr_t head = head_.load(std::memory_order_relaxed);

  do {
    new_head->next.store(GetPtr(head), std::memory_order_relaxed);
  } while (ABSL_PREDICT_FALSE(!head_.compare_exchange_weak(
      head, MakeTagged(new_head, head), std::memory_order_release,
      std::memory_order_relaxed)));
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
