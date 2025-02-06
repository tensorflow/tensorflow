/* Copyright 2022 Google LLC. All Rights Reserved.

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

#ifndef XLA_TSL_CONCURRENCY_REF_COUNT_H_
#define XLA_TSL_CONCURRENCY_REF_COUNT_H_

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace tsl {

namespace internal {
// TODO(ezhulenev): Replace with C++20 concept when available.
// https://en.cppreference.com/w/cpp/concepts/derived_from
template <typename Derived, typename Base>
using DerivedFrom = typename std::enable_if_t<std::is_base_of_v<Base, Derived>>;
}  // namespace internal

#ifndef NDEBUG
inline std::atomic<size_t> total_reference_counted_objects;

// Return the total number of reference-counted objects that are currently
// live in the process.  This is intended for debugging/assertions only, and
// shouldn't be used for mainline logic in the runtime.
inline size_t GetNumReferenceCountedObjects() {
  return total_reference_counted_objects.load(std::memory_order_relaxed);
}
inline void AddNumReferenceCountedObjects() {
  total_reference_counted_objects.fetch_add(1, std::memory_order_relaxed);
}
inline void DropNumReferenceCountedObjects() {
  total_reference_counted_objects.fetch_sub(1, std::memory_order_relaxed);
}
#else
inline void AddNumReferenceCountedObjects() {}
inline void DropNumReferenceCountedObjects() {}
#endif

// This class is a common base class for things that need an atomic reference
// count for ownership management.
//
// Subclasses of this are allowed to implement a Destroy() instance method,
// which allows custom allocation/deallocation logic.
//
// This class intentionally doesn't have a virtual destructor or anything else
// that would require a vtable, but subclasses can have one if they choose.
template <typename SubClass>
class ReferenceCounted {
 public:
  ReferenceCounted() : ReferenceCounted(1) {}
  explicit ReferenceCounted(unsigned ref_count) : ref_count_(ref_count) {
    AddNumReferenceCountedObjects();
  }

  ~ReferenceCounted() {
    assert(ref_count_.load() == 0 &&
           "Shouldn't destroy a reference counted object with references!");
    DropNumReferenceCountedObjects();
  }

  // Not copyable or movable.
  ReferenceCounted(const ReferenceCounted&) = delete;
  ReferenceCounted& operator=(const ReferenceCounted&) = delete;

  // Add a new reference to this object.
  void AddRef() {
    assert(ref_count_.load(std::memory_order_relaxed) >= 1);
    // It is OK to use std::memory_order_relaxed here as it does not affect the
    // ownership state of the object.
    ref_count_.fetch_add(1, std::memory_order_relaxed);
  }

  // Drop a reference to this object, potentially deallocating it.
  void DropRef() {
    assert(ref_count_.load(std::memory_order_relaxed) > 0);

    // If ref_count_==1, this object is owned only by the caller. Bypass a
    // locked op in that case.
    if (ref_count_.load(std::memory_order_acquire) == 1 ||
        ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // Make assert in ~ReferenceCounted happy
      assert((ref_count_.store(0, std::memory_order_relaxed), true));
      static_cast<SubClass*>(this)->Destroy();
    }
  }

  // Return reference count. This should be used for testing and debugging only.
  uint32_t NumRef() const { return ref_count_.load(); }

  // Return true if reference count is 1.
  bool IsUnique() const {
    return ref_count_.load(std::memory_order_acquire) == 1;
  }

 protected:
  // Subclasses are allowed to customize this, but the default implementation of
  // Destroy() just deletes the pointer.
  void Destroy() { delete static_cast<SubClass*>(this); }

 private:
  std::atomic<unsigned> ref_count_;
};

// This is a smart pointer that keeps the specified reference counted value
// around.
template <typename T>
class RCReference {
 public:
  RCReference() : pointer_(nullptr) {}

  RCReference(RCReference&& other) noexcept : pointer_(other.pointer_) {
    other.pointer_ = nullptr;
  }

  RCReference(const RCReference& other) : pointer_(other.pointer_) {
    if (pointer_) pointer_->AddRef();
  }

  RCReference& operator=(RCReference&& other) noexcept {
    reset(other.pointer_);
    other.pointer_ = nullptr;
    return *this;
  }

  RCReference& operator=(const RCReference& other) {
    reset(other.pointer_);
    if (pointer_) pointer_->AddRef();
    return *this;
  }

  // Support implicit conversion from RCReference<Derived> to RCReference<Base>.
  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  RCReference(RCReference<Derived>&& u) : pointer_(u.pointer_) {  // NOLINT
    u.pointer_ = nullptr;
  }
  template <typename Derived, internal::DerivedFrom<Derived, T>* = nullptr>
  RCReference(const RCReference<Derived>& u) : pointer_(u.pointer_) {  // NOLINT
    if (pointer_) pointer_->AddRef();
  }

  ~RCReference() {
    if (pointer_ != nullptr) pointer_->DropRef();
  }

  void reset(T* pointer = nullptr) {
    if (pointer_ != nullptr) pointer_->DropRef();
    pointer_ = pointer;
  }

  T* release() {
    T* tmp = pointer_;
    pointer_ = nullptr;
    return tmp;
  }

  T& operator*() const {
    assert(pointer_ && "null RCReference");
    return *pointer_;
  }

  T* operator->() const {
    assert(pointer_ && "null RCReference");
    return pointer_;
  }

  // Return a raw pointer.
  T* get() const { return pointer_; }

  // Make an explicit copy of this RCReference, increasing the refcount by one.
  [[deprecated("Use copy constructor instead.")]] RCReference CopyRef() const;

  explicit operator bool() const { return pointer_ != nullptr; }

  void swap(RCReference& other) noexcept {
    using std::swap;
    swap(pointer_, other.pointer_);
  }

  bool operator==(const RCReference& ref) const {
    return pointer_ == ref.pointer_;
  }
  bool operator!=(const RCReference& ref) const {
    return pointer_ != ref.pointer_;
  }

  friend bool operator==(const RCReference& ref, std::nullptr_t) {
    return ref.pointer_ == nullptr;
  }
  friend bool operator==(std::nullptr_t, const RCReference& ref) {
    return ref.pointer_ == nullptr;
  }
  friend bool operator!=(const RCReference& ref, std::nullptr_t) {
    return ref.pointer_ != nullptr;
  }
  friend bool operator!=(std::nullptr_t, const RCReference& ref) {
    return ref.pointer_ != nullptr;
  }

  template <typename R>
  friend RCReference<R> FormRef(R*);
  template <typename R>
  friend RCReference<R> TakeRef(R*);

 private:
  T* pointer_;

  template <typename R>
  friend class RCReference;
};

// Add a new reference to the specified pointer.
template <typename T>
RCReference<T> FormRef(T* pointer) {
  RCReference<T> ref;
  ref.pointer_ = pointer;
  pointer->AddRef();
  return ref;
}

// Return an RCReference for the specified object and *takes ownership* of a
// +1 reference.  When destroyed, this will drop the reference.
template <typename T>
RCReference<T> TakeRef(T* pointer) {
  RCReference<T> ref;
  ref.pointer_ = pointer;
  return ref;
}

template <typename T>
RCReference<T> RCReference<T>::CopyRef() const {
  if (!pointer_) return RCReference();
  return FormRef(get());
}

// Create a new reference counted object, similar to std::make_shared.
template <typename T, typename... Args>
RCReference<T> MakeRef(Args&&... args) {
  auto t = new T(std::forward<Args>(args)...);
  return TakeRef(t);
}
// For ADL style swap.
template <typename T>
void swap(RCReference<T>& a, RCReference<T>& b) noexcept {
  a.swap(b);
}

}  // namespace tsl

#endif  // XLA_TSL_CONCURRENCY_REF_COUNT_H_
