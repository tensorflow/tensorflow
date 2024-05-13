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

#ifndef XLA_MAYBE_OWNING_H_
#define XLA_MAYBE_OWNING_H_

#include <cstdint>
#include <memory>

// A unique_ptr like class which may or may not have ownership of its pointer.
// Uses least significant bit of the pointer to indicate ownership.
template <typename T>
class MaybeOwning final {
 public:
  MaybeOwning() = default;
  explicit MaybeOwning(std::unique_ptr<T> unique)
      : ptr_and_owning_bit_(TakeUnique(std::move(unique))) {}

  explicit MaybeOwning(const T* borrowed)
      : ptr_and_owning_bit_(Borrow(borrowed)) {}

  ~MaybeOwning() { MaybeDeleteOwned(); }

  const T* get() const { return RemoveMask(); }

  T* get_mutable() { return RemoveMask(); }

  const T* operator->() const { return get(); }
  const T& operator*() const { return *get(); }

  MaybeOwning<T>& operator=(std::unique_ptr<T> unique) {
    MaybeDeleteOwned();
    ptr_and_owning_bit_ = TakeUnique(std::move(std::move(unique)));
    return *this;
  }

  MaybeOwning& operator=(const T* borrowed) {
    MaybeDeleteOwned();
    ptr_and_owning_bit_ = Borrow(borrowed);
    return *this;
  }

  MaybeOwning& operator=(MaybeOwning&& other) {
    using std::swap;
    swap(ptr_and_owning_bit_, other.ptr_and_owning_bit_);
    return *this;
  }

  MaybeOwning(const MaybeOwning&) = delete;
  MaybeOwning(MaybeOwning&& other)
      : ptr_and_owning_bit_(other.ptr_and_owning_bit_) {
    other.ptr_and_owning_bit_ = 0;
  }

  MaybeOwning Clone() const {
    const T* ptr = get();
    if (ptr && OwnsPtr()) {
      return MaybeOwning(std::make_unique<T>(*ptr));
    }
    return MaybeOwning(ptr);
  }

  bool OwnsPtr() const { return kOwningBitMask & ptr_and_owning_bit_; }

 private:
  enum : uint64_t {
    kOwningBitMask = 1UL,
    kPointerMask = ~kOwningBitMask,
  };

  T* RemoveMask() const {
    return reinterpret_cast<T*>(ptr_and_owning_bit_ & kPointerMask);
  }

  static intptr_t TakeUnique(std::unique_ptr<T> unique) {
    T* released = unique.release();
    DCHECK_EQ(reinterpret_cast<intptr_t>(released) & kOwningBitMask, 0);
    return reinterpret_cast<intptr_t>(released) | kOwningBitMask;
  }

  static intptr_t Borrow(const T* borrowed) {
    DCHECK_EQ(reinterpret_cast<intptr_t>(borrowed) & kOwningBitMask, 0);
    return reinterpret_cast<intptr_t>(borrowed);
  }

  void MaybeDeleteOwned() {
    if (OwnsPtr()) {
      delete get();
    }
  }

  intptr_t ptr_and_owning_bit_ = 0;
};

#endif  // XLA_MAYBE_OWNING_H_
