/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_UTIL_UNIQUE_ANY_H_
#define XLA_TSL_UTIL_UNIQUE_ANY_H_

#include <cstddef>
#include <type_traits>
#include <utility>

#include "absl/log/check.h"

namespace tsl {

// A move-only variant of std::any.
//
// Provides the same interface as std::any but supports types that are
// move-only (not copyable). Uses small-buffer optimization to store
// values of 24 bytes or less inline without heap allocation.
class UniqueAny {
 public:
  constexpr UniqueAny() noexcept = default;
  ~UniqueAny() { reset(); }

  template <typename T, typename Decayed = std::decay_t<T>,
            std::enable_if_t<!std::is_same_v<Decayed, UniqueAny>>* = nullptr>
  UniqueAny(T&& value) {  // NOLINT(google-explicit-constructor)
    emplace<Decayed>(std::forward<T>(value));
  }

  template <typename T, typename... Args>
  explicit UniqueAny(std::in_place_type_t<T>, Args&&... args) {
    emplace<T>(std::forward<Args>(args)...);
  }

  UniqueAny(UniqueAny&& other) noexcept {
    if (other.storage_) {
      other.storage_->MoveTo(this);
      other.storage_ = nullptr;
    }
  }

  UniqueAny& operator=(UniqueAny&& other) noexcept {
    if (this != &other) {
      reset();
      if (other.storage_) {
        other.storage_->MoveTo(this);
        other.storage_ = nullptr;
      }
    }
    return *this;
  }

  template <typename T, typename Decayed = std::decay_t<T>,
            std::enable_if_t<!std::is_same_v<Decayed, UniqueAny>>* = nullptr>
  UniqueAny& operator=(T&& value) {
    reset();
    emplace<Decayed>(std::forward<T>(value));
    return *this;
  }

  template <typename T, typename... Args>
  T& emplace(Args&&... args) {
    reset();
    Storage<T>* s;
    if constexpr (IsSmall<T>()) {
      s = ::new (inline_storage_)
          Storage<T>(std::in_place, std::forward<Args>(args)...);
    } else {
      s = new Storage<T>(std::in_place, std::forward<Args>(args)...);
    }
    storage_ = s;
    return s->value;
  }

  void reset() noexcept {
    if (storage_) {
      if (is_inline()) {
        storage_->~StorageBase();
      } else {
        delete storage_;
      }
      storage_ = nullptr;
    }
  }

  bool has_value() const noexcept { return storage_ != nullptr; }

  void swap(UniqueAny& other) noexcept {
    UniqueAny tmp = std::move(other);
    other = std::move(*this);
    *this = std::move(tmp);
  }

 private:
  // Use small object optimization to keep small objects inline.
  static constexpr size_t kInlineSize = 24;
  static constexpr size_t kInlineAlign = alignof(std::max_align_t);

  template <typename T>
  struct TypeId {
    static const char kId;
  };

  struct StorageBase {
    virtual ~StorageBase() = default;
    virtual const void* type_id() const noexcept = 0;
    virtual void MoveTo(UniqueAny* other) noexcept = 0;
  };

  template <typename T>
  struct Storage final : StorageBase {
    template <typename... Args>
    explicit Storage(std::in_place_t, Args&&... args)
        : value(std::forward<Args>(args)...) {}

    const void* type_id() const noexcept final { return &TypeId<T>::kId; }

    void MoveTo(UniqueAny* other) noexcept final {
      if constexpr (IsSmall<T>()) {
        other->storage_ = ::new (other->inline_storage_)
            Storage(std::in_place, std::move(value));
        this->~Storage();
      } else {
        other->storage_ = this;
      }
    }

    T value;

   private:
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;
  };

  template <typename T>
  static constexpr bool IsSmall() {
    return sizeof(Storage<T>) <= kInlineSize &&
           alignof(Storage<T>) <= kInlineAlign &&
           std::is_nothrow_move_constructible_v<T>;
  }

  bool is_inline() const noexcept {
    return storage_ == reinterpret_cast<const StorageBase*>(inline_storage_);
  }

  // Storage for small object optimized unique any.
  alignas(kInlineAlign) unsigned char inline_storage_[kInlineSize] = {};
  StorageBase* storage_ = nullptr;

  template <typename T>
  friend T* any_cast(UniqueAny*) noexcept;

  template <typename T>
  friend const T* any_cast(const UniqueAny*) noexcept;
};

template <typename T>
const char UniqueAny::TypeId<T>::kId = 0;

template <typename T, typename... Args>
UniqueAny make_unique_any(Args&&... args) {
  return UniqueAny(std::in_place_type<T>, std::forward<Args>(args)...);
}

template <typename T>
T* any_cast(UniqueAny* any) noexcept {
  if (!any || !any->storage_ ||
      any->storage_->type_id() != &UniqueAny::TypeId<T>::kId) {
    return nullptr;
  }
  return &(static_cast<UniqueAny::Storage<T>*>(any->storage_)->value);
}

template <typename T>
const T* any_cast(const UniqueAny* any) noexcept {
  if (!any || !any->storage_ ||
      any->storage_->type_id() != &UniqueAny::TypeId<T>::kId) {
    return nullptr;
  }
  return &(static_cast<const UniqueAny::Storage<T>*>(any->storage_)->value);
}

template <typename T>
T& any_cast(UniqueAny& any) {
  T* p = any_cast<T>(&any);
  CHECK(p) << "any_cast type mismatch";  // Crash OK
  return *p;
}

template <typename T>
const T& any_cast(const UniqueAny& any) {
  const T* p = any_cast<T>(&any);
  CHECK(p) << "any_cast type mismatch";  // Crash OK
  return *p;
}

template <typename T>
T any_cast(UniqueAny&& any) {
  T* p = any_cast<T>(&any);
  CHECK(p) << "any_cast type mismatch";  // Crash OK
  return std::move(*p);
}

}  // namespace tsl

#endif  // XLA_TSL_UTIL_UNIQUE_ANY_H_
