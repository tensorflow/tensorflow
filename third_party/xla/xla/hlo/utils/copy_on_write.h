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

#ifndef XLA_HLO_UTILS_COPY_ON_WRITE_H_
#define XLA_HLO_UTILS_COPY_ON_WRITE_H_

#include <memory>
#include <variant>

namespace xla {

// Helper class to maintain a copy-on-write storage of an object of the
// specified type. Logically Variant<MutableOwned, ImmutableShared>.
// The class's purpose is to share (shared_ptr) underlying storage (when it's
// not changed) thus reducing memory footprint.
template <typename T>
class CopyOnWrite {
 public:
  static_assert(!std::is_const_v<T>);

  explicit CopyOnWrite(
      std::variant<std::unique_ptr<T>, std::shared_ptr<const T>> ptr)
      : ownership_(std::move(ptr)), ptr_([&]() -> decltype(ptr_) {
          if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
            return owned->get();
          }
          return std::get<std::shared_ptr<const T>>(ownership_).get();
        }()) {}

  // Obtains a const reference to the read-only copy of the object, could be
  // sharing the storage with other CopyOnWrite<T> instances.
  const T& get() const { return *ptr_; }

  // Obtains a mutable reference to an exclusively owned copy of the object. If
  // the object was sharing storage with other CopyOnWrite<T> instances, make a
  // deep copy inline and transform into exclusively owned copy.
  T& get_mutable() {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      return **owned;
    }
    auto& shared = std::get<std::shared_ptr<const T>>(ownership_);
    DeepCopyToNewUnique(T(*shared));
    return const_cast<T&>(*ptr_);
  }
  // Deep copies the provided value into an exclusively owned copy of the
  // object.
  void set(T&& value) {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      **owned = std::forward<T>(value);
    } else {
      DeepCopyToNewUnique(std::forward<T>(value));
    }
  }
  // If the instance is in MutableOwned state, move the storage into
  // ImmutableShared state.
  // If the instance is in ImmutableShared state, returns the shared storage.
  const std::shared_ptr<const T>& FreezeAndShare() const {
    if (auto* owned = std::get_if<std::unique_ptr<T>>(&ownership_)) {
      ownership_ = std::shared_ptr<const T>(std::move(*owned));
    }
    return std::get<std::shared_ptr<const T>>(ownership_);
  }

 private:
  void DeepCopyToNewUnique(T&& value) {
    auto owned = std::make_unique<T>(std::forward<T>(value));
    ptr_ = owned.get();
    ownership_ = std::move(owned);
  }

  mutable std::variant<std::unique_ptr<T>, std::shared_ptr<const T>> ownership_;
  const T* ptr_;
};

}  // namespace xla

#endif  // XLA_HLO_UTILS_COPY_ON_WRITE_H_
