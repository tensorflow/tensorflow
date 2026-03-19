/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_REF_WRAPPER_H_
#define XLA_PYTHON_IFRT_REF_WRAPPER_H_

#include <cstddef>
#include <utility>

#include "absl/strings/str_cat.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {
namespace ifrt {

// RCReferenceWrapper<T> behaves like tsl::RCReference<T>, but forwards a small
// number of methods (comparison, hash, stringify) to the referenced value of
// tsl::RCReference<T>. This forwarding facilitates using tsl::RCReference<T>
// with Abseil containers and logging.
template <typename T>
class RCReferenceWrapper {
 public:
  RCReferenceWrapper() = default;

  explicit RCReferenceWrapper(tsl::RCReference<T> ref) noexcept
      : ref_(std::move(ref)) {}

  template <typename U>
  explicit RCReferenceWrapper(tsl::RCReference<U> ref) noexcept
      : ref_(std::move(ref)) {}

  RCReferenceWrapper(const RCReferenceWrapper& other) = default;
  RCReferenceWrapper(RCReferenceWrapper&& other) noexcept = default;

  // Support implicit conversion from RCReferenceWrapper<Derived> to
  // RCReferenceWrapper<Base>.
  template <typename U>
  RCReferenceWrapper(const RCReferenceWrapper<U>& ref)  // NOLINT
      : ref_(ref.ref_) {}
  template <typename U>
  RCReferenceWrapper(RCReferenceWrapper<U>&& ref) noexcept  // NOLINT
      : ref_(std::move(ref.ref_)) {}

  void reset(T* pointer = nullptr) { ref_.reset(pointer); }
  T* release() { return ref_.release(); }
  void swap(RCReferenceWrapper& other) noexcept { std::swap(ref_, other.ref_); }

  RCReferenceWrapper& operator=(const RCReferenceWrapper& other) = default;
  RCReferenceWrapper& operator=(RCReferenceWrapper&& other) = default;

  template <typename U>
  RCReferenceWrapper& operator=(const RCReferenceWrapper<U>& other) {
    ref_ = other.ref_;
    return *this;
  }
  template <typename U>
  RCReferenceWrapper& operator=(RCReferenceWrapper<U>&& other) {
    ref_ = std::move(other.ref_);
    return *this;
  }

  T* get() const { return ref_.get(); }
  T& operator*() const { return *ref_; }
  T* operator->() const { return ref_.get(); }
  explicit operator bool() const { return static_cast<bool>(ref_); }

  bool operator==(const RCReferenceWrapper& other) const {
    return ref_ == other.ref_ ||
           (ref_ != nullptr && other.ref_ != nullptr && *ref_ == *other.ref_);
  }

  bool operator!=(const RCReferenceWrapper& other) const {
    return !(*this == other);
  }

  template <typename U>
  bool operator==(const RCReferenceWrapper<U>& other) const {
    return ref_ == other.ref_ ||
           (ref_ != nullptr && other.ref_ != nullptr && *ref_ == *other.ref_);
  }

  template <typename U>
  bool operator!=(const RCReferenceWrapper<U>& other) const {
    return !(*this == other);
  }

  friend bool operator==(const RCReferenceWrapper& wrapper, std::nullptr_t) {
    return wrapper.ref_ == nullptr;
  }

  friend bool operator==(std::nullptr_t, const RCReferenceWrapper& wrapper) {
    return wrapper.ref_ == nullptr;
  }

  friend bool operator!=(const RCReferenceWrapper& wrapper, std::nullptr_t) {
    return wrapper.ref_ != nullptr;
  }

  friend bool operator!=(std::nullptr_t, const RCReferenceWrapper& wrapper) {
    return wrapper.ref_ != nullptr;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const RCReferenceWrapper& wrapper) {
    if (wrapper.ref_ == nullptr) {
      sink.Append("<nullptr>");
    } else {
      sink.Append(absl::StrCat(*wrapper.ref_));
    }
  }

  template <typename H>
  friend H AbslHashValue(H h, const RCReferenceWrapper& wrapper) {
    if (wrapper.ref_ == nullptr) {
      return H::combine(std::move(h), nullptr);
    } else {
      return H::combine(std::move(h), *wrapper.ref_);
    }
  }

 private:
  // For copy and move constructors and operators.
  template <typename U>
  friend class RCReferenceWrapper;

  tsl::RCReference<T> ref_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_REF_WRAPPER_H_
