/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Utility functions in support of the XRT API.

#ifndef TENSORFLOW_COMPILER_XRT_XRT_REFPTR_H_
#define TENSORFLOW_COMPILER_XRT_XRT_REFPTR_H_

#include <cstddef>

namespace tensorflow {

// Reference counted smart pointer for XRT objects providing the standard
// Ref()/Unref() APIs.
template <typename T>
class RefPtr {
 public:
  RefPtr() = default;
  // Creates a RefPtr from a pointer. This is an ownership transfer operation,
  // and the caller has to own a valid reference to ptr (unless ptr is nullptr).
  RefPtr(T* ptr) : ptr_(ptr) {}  // NOLINT
  RefPtr(const RefPtr& other) : ptr_(other.ptr_) { Acquire(ptr_); }
  RefPtr(RefPtr&& other) : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  ~RefPtr() { Release(ptr_); }

  RefPtr& operator=(const RefPtr& other) {
    if (this != &other) {
      Acquire(other.ptr_);
      Release(ptr_);
      ptr_ = other.ptr_;
    }
    return *this;
  }

  RefPtr& operator=(RefPtr&& other) {
    if (this != &other) {
      Release(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  operator bool() const { return ptr_ != nullptr; }  // NOLINT
  bool operator==(const RefPtr& rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const RefPtr& rhs) const { return ptr_ != rhs.ptr_; }
  bool operator==(const T* ptr) const { return ptr_ == ptr; }
  bool operator!=(const T* ptr) const { return ptr_ != ptr; }
  bool operator==(std::nullptr_t ptr) const { return ptr_ == ptr; }
  bool operator!=(std::nullptr_t ptr) const { return ptr_ != ptr; }

  T* get() const { return ptr_; }

  T* operator->() const {
    CHECK(ptr_ != nullptr);  // Crash OK
    return ptr_;
  }

  T& operator*() const {
    CHECK(ptr_ != nullptr);  // Crash OK
    return *ptr_;
  }

  T* release() {
    T* ptr = ptr_;
    ptr_ = nullptr;
    return ptr;
  }

  // Resets the RefPtr from a pointer. This is an ownership transfer operation,
  // and the caller has to own a valid reference to ptr (unless ptr is nullptr).
  void reset(T* ptr = nullptr) {
    Release(ptr_);
    ptr_ = ptr;
  }

 private:
  static void Release(T* ptr) {
    if (ptr != nullptr) {
      ptr->Unref();
    }
  }

  static void Acquire(T* ptr) {
    if (ptr != nullptr) {
      ptr->Ref();
    }
  }

  T* ptr_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_XRT_XRT_REFPTR_H_
