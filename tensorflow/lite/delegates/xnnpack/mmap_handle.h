/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_MMAP_HANDLE_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_MMAP_HANDLE_H_

#if defined(_MSC_VER)
#include <windows.h>
#endif

#include <cstddef>
#include <cstdint>
#include <utility>

#include "tensorflow/lite/delegates/xnnpack/file_util.h"

namespace tflite::xnnpack {

// Calls the provided callback at then end of the scope this was created into.
template <class F>
class ScopeGuard {
 public:
  explicit ScopeGuard(F&& callback) : callback_(std::forward<F>(callback)) {}
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&& other)
      : active_(other.active_), callback_(std::move(other.callback_)) {
    other.Deactivate();
  }
  ScopeGuard& operator=(ScopeGuard&& other) {
    if (this != &other) {
      active_ = std::move(other.active_);
      callback_ = std::move(other.callback_);
      other.Deactivate();
    }
  }

  ~ScopeGuard() {
    if (active_) {
      callback_();
    }
  }

  void Deactivate() { active_ = false; }

 private:
  F callback_;
  bool active_ = true;
};

template <class F>
ScopeGuard(F&&) -> ScopeGuard<F>;

// Handles MMap allocations lifetime.
//
// When mapped, provides a view over the allocation for convenience.
//
// WARNING: the interface in this file is still under experimentation and WILL
// CHANGE. Do not rely on it.
class MMapHandle {
 public:
  using value_type = uint8_t;

  MMapHandle() = default;
  ~MMapHandle();
  MMapHandle(const MMapHandle&) = delete;
  MMapHandle& operator=(const MMapHandle&) = delete;
  MMapHandle(MMapHandle&&);
  MMapHandle& operator=(MMapHandle&&);

  // Maps the file at the given path.
  [[nodiscard /*Mapping a file can fail.*/]]
  bool Map(const char* path, size_t offset = 0);

  // Maps the fd associated to the file descriptor.
  //
  // The debug_path is printed along the error messages.
  [[nodiscard /*Mapping a file can fail.*/]]
  bool Map(const FileDescriptorView& fd, size_t offset = 0,
           const char* debug_path = nullptr);

  // Tries to resize the current mapping.
  //
  // Only succeeds if the mapping could be resized without being moved.
  //
  // WARNING: expects `IsMapped()` to be true.
  [[nodiscard /*Resizing a file can fail.*/]]
  bool Resize(size_t new_size);

  // Unmaps an existing mapping.
  void UnMap();

  // Returns true if a mapping exists.
  bool IsMapped() const { return data_ != nullptr; }

  // Returns the mapping buffer.
  uint8_t* data() { return data_ + offset_page_adjustment_; }

  // Returns the mapping buffer.
  const uint8_t* data() const { return data_ + offset_page_adjustment_; }

  // Returns the mapping size in bytes.
  size_t size() const { return size_; }

  size_t offset() const { return offset_; }

  uint8_t* begin() { return data(); }

  const uint8_t* begin() const { return data(); }

  uint8_t* end() { return data() + size(); }

  const uint8_t* end() const { return data() + size(); }

  friend void swap(MMapHandle& a, MMapHandle& b);

 private:
  size_t size_ = 0;
  size_t offset_ = 0;
  size_t offset_page_adjustment_ = 0;
  uint8_t* data_ = nullptr;
#if defined(_MSC_VER)
  HANDLE file_mapping_ = 0;
#endif
};

}  // namespace tflite::xnnpack

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_MMAP_HANDLE_H_
