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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_TEST_HELPERS_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_TEST_HELPERS_H_

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <ostream>
#include <random>
#include <string>

#if defined(_MSC_VER)
#include <fcntl.h>
#include <io.h>
#endif

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/file_util.h"

namespace tflite::xnnpack {

inline std::string GenerateRandomString(const size_t size) {
  constexpr char chars[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz";
  // NOLINTNEXTLINE(runtime/random_device)
  std::mt19937 rg{std::random_device{}()};
  std::uniform_int_distribution<std::string::size_type> pick(0,
                                                             sizeof(chars) - 1);
  std::string str(size, 'a');
  std::generate(begin(str), end(str), [&] { return pick(rg); });
  return str;
};

template <class T>
class LightSpan {
 public:
  using value_type = T;

  LightSpan(const void* data, const size_t size)
      : ptr_(reinterpret_cast<T*>(data)), size_(size) {}

  size_t size() const { return size(); }
  const T* begin() const { return ptr_; }
  const T* end() const { return ptr_ + size_; }

  friend std::ostream& operator<<(std::ostream& os, const LightSpan<T>& s) {
    os << '[';
    auto it = s.begin();
    if (it != s.end()) {
      os << +*it;
    }
    ++it;
    for (; it != s.end(); ++it) {
      os << ", " << +*it;
    }
    return os << ']';
  }

 private:
  T* ptr_;
  size_t size_;
};

// Wraps a call to `mkstemp` to create temporary files.
class TempFileDesc : public FileDescriptor {
 public:
  static constexpr struct AutoClose {
  } kAutoClose{};

#if defined(_MSC_VER)
  TempFileDesc() {
    char filename[L_tmpnam_s];
    errno_t err = tmpnam_s(filename, L_tmpnam_s);
    if (err) {
      fprintf(stderr, "Could not create temporary filename.\n");
      std::abort();
    }
    path_ = filename;
    FileDescriptor fd =
        FileDescriptor::Open(path_.c_str(), _O_CREAT | _O_EXCL | _O_RDWR, 0644);
    if (!fd.IsValid()) {
      fprintf(stderr, "Could not create temporary file.\n");
      std::abort();
    }
    Reset(fd.Release());
  }
#else
  TempFileDesc() {
    Reset(mkstemp(path_.data()));
    if (Value() < 0) {
      perror("Could not create temporary file");
    }
  }
#endif

  explicit TempFileDesc(AutoClose) : TempFileDesc() { Close(); }

  TempFileDesc(const TempFileDesc&) = delete;
  TempFileDesc& operator=(const TempFileDesc&) = delete;

  friend void swap(TempFileDesc& a, TempFileDesc& b) {
    std::swap(static_cast<FileDescriptor&>(a), static_cast<FileDescriptor&>(b));
    std::swap(a.path_, b.path_);
  }

  TempFileDesc(TempFileDesc&& other) { swap(*this, other); }
  TempFileDesc& operator=(TempFileDesc&& other) {
    swap(*this, other);
    return *this;
  }

  const std::string& GetPath() const { return path_; }

  const char* GetCPath() const { return path_.c_str(); }

 private:
  std::string path_ = testing::TempDir() + "/weight_cache_test_file.XXXXXX";
};

}  // namespace tflite::xnnpack

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_TEST_HELPERS_H_
