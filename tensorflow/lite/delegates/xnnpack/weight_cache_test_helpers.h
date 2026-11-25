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
#include <cstdint>
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
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"

#if defined(XNNPACK_MMAP_HAS_VMA_NAME) && \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
#define XNNPACK_TEST_HAS_VMA_NAME 1
#endif

#if INTPTR_MAX == INT64_MAX &&                           \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG) && \
    !defined(__SANITIZE_ADDRESS__) && !defined(__SANITIZE_MEMORY__)
#if defined(__has_feature)
#if !__has_feature(address_sanitizer) && !__has_feature(memory_sanitizer)
#define XNNPACK_TEST_CAN_PROBE_UNMAPPABLE_SIZE 1
#endif
#else
#define XNNPACK_TEST_CAN_PROBE_UNMAPPABLE_SIZE 1
#endif
#endif

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
#include <sys/mman.h>
#include <sys/prctl.h>
#include <unistd.h>

#include <cinttypes>
#include <cstring>
#include <fstream>

#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#endif
#ifndef PR_SET_VMA_ANON_NAME
#define PR_SET_VMA_ANON_NAME 0
#endif
#endif

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
      return;
    }
    path_ = filename;
    FileDescriptor fd =
        FileDescriptor::Open(path_.c_str(), _O_CREAT | _O_EXCL | _O_RDWR, 0644);
    if (!fd.IsValid()) {
      fprintf(stderr, "Could not create temporary file.\n");
      return;
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

#if defined(XNNPACK_TEST_HAS_VMA_NAME)
// Returns the `/proc/self/maps` line of the mapping that holds `addr`.
inline std::string GetMapsLine(const void* addr) {
  const uintptr_t target = reinterpret_cast<uintptr_t>(addr);
  std::ifstream maps("/proc/self/maps");
  for (std::string line; std::getline(maps, line);) {
    uintptr_t start = 0;
    uintptr_t end = 0;
    if (std::sscanf(line.c_str(), "%" SCNxPTR "-%" SCNxPTR, &start, &end) ==
            2 &&
        start <= target && target < end) {
      return line;
    }
  }
  return "";
}

// Probes whether the running kernel supports `PR_SET_VMA_ANON_NAME`.
inline bool AnonVmaNamingSupported() {
  static const bool supported = []() {
    static constexpr char kProbeName[] = "xnnpack-probe";
    const size_t page_size = getpagesize();
    void* const probe =
        mmap(/*addr=*/nullptr, page_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
    if (probe == MAP_FAILED) {
      return false;
    }
    const bool ok =
        prctl(PR_SET_VMA, static_cast<uintptr_t>(PR_SET_VMA_ANON_NAME),
              reinterpret_cast<uintptr_t>(probe),
              static_cast<uintptr_t>(page_size),
              reinterpret_cast<uintptr_t>(kProbeName)) == 0 &&
        std::strstr(GetMapsLine(probe).c_str(), "[anon:xnnpack-probe]") !=
            nullptr;
    munmap(probe, page_size);
    return ok;
  }();
  return supported;
}
#endif

}  // namespace tflite::xnnpack

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_WEIGHT_CACHE_TEST_HELPERS_H_
