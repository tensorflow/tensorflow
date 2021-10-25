/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_

#ifdef __ANDROID__
#include <dlfcn.h>
#endif

#include <cstdio>
#include <memory>
#include <utility>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

// This class offers a handle to C Standard Library (LibC) shared object
// library on Android. It offers pointers to functions in LibC.
// Fmemopen is available as native API from Android SDK 23 onwards. In order to
// support Android devices from SDK 21 onwards, we load fmemopen dynamically
// from the libc shared object library.
// TODO(b/172544567): Support Apple.
class LibCHandle {
 public:
  // Factory to get an initialised instance of LibCHandle.
  // Loads the libc dynamic library and gets handle to all the
  // required functions. Stores the initialisation status in `status`.
  static LibCHandle Create(Status& status);
  LibCHandle(LibCHandle const&) = delete;
  LibCHandle& operator=(const LibCHandle&) = delete;
  LibCHandle(LibCHandle&& other)
      : libc_(std::exchange(other.libc_, nullptr)),
        fmemopen_(std::exchange(other.fmemopen_, nullptr)) {}
  LibCHandle& operator=(LibCHandle&& other) {
    if (&other != this) {
      CloseHandle();
      libc_ = std::exchange(other.libc_, nullptr);
      fmemopen_ = std::exchange(other.fmemopen_, nullptr);
    }
    return *this;
  }
  ~LibCHandle() { CloseHandle(); }
  // Definition can be found here
  // https://man7.org/linux/man-pages/man3/fmemopen.3.html
  FILE* fmemopen(void* buf, size_t size, const char* mode) const;

 private:
  using FmemopenPtr = FILE* (*)(void*, size_t, const char*);
  LibCHandle(void* libc, FmemopenPtr ptr) : libc_(libc), fmemopen_(ptr) {}
  // Closes the dynamic library loaded in libc_.
  void CloseHandle() {
#ifdef __ANDROID__
    if (libc_ != nullptr) {
      dlclose(libc_);
    }
#endif
  }
  void* libc_ = nullptr;
  FmemopenPtr fmemopen_ = nullptr;
};

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_LIBC_HANDLE_H_
