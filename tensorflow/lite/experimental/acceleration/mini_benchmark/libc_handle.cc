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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/libc_handle.h"

#ifdef __ANDROID__
#include <dlfcn.h>
#endif
#include <stdio.h>

#include "tensorflow/lite/experimental/acceleration/mini_benchmark/decode_jpeg_status.h"

namespace tflite {
namespace acceleration {
namespace decode_jpeg_kernel {

LibCHandle LibCHandle::Create(Status &status) {
#ifndef __ANDROID__
#ifndef _WIN32
  // Use the statically linked C lib.
  return LibCHandle(nullptr, ::fmemopen);
#else   // _WIN32
  status = {kTfLiteError, "Windows not supported."};
  return LibCHandle(nullptr, nullptr);
#endif  // !_WIN32
#else   // __ANDROID__
  void *libc = nullptr;
  FmemopenPtr fmemopen_ptr = nullptr;
  if (!(libc = dlopen("libc.so", RTLD_NOW | RTLD_LOCAL))) {
    status = {kTfLiteError,
              "Failed to load the libc dynamic shared object library."};
    return LibCHandle(nullptr, nullptr);
  }

  if (!(fmemopen_ptr =
            reinterpret_cast<FmemopenPtr>(dlsym(libc, "fmemopen")))) {
    status = {kTfLiteError, "Failed to dynamically load the method: fmemopen"};
    return LibCHandle(nullptr, nullptr);
  }
  status = {kTfLiteOk, ""};
  return LibCHandle(libc, fmemopen_ptr);
#endif  // !__ANDROID__
}

FILE *LibCHandle::fmemopen(void *buf, size_t size, const char *mode) const {
  return fmemopen_(buf, size, mode);
}

}  // namespace decode_jpeg_kernel
}  // namespace acceleration
}  // namespace tflite
