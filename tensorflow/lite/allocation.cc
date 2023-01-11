/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/allocation.h"

#include <stddef.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

#ifndef TFLITE_MCU
FileCopyAllocation::FileCopyAllocation(const char* filename,
                                       ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kFileCopy) {
  // Obtain the file size using fstat, or report an error if that fails.
  std::unique_ptr<FILE, decltype(&fclose)> file(fopen(filename, "rb"), fclose);
  if (!file) {
    error_reporter_->Report("Could not open '%s'.", filename);
    return;
  }
  struct stat sb;

// support usage of msvc's posix-like fileno symbol
#ifdef _WIN32
#define FILENO(_x) _fileno(_x)
#else
#define FILENO(_x) fileno(_x)
#endif
  if (fstat(FILENO(file.get()), &sb) != 0) {
    error_reporter_->Report("Failed to get file size of '%s'.", filename);
    return;
  }
#undef FILENO
  buffer_size_bytes_ = sb.st_size;
  std::unique_ptr<char[]> buffer(new char[buffer_size_bytes_]);
  if (!buffer) {
    error_reporter_->Report("Malloc of buffer to hold copy of '%s' failed.",
                            filename);
    return;
  }
  size_t bytes_read =
      fread(buffer.get(), sizeof(char), buffer_size_bytes_, file.get());
  if (bytes_read != buffer_size_bytes_) {
    error_reporter_->Report("Read of '%s' failed (too few bytes read).",
                            filename);
    return;
  }
  // Versions of GCC before 6.2.0 don't support std::move from non-const
  // char[] to const char[] unique_ptrs.
  copied_buffer_.reset(const_cast<char const*>(buffer.release()));
}

FileCopyAllocation::~FileCopyAllocation() {}

const void* FileCopyAllocation::base() const { return copied_buffer_.get(); }

size_t FileCopyAllocation::bytes() const { return buffer_size_bytes_; }

bool FileCopyAllocation::valid() const { return copied_buffer_ != nullptr; }
#endif

MemoryAllocation::MemoryAllocation(const void* ptr, size_t num_bytes,
                                   ErrorReporter* error_reporter)
    : Allocation(error_reporter, Allocation::Type::kMemory) {
#ifdef __arm__
  if ((reinterpret_cast<uintptr_t>(ptr) & 0x3) != 0) {
    // The flatbuffer schema has alignment requirements of up to 16 bytes to
    // guarantee that data can be correctly accesses by various backends.
    // Therefore, model pointer should also be 16-bytes aligned to preserve this
    // requirement. But this condition only checks 4-bytes alignment which is
    // the mininum requirement to prevent SIGBUS fault on 32bit ARM. Some models
    // could require 8 or 16 bytes alignment which is not checked yet.
    //
    // Note that 64-bit ARM may also suffer a performance impact, but no crash -
    // that case is not checked.
    TF_LITE_REPORT_ERROR(error_reporter,
                         "The supplied buffer is not 4-bytes aligned");
    buffer_ = nullptr;
    buffer_size_bytes_ = 0;
    return;
  }
#endif  // __arm__

  buffer_ = ptr;
  buffer_size_bytes_ = num_bytes;
}

MemoryAllocation::~MemoryAllocation() {}

const void* MemoryAllocation::base() const { return buffer_; }

size_t MemoryAllocation::bytes() const { return buffer_size_bytes_; }

bool MemoryAllocation::valid() const { return buffer_ != nullptr; }

}  // namespace tflite
