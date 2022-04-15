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
/// \file
/// Memory management for TF Lite.
#ifndef TENSORFLOW_LITE_ALLOCATION_H_
#define TENSORFLOW_LITE_ALLOCATION_H_

#include <stddef.h>

#include <cstdio>
#include <cstdlib>
#include <memory>

#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

// A memory allocation handle. This could be a mmap or shared memory.
class Allocation {
 public:
  virtual ~Allocation() {}

  enum class Type {
    kMMap,
    kFileCopy,
    kMemory,
  };

  // Base pointer of this allocation
  virtual const void* base() const = 0;
  // Size in bytes of the allocation
  virtual size_t bytes() const = 0;
  // Whether the allocation is valid
  virtual bool valid() const = 0;
  // Return the type of the Allocation.
  Type type() const { return type_; }

 protected:
  Allocation(ErrorReporter* error_reporter, Type type)
      : error_reporter_(error_reporter), type_(type) {}
  ErrorReporter* error_reporter_;

 private:
  const Type type_;
};

// Note that not all platforms support MMAP-based allocation.
// Use `IsSupported()` to check.
class MMAPAllocation : public Allocation {
 public:
  // Loads and maps the provided file to a memory region.
  MMAPAllocation(const char* filename, ErrorReporter* error_reporter);

  // Maps the provided file descriptor to a memory region.
  // Note: The provided file descriptor will be dup'ed for usage; the caller
  // retains ownership of the provided descriptor and should close accordingly.
  MMAPAllocation(int fd, ErrorReporter* error_reporter);

  // Maps the provided file descriptor, with the given offset and length (both
  // in bytes), to a memory region.
  // Note: The provided file descriptor will be dup'ed for usage; the caller
  // retains ownership of the provided descriptor and should close accordingly.
  MMAPAllocation(int fd, size_t offset, size_t length,
                 ErrorReporter* error_reporter);

  virtual ~MMAPAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

  int fd() const { return mmap_fd_; }

  static bool IsSupported();

 protected:
  // Data required for mmap.
  int mmap_fd_ = -1;  // mmap file descriptor
  const void* mmapped_buffer_;
  size_t buffer_size_bytes_ = 0;
  // Used when the address to mmap is not page-aligned.
  size_t offset_in_buffer_ = 0;

 private:
  // Assumes ownership of the provided `owned_fd` instance.
  MMAPAllocation(ErrorReporter* error_reporter, int owned_fd);

  // Assumes ownership of the provided `owned_fd` instance, and uses the given
  // offset and length (both in bytes) for memory mapping.
  MMAPAllocation(ErrorReporter* error_reporter, int owned_fd, size_t offset,
                 size_t length);
};

class FileCopyAllocation : public Allocation {
 public:
  // Loads the provided file into a heap memory region.
  FileCopyAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~FileCopyAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  std::unique_ptr<const char[]> copied_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class MemoryAllocation : public Allocation {
 public:
  // Provides a (read-only) view of the provided buffer region as an allocation.
  // Note: The caller retains ownership of `ptr`, and must ensure it remains
  // valid for the lifetime of the class instance.
  MemoryAllocation(const void* ptr, size_t num_bytes,
                   ErrorReporter* error_reporter);
  virtual ~MemoryAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  const void* buffer_;
  size_t buffer_size_bytes_ = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ALLOCATION_H_
