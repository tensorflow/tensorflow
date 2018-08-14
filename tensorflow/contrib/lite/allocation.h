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
// Main abstraction controlling the tflite interpreter.
// See context.h for the API for defining operations (TfLiteRegistration).
#ifndef TENSORFLOW_CONTRIB_LITE_ALLOCATION_H_
#define TENSORFLOW_CONTRIB_LITE_ALLOCATION_H_

#include <cstdio>
#include <cstdlib>
#include <vector>
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/error_reporter.h"
#include "tensorflow/contrib/lite/simple_memory_arena.h"
#include "tensorflow/contrib/lite/string.h"

namespace tflite {

// A memory allocation handle. This could be a mmap or shared memory.
class Allocation {
 public:
  Allocation(ErrorReporter* error_reporter) : error_reporter_(error_reporter) {}
  virtual ~Allocation() {}

  // Base pointer of this allocation
  virtual const void* base() const = 0;
  // Size in bytes of the allocation
  virtual size_t bytes() const = 0;
  // Whether the allocation is valid
  virtual bool valid() const = 0;

 protected:
  ErrorReporter* error_reporter_;
};

class MMAPAllocation : public Allocation {
 public:
  MMAPAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~MMAPAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

  static bool IsSupported();

 protected:
  // Data required for mmap.
  int mmap_fd_ = -1;  // mmap file descriptor
  const void* mmapped_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class FileCopyAllocation : public Allocation {
 public:
  FileCopyAllocation(const char* filename, ErrorReporter* error_reporter);
  virtual ~FileCopyAllocation();
  const void* base() const override;
  size_t bytes() const override;
  bool valid() const override;

 private:
  // Data required for mmap.
  std::unique_ptr<const char[]> copied_buffer_;
  size_t buffer_size_bytes_ = 0;
};

class MemoryAllocation : public Allocation {
 public:
  // Allocates memory with the pointer and the number of bytes of the memory.
  // The pointer has to remain alive and unchanged until the destructor is
  // called.
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

#endif  // TENSORFLOW_CONTRIB_LITE_ALLOCATION_H_
