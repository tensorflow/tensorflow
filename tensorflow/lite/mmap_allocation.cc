/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <fcntl.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {

MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, open(filename, O_RDONLY)) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, dup(fd)) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(MAP_FAILED),
      buffer_size_bytes_(0) {
  if (mmap_fd_ == -1) {
    return;
  }
  struct stat sb;
  fstat(mmap_fd_, &sb);
  buffer_size_bytes_ = sb.st_size;
  mmapped_buffer_ =
      mmap(nullptr, buffer_size_bytes_, PROT_READ, MAP_SHARED, mmap_fd_, 0);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(error_reporter, "Mmap of '%d' failed.", mmap_fd_);
    return;
  }
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
    munmap(const_cast<void*>(mmapped_buffer_), buffer_size_bytes_);
  }
  if (mmap_fd_ != -1) close(mmap_fd_);
}

const void* MMAPAllocation::base() const { return mmapped_buffer_; }

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != MAP_FAILED; }

bool MMAPAllocation::IsSupported() { return true; }

}  // namespace tflite
