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

#include <cerrno>

#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/core/api/error_reporter.h"

namespace tflite {
namespace {

size_t GetFdSizeBytes(int fd) {
  if (fd < 0) {
    return 0;
  }

  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }

  return fd_stat.st_size;
}

}  // namespace

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

MMAPAllocation::MMAPAllocation(int fd, size_t offset, size_t length,
                               ErrorReporter* error_reporter)
    : MMAPAllocation(error_reporter, dup(fd), offset, length) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd)
    : MMAPAllocation(error_reporter, owned_fd, /*offset=*/0,
                     /*length=*/GetFdSizeBytes(owned_fd)) {}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               size_t offset, size_t length)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(MAP_FAILED),
      buffer_size_bytes_(length) {
  if (owned_fd < 0) {
    return;
  }

#ifdef __ANDROID__
  static int pagesize = getpagesize();
#else
  static int pagesize = sysconf(_SC_PAGE_SIZE);
#endif

  offset_in_buffer_ = offset % pagesize;
  offset_of_buffer_in_file_ = offset - offset_in_buffer_;

  size_t file_size = GetFdSizeBytes(mmap_fd_);
  if (length + offset > file_size) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Asked to mmap '%d' bytes from fd '%d' at offset "
                         "'%d'. This is over the length of file '%d'.",
                         length, mmap_fd_, offset, file_size);
    return;
  }

  mmapped_buffer_ =
      mmap(nullptr, /*__len=*/length + offset_in_buffer_, PROT_READ, MAP_SHARED,
           mmap_fd_, /*__offset=*/offset - offset_in_buffer_);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Mmap of '%d' at offset '%d' failed with error '%d'.",
                         mmap_fd_, offset, errno);
    return;
  }
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
    munmap(const_cast<void*>(mmapped_buffer_),
           buffer_size_bytes_ + offset_in_buffer_);
  }
  if (mmap_fd_ >= 0) {
    close(mmap_fd_);
  }
}

const void* MMAPAllocation::base() const {
  return reinterpret_cast<const void*>(
      reinterpret_cast<const char*>(mmapped_buffer_) + offset_in_buffer_);
}

size_t MMAPAllocation::bytes() const { return buffer_size_bytes_; }

bool MMAPAllocation::valid() const { return mmapped_buffer_ != MAP_FAILED; }

bool MMAPAllocation::IsSupported() { return true; }

}  // namespace tflite
