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

#include <stddef.h>

#if defined(_WIN32)
#include <windows.h>
// <windows.h> must precede the CRT headers below on some toolchains.
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif  // defined(_WIN32)

#include <cerrno>
#include <cstdio>

#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/lite/util.h"

#if defined(_WIN32)
// Windows has no <sys/mman.h> and therefore no MAP_FAILED. Reuse the POSIX
// sentinel so the platform-independent members below (e.g. `valid()`) can be
// shared across platforms. `MapViewOfFile` reports failure by returning NULL,
// which is translated to this sentinel at the mapping site.
#ifndef MAP_FAILED
#define MAP_FAILED (reinterpret_cast<void*>(-1))
#endif  // MAP_FAILED
#endif  // defined(_WIN32)

namespace tflite {
namespace {

int OpenFileReadOnly(const char* filename) {
#if defined(_WIN32)
  return _open(filename, _O_RDONLY | _O_BINARY);
#else
  return open(filename, O_RDONLY);
#endif  // defined(_WIN32)
}

// Note: On Windows, calling _dup with an invalid fd (like -1) crashes the
// process via the CRT invalid parameter handler. Guarding against negative fds
// avoids this, but callers must still ensure that any positive fd passed is
// valid.
int DuplicateFd(int fd) {
  if (fd < 0) {
    return -1;
  }
#if defined(_WIN32)
  return _dup(fd);
#else
  return dup(fd);
#endif  // defined(_WIN32)
}

void CloseFd(int fd) {
#if defined(_WIN32)
  _close(fd);
#else
  close(fd);
#endif  // defined(_WIN32)
}

size_t GetFdSizeBytes(int fd) {
  if (fd < 0) {
    return 0;
  }

#if defined(_WIN32)
  struct _stat64 fd_stat;
  if (_fstat64(fd, &fd_stat) != 0) {
    return 0;
  }
#else
  struct stat fd_stat;
  if (fstat(fd, &fd_stat) != 0) {
    return 0;
  }
#endif  // defined(_WIN32)

  return fd_stat.st_size;
}

// Returns the granularity that memory-map offsets must be aligned to. On POSIX
// this is the page size; on Windows, `MapViewOfFile` requires the file offset
// to be a multiple of the (typically larger) allocation granularity.
size_t GetMmapOffsetGranularity() {
#if defined(_WIN32)
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  return system_info.dwAllocationGranularity;
#else
  return sysconf(_SC_PAGE_SIZE);
#endif  // defined(_WIN32)
}

}  // namespace

MMAPAllocation::MMAPAllocation(const char* filename,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, OpenFileReadOnly(filename), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, DuplicateFd(fd), map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(const char* filename, size_t offset,
                               size_t length, ErrorReporter* error_reporter,
                               bool map_private)
    : MMAPAllocation(error_reporter, OpenFileReadOnly(filename), offset, length,
                     map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Could not open '%s'.", filename);
  }
}

MMAPAllocation::MMAPAllocation(int fd, size_t offset, size_t length,
                               ErrorReporter* error_reporter, bool map_private)
    : MMAPAllocation(error_reporter, DuplicateFd(fd), offset, length,
                     map_private) {
  if (mmap_fd_ == -1) {
    TF_LITE_REPORT_ERROR(error_reporter, "Failed to dup '%d' file descriptor.",
                         fd);
  }
}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               bool map_private)
    : MMAPAllocation(error_reporter, owned_fd, /*offset=*/0,
                     /*length=*/GetFdSizeBytes(owned_fd), map_private) {}

MMAPAllocation::MMAPAllocation(ErrorReporter* error_reporter, int owned_fd,
                               size_t offset, size_t length, bool map_private)
    : Allocation(error_reporter, Allocation::Type::kMMap),
      mmap_fd_(owned_fd),
      mmapped_buffer_(MAP_FAILED),
      buffer_size_bytes_(length) {
  if (owned_fd < 0) {
    return;
  }

  // On Windows, MapViewOfFile with size 0 maps the entire file, which would
  // make the allocation valid when 0 size was explicitly requested. POSIX
  // mmap with size 0 fails (typically with EINVAL). Enforce failure for 0
  // size on all platforms for consistency.
  if (length == 0) {
    TF_LITE_REPORT_ERROR(error_reporter, "mmap of size 0 requested.");
    return;
  }

  const size_t offset_granularity = GetMmapOffsetGranularity();
  offset_in_buffer_ = offset % offset_granularity;
  offset_of_buffer_in_file_ = offset - offset_in_buffer_;

  size_t file_size = GetFdSizeBytes(mmap_fd_);
  CheckedInt<size_t> checked_length_offset =
      CheckedInt<size_t>(length) + offset;
  if (checked_length_offset.Overflow() ||
      checked_length_offset.Value() > file_size) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Asked to mmap '%d' bytes from fd '%d' at offset "
                         "'%d'. This is over the length of file '%d'.",
                         length, mmap_fd_, offset, file_size);
    return;
  }

#if defined(_WIN32)
  HANDLE file_handle = reinterpret_cast<HANDLE>(_get_osfhandle(mmap_fd_));
  if (file_handle == INVALID_HANDLE_VALUE) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Failed to get file handle from fd '%d'.", mmap_fd_);
    return;
  }

  // A read-only file descriptor supports both a read-only (shared) mapping and
  // a copy-on-write (writeable, private) mapping.
  HANDLE file_mapping = ::CreateFileMappingA(
      file_handle, /*lpFileMappingAttributes=*/nullptr,
      /*flProtect=*/map_private ? PAGE_WRITECOPY : PAGE_READONLY,
      /*dwMaximumSizeHigh=*/0, /*dwMaximumSizeLow=*/0, /*lpName=*/nullptr);
  if (file_mapping == nullptr) {
    TF_LITE_REPORT_ERROR(
        error_reporter, "CreateFileMapping of fd '%d' failed with error '%lu'.",
        mmap_fd_, ::GetLastError());
    return;
  }

  const uint64_t map_offset = offset_of_buffer_in_file_;
  mmapped_buffer_ = ::MapViewOfFile(
      file_mapping,
      /*dwDesiredAccess=*/map_private ? FILE_MAP_COPY : FILE_MAP_READ,
      /*dwFileOffsetHigh=*/static_cast<DWORD>(map_offset >> 32),
      /*dwFileOffsetLow=*/static_cast<DWORD>(map_offset & 0xFFFFFFFFULL),
      /*dwNumberOfBytesToMap=*/length + offset_in_buffer_);

  // The mapped view keeps the mapping object alive, so the handle can be closed
  // immediately; the view remains valid until UnmapViewOfFile().
  ::CloseHandle(file_mapping);

  if (mmapped_buffer_ == nullptr) {
    mmapped_buffer_ = MAP_FAILED;
    TF_LITE_REPORT_ERROR(error_reporter,
                         "MapViewOfFile of '%d' at offset '%d' failed with "
                         "error '%lu'.",
                         mmap_fd_, offset, ::GetLastError());
    return;
  }
#else
  mmapped_buffer_ = mmap(nullptr, /*__len=*/length + offset_in_buffer_,
                         map_private ? (PROT_READ | PROT_WRITE) : PROT_READ,
                         map_private ? MAP_PRIVATE : MAP_SHARED, mmap_fd_,
                         /*__offset=*/offset_of_buffer_in_file_);
  if (mmapped_buffer_ == MAP_FAILED) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Mmap of '%d' at offset '%d' failed with error '%d'.",
                         mmap_fd_, offset, errno);
    return;
  }
#endif  // defined(_WIN32)
}

MMAPAllocation::~MMAPAllocation() {
  if (valid()) {
#if defined(_WIN32)
    ::UnmapViewOfFile(mmapped_buffer_);
#else
    munmap(const_cast<void*>(mmapped_buffer_),
           buffer_size_bytes_ + offset_in_buffer_);
#endif  // defined(_WIN32)
  }
  if (mmap_fd_ >= 0) {
    CloseFd(mmap_fd_);
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
