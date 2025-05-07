/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/file_util.h"

#include <fcntl.h>

#if defined(_MSC_VER)
#include <io.h>
#define F_OK 0
#else
#include <unistd.h>
#endif  // defined(_MSC_VER)

// We currently use the memfd_create system call to create in-memory files which
// is only supported on Linux and Android.
#if defined(__linux__) || defined(__ANDROID__)
#ifndef TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED
// Some systems have syscall.h but don't define the SYS_memfd_create macro. We
// detect those by actually doing the include and checking for its definition.
#include <sys/syscall.h>
#ifdef SYS_memfd_create
#define TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED 1
#endif  // SYS_memfd_create
#endif  // TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED
#endif  // defined(__linux__) || defined(__ANDROID__)

#include <cstdio>

#if !TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#endif

namespace tflite {
namespace xnnpack {

FileDescriptor FileDescriptor::Duplicate(int fd) {
  return FileDescriptor(dup(fd));
}

FileDescriptor FileDescriptor::Duplicate() const {
  if (!IsValid()) {
    return FileDescriptor(-1);
  }
  return FileDescriptor(dup(fd_));
}

void FileDescriptor::Reset(int new_fd) {
  if (fd_ == new_fd) {
    return;
  }
  if (IsValid()) {
    close(fd_);
  }
  fd_ = new_fd;
}

off_t FileDescriptor::GetPos() const { return lseek(fd_, 0, SEEK_CUR); }

off_t FileDescriptor::SetPos(off_t position) const {
  return lseek(fd_, position, SEEK_SET);
}

off_t FileDescriptor::SetPosFromEnd(off_t offset) const {
  return lseek(fd_, offset, SEEK_END);
}

off_t FileDescriptor::MovePos(off_t offset) const {
  return lseek(fd_, offset, SEEK_CUR);
}

FileDescriptor FileDescriptor::Open(const char* path, int flags, mode_t mode) {
  return FileDescriptor(open(path, flags, mode));
}

void FileDescriptor::Close() { Reset(-1); }

bool FileDescriptor::Read(void* dst, size_t count) const {
  char* dst_it = reinterpret_cast<char*>(dst);
  while (count > 0) {
    const auto bytes = read(fd_, dst_it, count);
    if (bytes == -1) {
      return false;
    } else if (bytes == 0) {
      break;
    }
    count -= bytes;
    dst_it += bytes;
  }
  return true;
}

bool FileDescriptor::Write(const void* src, size_t count) const {
  const char* src_it = reinterpret_cast<const char*>(src);
  while (count > 0) {
    const auto bytes = write(fd_, src_it, count);
    if (bytes == -1) {
      return false;
    }
    count -= bytes;
    src_it += bytes;
  }
  return true;
}

bool InMemoryFileDescriptorAvailable() {
#if TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED
  // Test if the syscall memfd_create is available.
  const int test_fd = syscall(SYS_memfd_create, "test fd", 0);
  if (test_fd != -1) {
    close(test_fd);
    return true;
  }
#endif
  return false;
}

FileDescriptor CreateInMemoryFileDescriptor(const char* path) {
#ifdef TFLITE_XNNPACK_IN_MEMORY_FILE_ENABLED
  return FileDescriptor(
      syscall(SYS_memfd_create, "XNNPack in-memory weight cache", 0));
#else
  TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,
                  "XNNPack weight cache: in-memory cache is not enabled for "
                  "this build.");
  return FileDescriptor(-1);
#endif
}

}  // namespace xnnpack
}  // namespace tflite
