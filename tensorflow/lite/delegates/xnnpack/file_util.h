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
#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_FILE_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_FILE_UTIL_H_

#include <sys/types.h>

#include <cstddef>
#include <utility>

namespace tflite {
namespace xnnpack {

#if defined(_MSC_VER)
using mode_t = int;
#endif

// Wraps a C file descriptor and closes it when destroyed.
//
// Note that constness of the wrapped does NOT propagate to the file operations.
class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}

  FileDescriptor() = default;

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) : fd_(other.fd_) { other.fd_ = -1; }

  FileDescriptor& operator=(FileDescriptor&& other) {
    if (other.fd_ != fd_) {
      Close();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  ~FileDescriptor() { Close(); }

  // Duplicates an existing raw file descriptor.
  static FileDescriptor Duplicate(int fd);

  // Checks that the file descriptor has a valid value.
  //
  // WARNING: this does not check that the descriptor points to an open file.
  bool IsValid() const { return fd_ >= 0; }

  // Returns the file descriptor value.
  int Value() const { return fd_; }

  // Closes the current file descriptor if needed and assigns the given value.
  void Reset(int new_fd);

  // Returns the cursor position in the current file.
  //
  // Equivalent to MovePos(0).
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t GetPos() const;

  // Sets the absolute cursor position in the current file.
  //
  // Returns the cursor position in the file or -1 on error.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t SetPos(off_t position) const;

  // Sets the cursor position relative to the file end.
  //
  // Returns the cursor position in the file or -1 on error.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t SetPosFromEnd(off_t offset) const;

  // Moves the cursor position by the given offset in the current file.
  //
  // Returns the cursor position in the file or -1 on error.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t MovePos(off_t offset) const;

  // Duplicates the current file descriptor and returns the new file descriptor.
  //
  // If the file descriptor is invalid, returns a new invalid FileDescriptor
  // object.
  FileDescriptor Duplicate() const;

  // Opens a file.
  //
  // Directly maps to the standard C function `open`.
  static FileDescriptor Open(const char* path, int flags, mode_t mode = 0);

  // Closes the current file descriptor and sets it to -1.
  void Close();

  // Reads `count` bytes from the file at the current position to `dst`.
  //
  // Returns true if all the data available in the file was read to the buffer
  // (i.e. `count` bytes were read or EOF was reached).
  //
  // This is a convenience function wrapping the standard `read` function. If
  // you need finer grain control use that directly.
  [[nodiscard /*Reading from a file may fail.*/]]
  bool Read(void* dst, size_t count) const;

  // Writes `count` bytes to the file at the current position from `src`.
  //
  // This is a convenience function wrapping the standard `write` function. If
  // you need finer grain control use that directly.
  [[nodiscard /*Reading from a file may fail.*/]]
  bool Write(const void* src, size_t count) const;

  // Returns the current file descriptor value and stops managing it.
  int Release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }

  friend void swap(FileDescriptor& f1, FileDescriptor& f2) {
    using std::swap;
    swap(f1.fd_, f2.fd_);
  }

 private:
  int fd_ = -1;
};

// Checks if the current build and system support creating an in-memory file
// descriptor.
bool InMemoryFileDescriptorAvailable();

// Creates a new file descriptor that isn't backed by a file system. The file
// will be automatically cleaned up when the last file descriptor pointing to it
// is closed.
FileDescriptor CreateInMemoryFileDescriptor(const char* path);

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_FILE_UTIL_H_
