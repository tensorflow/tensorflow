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

#include <utility>

namespace tflite {
namespace xnnpack {

// Wraps a C file descriptor and closes it when destroyed.
class FileDescriptor {
 public:
  explicit FileDescriptor(int fd) : fd_(fd) {}

  FileDescriptor() = default;

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  FileDescriptor(FileDescriptor&& other) : fd_(other.fd_) { other.fd_ = -1; }

  FileDescriptor& operator=(FileDescriptor&& other) {
    Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
  }

  ~FileDescriptor() { Close(); }

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
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t GetPos() const;

  // Sets the absolute cursor position in the current file.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t SetPos(off_t position);

  // Moves the cursor position by the given offset in the current file.
  //
  // WARNING: the file descriptor must be valid and the file must be opened.
  off_t MovePos(off_t offset);

  // Duplicates the current file descriptor and returns the new file descriptor.
  FileDescriptor Duplicate() const;

  // Closes the current file descriptor and set it to -1.
  void Close();

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
