/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FILE_LOCK_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FILE_LOCK_H_

#ifndef _WIN32
#include <unistd.h>
#endif  // !_WIN32

#include <string>

namespace tflite {
namespace acceleration {

// A simple mutex lock implemented with file descriptor. Not supported in
// Windows. This lock will release safely when the calling thread / process
// crashes.
class FileLock {
 public:
  explicit FileLock(const std::string& path) : path_(path) {}

  // Move only.
  FileLock(FileLock&& other) = default;
  FileLock& operator=(FileLock&& other) = default;

  ~FileLock() {
#ifndef _WIN32
    if (fd_ >= 0) {
      close(fd_);
    }
#endif  // !_WIN32
  }

  // Returns whether the lock is acquired successfully.
  bool TryLock();

 private:
  std::string path_;
  int fd_ = -1;
};

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FILE_LOCK_H_
