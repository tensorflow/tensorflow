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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/file_lock.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // !_WIN32

#include <string>

namespace tflite {
namespace acceleration {

bool FileLock::TryLock() {
#ifndef _WIN32
  if (fd_ < 0) {
    // O_CLOEXEC is needed for correctness, as another thread may call
    // popen() and the callee would then inherit the lock if it's not O_CLOEXEC.
    fd_ = open(path_.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0600);
  }
  if (fd_ < 0) {
    return false;
  }
  if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
    return true;
  }
#endif  // !_WIN32
  return false;
}

}  // namespace acceleration
}  // namespace tflite
