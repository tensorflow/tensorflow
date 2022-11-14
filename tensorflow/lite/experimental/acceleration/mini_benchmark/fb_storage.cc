/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/fb_storage.h"

#include <fcntl.h>
#include <string.h>
#ifndef _WIN32
#include <sys/file.h>
#include <unistd.h>
#endif

#include <fstream>
#include <sstream>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/c/c_api_types.h"

// We only really care about Android, but we want the code to be portable for
// ease of testing. See also discussion in cl/224174491.
#ifndef TEMP_FAILURE_RETRY
#ifdef __ANDROID__
#error "TEMP_FAILURE_RETRY not set although on Android"
#else  // ! defined(__ANDROID__)
#define TEMP_FAILURE_RETRY(exp) exp
#endif  // defined(__ANDROID__)
#endif  // defined(TEMP_FAILURE_RETRY)

namespace tflite {
namespace acceleration {
FileStorage::FileStorage(absl::string_view path, ErrorReporter* error_reporter)
    : path_(path), error_reporter_(error_reporter) {}

MinibenchmarkStatus FileStorage::ReadFileIntoBuffer() {
#ifndef _WIN32
  buffer_.clear();
  // O_CLOEXEC is needed for correctness, as another thread may call
  // popen() and the callee inherit the lock if it's not O_CLOEXEC.
  int fd = TEMP_FAILURE_RETRY(open(path_.c_str(), O_RDONLY | O_CLOEXEC, 0600));
  int open_error_no = errno;
  if (fd < 0) {
    // Try to create if it doesn't exist.
    int fd = TEMP_FAILURE_RETRY(
        open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600));
    if (fd >= 0) {
      // Successfully created file, all good.
      close(fd);
      return kMinibenchmarkSuccess;
    }
    int create_error_no = errno;
    TF_LITE_REPORT_ERROR(
        error_reporter_,
        "Could not open %s for reading: %s, creating failed as well: %s",
        path_.c_str(), std::strerror(open_error_no),
        std::strerror(create_error_no));
    return kMinibenchmarkCantCreateStorageFile;
  }
  int lock_status = flock(fd, LOCK_EX);
  int lock_error_no = errno;
  if (lock_status < 0) {
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not flock %s: %s",
                         path_.c_str(), std::strerror(lock_error_no));
    return kMinibenchmarkFlockingStorageFileFailed;
  }
  char buffer[512];
  while (true) {
    int bytes_read = TEMP_FAILURE_RETRY(read(fd, buffer, 512));
    int read_error_no = errno;
    if (bytes_read == 0) {
      // EOF
      close(fd);
      return kMinibenchmarkSuccess;
    } else if (bytes_read < 0) {
      close(fd);
      TF_LITE_REPORT_ERROR(error_reporter_, "Error reading %s: %s",
                           path_.c_str(), std::strerror(read_error_no));
      return kMinibenchmarkErrorReadingStorageFile;
    } else {
      buffer_.append(buffer, bytes_read);
    }
  }
#else  // _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#endif
}

MinibenchmarkStatus FileStorage::AppendDataToFile(absl::string_view data) {
#ifndef _WIN32
  // We use a file descriptor (as opposed to FILE* or C++ streams) for writing
  // because we want to be able to use fsync and flock.
  // O_CLOEXEC is needed for correctness, as another thread may call
  // popen() and the callee inherit the lock if it's not O_CLOEXEC.
  int fd = TEMP_FAILURE_RETRY(
      open(path_.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0600));
  if (fd < 0) {
    int error_no = errno;
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not open %s for writing: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkFailedToOpenStorageFileForWriting;
  }
  int lock_status = flock(fd, LOCK_EX);
  int lock_error_no = errno;
  if (lock_status < 0) {
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Could not flock %s: %s",
                         path_.c_str(), std::strerror(lock_error_no));
    return kMinibenchmarkFlockingStorageFileFailed;
  }
  absl::string_view bytes = data;
  while (!bytes.empty()) {
    ssize_t bytes_written =
        TEMP_FAILURE_RETRY(write(fd, bytes.data(), bytes.size()));
    if (bytes_written < 0) {
      int error_no = errno;
      close(fd);
      TF_LITE_REPORT_ERROR(error_reporter_, "Could not write to %s: %s",
                           path_.c_str(), std::strerror(error_no));
      return kMinibenchmarkErrorWritingStorageFile;
    }
    bytes.remove_prefix(bytes_written);
  }
  if (TEMP_FAILURE_RETRY(fsync(fd)) < 0) {
    int error_no = errno;
    close(fd);
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to fsync %s: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkErrorFsyncingStorageFile;
  }
  if (TEMP_FAILURE_RETRY(close(fd)) < 0) {
    int error_no = errno;
    TF_LITE_REPORT_ERROR(error_reporter_, "Failed to close %s: %s",
                         path_.c_str(), std::strerror(error_no));
    return kMinibenchmarkErrorClosingStorageFile;
  }
  return kMinibenchmarkSuccess;
#else   // _WIN32
  return kMinibenchmarkUnsupportedPlatform;
#endif  // !_WIN32
}

const char kFlatbufferStorageIdentifier[] = "STO1";
}  // namespace acceleration
}  // namespace tflite
