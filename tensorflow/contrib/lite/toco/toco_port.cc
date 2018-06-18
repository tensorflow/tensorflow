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
#include <cstring>

#include "tensorflow/contrib/lite/toco/toco_port.h"
#include "tensorflow/contrib/lite/toco/toco_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {
namespace port {
void CopyToBuffer(const string& src, char* dest) {
  memcpy(dest, src.data(), src.size());
}

#ifdef PLATFORM_GOOGLE
void CopyToBuffer(const Cord& src, char* dest) { src.CopyToArray(dest); }
#endif
}  // namespace port
}  // namespace toco

#if defined(PLATFORM_GOOGLE) && !defined(__APPLE__) && !defined(__ANDROID__)

// Wrap Google file operations.

#include "base/init_google.h"
#include "file/base/file.h"
#include "file/base/filesystem.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"

namespace toco {
namespace port {

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags) {
  ::InitGoogle(usage, argc, argv, remove_flags);
}

void CheckInitGoogleIsDone(const char* message) {
  ::CheckInitGoogleIsDone(message);
}

namespace file {

// Conversion to our wrapper Status.
tensorflow::Status ToStatus(const ::util::Status& uts) {
  if (!uts.ok()) {
    return tensorflow::Status(tensorflow::errors::Code(uts.error_code()),
                              uts.error_message());
  }
  return tensorflow::Status::OK();
}

// Conversion to our wrapper Options.
toco::port::file::Options ToOptions(const ::file::Options& options) {
  CHECK_EQ(&options, &::file::Defaults());
  return Options();
}

tensorflow::Status Writable(const string& filename) {
  File* f = nullptr;
  const auto status = ::file::Open(filename, "w", &f, ::file::Defaults());
  if (f) {
    QCHECK_OK(f->Close(::file::Defaults()));
  }
  return ToStatus(status);
}

tensorflow::Status Readable(const string& filename,
                            const file::Options& options) {
  return ToStatus(::file::Readable(filename, ::file::Defaults()));
}

tensorflow::Status Exists(const string& filename,
                          const file::Options& options) {
  auto status = ::file::Exists(filename, ::file::Defaults());
  return ToStatus(status);
}

tensorflow::Status GetContents(const string& filename, string* contents,
                               const file::Options& options) {
  return ToStatus(::file::GetContents(filename, contents, ::file::Defaults()));
}

tensorflow::Status SetContents(const string& filename, const string& contents,
                               const file::Options& options) {
  return ToStatus(::file::SetContents(filename, contents, ::file::Defaults()));
}

string JoinPath(const string& a, const string& b) {
  return ::file::JoinPath(a, b);
}

}  // namespace file
}  // namespace port
}  // namespace toco

#else  // (__APPLE__ || __ANDROID__)

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

#if defined(PLATFORM_GOOGLE)
#include "base/commandlineflags.h"
#endif

namespace toco {
namespace port {

static bool port_initialized = false;

void InitGoogle(const char* usage, int* argc, char*** argv, bool remove_flags) {
  if (!port_initialized) {
#if defined(PLATFORM_GOOGLE)
    ParseCommandLineFlags(argc, argv, remove_flags);
#endif
    port_initialized = true;
  }
}

void CheckInitGoogleIsDone(const char* message) {
  CHECK(port_initialized) << message;
}

namespace file {

tensorflow::Status Writable(const string& filename) {
  FILE* f = fopen(filename.c_str(), "w");
  if (f) {
    fclose(f);
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("not writable");
}

tensorflow::Status Readable(const string& filename,
                            const file::Options& options) {
  FILE* f = fopen(filename.c_str(), "r");
  if (f) {
    fclose(f);
    return tensorflow::Status::OK();
  }
  return tensorflow::errors::NotFound("not readable");
}

tensorflow::Status Exists(const string& filename,
                          const file::Options& options) {
  struct stat statbuf;
  int ret = stat(filename.c_str(), &statbuf);
  if (ret == -1) {
    return tensorflow::errors::NotFound("file doesn't exist");
  }
  return tensorflow::Status::OK();
}

tensorflow::Status GetContents(const string& path, string* output,
                               const file::Options& options) {
  output->clear();

  int fd = open(path.c_str(), O_RDONLY);
  if (fd == -1) {
    return tensorflow::errors::NotFound("can't open() for read");
  }

  // Direct read, for speed.
  const int kBufSize = 1 << 16;
  char buffer[kBufSize];
  while (true) {
    int size = read(fd, buffer, kBufSize);
    if (size == 0) {
      // Done.
      close(fd);
      return tensorflow::Status::OK();
    } else if (size == -1) {
      // Error.
      close(fd);
      return tensorflow::errors::Internal("error during read()");
    } else {
      output->append(buffer, size);
    }
  }

  CHECK(0);
  return tensorflow::errors::Internal("internal error");
}

tensorflow::Status SetContents(const string& filename, const string& contents,
                               const file::Options& options) {
  int fd = open(filename.c_str(), O_WRONLY | O_CREAT, 0664);
  if (fd == -1) {
    return tensorflow::errors::Internal("can't open() for write");
  }

  size_t i = 0;
  while (i < contents.size()) {
    size_t to_write = contents.size() - i;
    ssize_t written = write(fd, &contents[i], to_write);
    if (written == -1) {
      close(fd);
      return tensorflow::errors::Internal("write() error");
    }
    i += written;
  }
  close(fd);

  return tensorflow::Status::OK();
}

string JoinPath(const string& base, const string& filename) {
  if (base.empty()) return filename;
  string base_fixed = base;
  if (!base_fixed.empty() && base_fixed.back() == '/') base_fixed.pop_back();
  string filename_fixed = filename;
  if (!filename_fixed.empty() && filename_fixed.front() == '/')
    filename_fixed.erase(0, 1);
  return base_fixed + "/" + filename_fixed;
}

}  // namespace file
}  // namespace port
}  // namespace toco

#endif  // (__APPLE || __ANDROID__)
