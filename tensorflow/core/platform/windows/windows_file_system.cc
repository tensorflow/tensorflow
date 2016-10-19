/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <Windows.h>
#include <direct.h>
#include <errno.h>
#include <fcntl.h>
#include <io.h>
#include <Shlwapi.h>
#undef StrCat
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/error.h"
#include "tensorflow/core/platform/windows/windows_file_system.h"

// TODO(mrry): Prevent this Windows.h #define from leaking out of our headers.
#undef DeleteFile

namespace tensorflow {

namespace {

// read() based random-access
class WindowsRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  FILE* file_;

 public:
  WindowsRandomAccessFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}
  ~WindowsRandomAccessFile() override {
    if (file_ != NULL) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    int seek_result = fseek(file_, offset, SEEK_SET);
    if (seek_result) {
      return IOError(filename_, errno);
    }
    while (n > 0 && s.ok()) {
      size_t r = fread(dst, 1, n, file_);
      if (r > 0) {
        dst += r;
        n -= r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read fewer bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }
};

class WindowsWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  WindowsWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~WindowsWritableFile() override {
    if (file_ != NULL) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(const StringPiece& data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = NULL;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }
};

}  // namespace

Status WindowsFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  result->reset();
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "r");
  if (f == NULL) {
    s = IOError(fname, errno);
  } else {
    result->reset(new WindowsRandomAccessFile(translated_fname, f));
  }
  return s;
}

Status WindowsFileSystem::NewWritableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == NULL) {
    result->reset();
    s = IOError(fname, errno);
  } else {
    result->reset(new WindowsWritableFile(translated_fname, f));
  }
  return s;
}

Status WindowsFileSystem::NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == NULL) {
    result->reset();
    s = IOError(fname, errno);
  } else {
    result->reset(new WindowsWritableFile(translated_fname, f));
  }
  return s;
}

Status WindowsFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  return errors::Unimplemented(
      "WindowsFileSystem::NewReadOnlyMemoryRegionFromFile");
}

bool WindowsFileSystem::FileExists(const string& fname) {
  return _access(TranslateName(fname).c_str(), 0) == 0;
}

Status WindowsFileSystem::GetChildren(const string& dir,
                                      std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();

  WIN32_FIND_DATA find_data;
  HANDLE find_handle = FindFirstFile(translated_dir.c_str(), &find_data);
  if (find_handle == INVALID_HANDLE_VALUE) {
    // TODO(mrry): Convert to a more specific error.
    return errors::Unknown("Error code: ", GetLastError());
  }
  result->push_back(find_data.cFileName);
  while (FindNextFile(find_handle, &find_data)) {
    result->push_back(find_data.cFileName);
  }
  if (!FindClose(find_handle)) {
    // TODO(mrry): Convert to a more specific error.
    return errors::Unknown("Error closing find handle: ", GetLastError());
  }
  return Status::OK();
}

Status WindowsFileSystem::DeleteFile(const string& fname) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status WindowsFileSystem::CreateDir(const string& name) {
  Status result;
  if (_mkdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status WindowsFileSystem::DeleteDir(const string& name) {
  Status result;
  if (_rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status WindowsFileSystem::GetFileSize(const string& fname, uint64* size) {
  Status s;
  struct _stat sbuf;
  if (_stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status WindowsFileSystem::RenameFile(const string& src, const string& target) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status WindowsFileSystem::Stat(const string& fname, FileStatistics* stat) {
  Status s;
  struct _stat sbuf;
  if (_stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stat->mtime_nsec = sbuf.st_mtime * 1e9;
    stat->length = sbuf.st_size;
    stat->is_directory = PathIsDirectory(TranslateName(fname).c_str());
  }
  return s;
}

}  // namespace tensorflow