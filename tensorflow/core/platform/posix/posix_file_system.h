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

#ifndef TENSORFLOW_CORE_PLATFORM_POSIX_POSIX_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_POSIX_POSIX_FILE_SYSTEM_H_

#include "tensorflow/core/platform/env.h"

namespace tensorflow {

class PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() {}

  ~PosixFileSystem() {}

  Status NewRandomAccessFile(const string& fname,
                             RandomAccessFile** result) override;

  Status NewWritableFile(const string& fname, WritableFile** result) override;

  Status NewAppendableFile(const string& fname, WritableFile** result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, ReadOnlyMemoryRegion** result) override;

  bool FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& name) override;

  Status DeleteDir(const string& name) override;

  Status GetFileSize(const string& fname, uint64* size) override;

  Status RenameFile(const string& src, const string& target) override;
};

Status IOError(const string& context, int err_number);

class LocalPosixFileSystem : public PosixFileSystem {
 public:
  string TranslateName(const string& name) const override {
    return GetNameFromURI(name);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_POSIX_POSIX_FILE_SYSTEM_H_
