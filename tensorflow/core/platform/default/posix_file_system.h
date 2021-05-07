/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/platform/path.h"

namespace tensorflow {

class PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() {}

  ~PosixFileSystem() {}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname, TransactionToken* token) override;

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override;

  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stats) override;

  Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                          std::vector<string>* results) override;

  Status DeleteFile(const string& fname, TransactionToken* token) override;

  Status CreateDir(const string& name, TransactionToken* token) override;

  Status DeleteDir(const string& name, TransactionToken* token) override;

  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* size) override;

  Status RenameFile(const string& src, const string& target,
                    TransactionToken* token) override;

  Status CopyFile(const string& src, const string& target,
                  TransactionToken* token) override;
};

Status IOError(const string& context, int err_number);

class LocalPosixFileSystem : public PosixFileSystem {
 public:
  string TranslateName(const string& name) const override {
    StringPiece scheme, host, path;
    io::ParseURI(name, &scheme, &host, &path);
    return string(path);
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_POSIX_POSIX_FILE_SYSTEM_H_
