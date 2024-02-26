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

#ifndef TENSORFLOW_TSL_PLATFORM_DEFAULT_POSIX_FILE_SYSTEM_H_
#define TENSORFLOW_TSL_PLATFORM_DEFAULT_POSIX_FILE_SYSTEM_H_

#include "tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace tsl {

class PosixFileSystem : public FileSystem {
 public:
  PosixFileSystem() {}

  ~PosixFileSystem() override {}

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status NewWritableFile(const string& fname, TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override;

  absl::Status NewAppendableFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const string& filename, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  absl::Status FileExists(const string& fname,
                          TransactionToken* token) override;

  absl::Status GetChildren(const string& dir, TransactionToken* token,
                           std::vector<string>* result) override;

  absl::Status Stat(const string& fname, TransactionToken* token,
                    FileStatistics* stats) override;

  absl::Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                                std::vector<string>* results) override;

  absl::Status DeleteFile(const string& fname,
                          TransactionToken* token) override;

  absl::Status CreateDir(const string& name, TransactionToken* token) override;

  absl::Status DeleteDir(const string& name, TransactionToken* token) override;

  absl::Status GetFileSize(const string& fname, TransactionToken* token,
                           uint64* size) override;

  absl::Status RenameFile(const string& src, const string& target,
                          TransactionToken* token) override;

  absl::Status CopyFile(const string& src, const string& target,
                        TransactionToken* token) override;
};

class LocalPosixFileSystem : public PosixFileSystem {
 public:
  string TranslateName(const string& name) const override {
    StringPiece scheme, host, path;
    io::ParseURI(name, &scheme, &host, &path);
    return string(path);
  }
};

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_DEFAULT_POSIX_FILE_SYSTEM_H_
