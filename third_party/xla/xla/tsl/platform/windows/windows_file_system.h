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

#ifndef XLA_TSL_PLATFORM_WINDOWS_WINDOWS_FILE_SYSTEM_H_
#define XLA_TSL_PLATFORM_WINDOWS_WINDOWS_FILE_SYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/path.h"

#ifdef PLATFORM_WINDOWS
#undef CopyFile
#undef DeleteFile
#endif

namespace tsl {

class WindowsFileSystem : public FileSystem {
 public:
  WindowsFileSystem() = default;

  ~WindowsFileSystem() = default;

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status NewWritableFile(const std::string& fname,
                               TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override;

  absl::Status NewAppendableFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  absl::Status FileExists(absl::string_view fname,
                          TransactionToken* token) override;

  absl::Status GetChildren(const std::string& dir, TransactionToken* token,
                           std::vector<std::string>* result) override;

  absl::Status GetMatchingPaths(const std::string& pattern,
                                TransactionToken* token,
                                std::vector<std::string>* result) override;

  bool Match(absl::string_view filename, absl::string_view pattern) override;

  absl::Status Stat(const std::string& fname, TransactionToken* token,
                    FileStatistics* stat) override;

  absl::Status DeleteFile(const std::string& fname,
                          TransactionToken* token) override;

  absl::Status CreateDir(const std::string& name,
                         TransactionToken* token) override;

  absl::Status CreateDir(const std::string& name, TransactionToken* token,
                         uint32_t mode) override;

  absl::Status DeleteDir(const std::string& name,
                         TransactionToken* token) override;

  absl::Status DeleteRecursively(const std::string& dirname,
                                 TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override;

  absl::Status GetFileSize(const std::string& fname, TransactionToken* token,
                           uint64_t* size) override;

  absl::Status IsDirectory(const std::string& fname,
                           TransactionToken* token) override;

  absl::Status RenameFile(const std::string& src, const std::string& target,
                          TransactionToken* token) override;

  std::string TranslateName(absl::string_view name) const override {
    return std::string(name);
  }

  char Separator() const override { return '\\'; };
};

class LocalWinFileSystem : public WindowsFileSystem {
 public:
  std::string TranslateName(absl::string_view name) const override {
    absl::string_view scheme, host, path;
    io::ParseURI(name, &scheme, &host, &path);
    return std::string(path);
  }
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_WINDOWS_WINDOWS_FILE_SYSTEM_H_
