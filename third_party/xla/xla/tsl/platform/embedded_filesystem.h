/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_TSL_PLATFORM_EMBEDDED_FILESYSTEM_H_
#define XLA_TSL_PLATFORM_EMBEDDED_FILESYSTEM_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/file_statistics.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {

// Simple filesystem modeling in-memory immutable files. Directories are not a
// concept, and / is not treated specially. The only way to create a file is to
// call EmbedFile. All contents must outlive the filesystem and all file
// objects. Unlike RamFileSystem, the contents are not owned by the file system,
// providing a copy-free view.

class EmbedRandomAccessFile : public RandomAccessFile {
 public:
  EmbedRandomAccessFile(absl::string_view name, absl::string_view contents);
  ~EmbedRandomAccessFile() override;

  absl::Status Name(absl::string_view* result) const override;

  absl::Status Read(uint64_t offset, absl::string_view& result,
                    absl::Span<char> scratch) const override;

 private:
  absl::string_view name_;
  absl::string_view contents_;
};

class EmbedFileSystem : public tsl::FileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  ~EmbedFileSystem() override;

  // Creates a new embedded file, overwriting it if it exists.
  absl::Status EmbedFile(absl::string_view fname, absl::string_view contents);

  absl::Status NewRandomAccessFile(
      const std::string& fname,
      std::unique_ptr<RandomAccessFile>* result) override;

  absl::Status FileExists(const std::string& fname) override;

  absl::Status GetFileSize(const std::string& fname,
                           uint64_t* file_size) override;

  absl::Status GetMatchingPaths(const std::string& pattern,
                                std::vector<std::string>* results) override;

  // Unimplemented operations below here.
  absl::Status NewWritableFile(const std::string& fname,
                               std::unique_ptr<WritableFile>* result) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support writable files.");
  }

  absl::Status NewAppendableFile(
      const std::string& fname,
      std::unique_ptr<WritableFile>* result) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support appendable files.");
  }

  absl::Status IsDirectory(const std::string& fname) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support directories.");
  }

  absl::Status GetChildren(const std::string& dirname,
                           std::vector<std::string>* result) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support directories.");
  }

  absl::Status Stat(const std::string& fname, FileStatistics* output) override {
    return absl::UnimplementedError("EmbedFileSystem does not support stat.");
  }

  absl::Status DeleteFile(const std::string& fname) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support deletion.");
  }

  absl::Status DeleteDir(const std::string& dir) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support deletion (or directories).");
  }

  absl::Status DeleteRecursively(const std::string& dir,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support deletion.");
  }

  absl::Status CreateDir(const std::string& dir) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support directories.");
  }

  absl::Status RecursivelyCreateDir(const std::string& dir) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support directories.");
  }

  absl::Status RenameFile(const std::string& src,
                          const std::string& dst) override {
    return absl::UnimplementedError(
        "EmbedFileSystem does not support renames.");
  }

  absl::Status CopyFile(const std::string& src,
                        const std::string& dst) override {
    return absl::UnimplementedError("EmbedFileSystem does not support copies.");
  }

 private:
  absl::Mutex fs_lock_;
  absl::flat_hash_map<std::string, absl::string_view> fs_
      ABSL_GUARDED_BY(fs_lock_);
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_EMBEDDED_FILESYSTEM_H_
