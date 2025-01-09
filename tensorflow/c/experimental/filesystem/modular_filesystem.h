/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/file_system.h"

/// This file builds classes needed to hold a filesystem implementation in the
/// modular world. Once all TensorFlow filesystems are converted to use the
/// plugin based approach, this file will replace the one in core/platform and
/// the names will lose the `Modular` part. Until that point, the `Modular*`
/// classes here are experimental and subject to breaking changes.
/// For documentation on these methods, consult `core/platform/filesystem.h`.

namespace tensorflow {

// TODO(b/143949615): After all filesystems are converted, this file will be
// moved to core/platform, and this class can become a singleton and replace the
// need for `Env::Default()`. At that time, we might decide to remove the need
// for `Env::Default()` altogether, but that's a different project, not in
// scope for now. I'm just mentioning this here as that transition will mean
// removal of the registration part from `Env` and adding it here instead: we
// will need tables to hold for each scheme the function tables that implement
// the needed functionality instead of the current `FileSystemRegistry` code in
// `core/platform/env.cc`.
class ModularFileSystem final : public FileSystem {
 public:
  ModularFileSystem(
      std::unique_ptr<TF_Filesystem> filesystem,
      std::unique_ptr<const TF_FilesystemOps> filesystem_ops,
      std::unique_ptr<const TF_RandomAccessFileOps> random_access_file_ops,
      std::unique_ptr<const TF_WritableFileOps> writable_file_ops,
      std::unique_ptr<const TF_ReadOnlyMemoryRegionOps>
          read_only_memory_region_ops,
      std::function<void*(size_t)> plugin_memory_allocate,
      std::function<void(void*)> plugin_memory_free)
      : filesystem_(std::move(filesystem)),
        ops_(std::move(filesystem_ops)),
        random_access_file_ops_(std::move(random_access_file_ops)),
        writable_file_ops_(std::move(writable_file_ops)),
        read_only_memory_region_ops_(std::move(read_only_memory_region_ops)),
        plugin_memory_allocate_(std::move(plugin_memory_allocate)),
        plugin_memory_free_(std::move(plugin_memory_free)) {}

  ~ModularFileSystem() override { ops_->cleanup(filesystem_.get()); }

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
  absl::Status FileExists(const std::string& fname,
                          TransactionToken* token) override;
  bool FilesExist(const std::vector<std::string>& files,
                  TransactionToken* token,
                  std::vector<absl::Status>* status) override;
  absl::Status GetChildren(const std::string& dir, TransactionToken* token,
                           std::vector<std::string>* result) override;
  absl::Status GetMatchingPaths(const std::string& pattern,
                                TransactionToken* token,
                                std::vector<std::string>* results) override;
  absl::Status DeleteFile(const std::string& fname,
                          TransactionToken* token) override;
  absl::Status DeleteRecursively(const std::string& dirname,
                                 TransactionToken* token,
                                 int64_t* undeleted_files,
                                 int64_t* undeleted_dirs) override;
  absl::Status DeleteDir(const std::string& dirname,
                         TransactionToken* token) override;
  absl::Status RecursivelyCreateDir(const std::string& dirname,
                                    TransactionToken* token) override;
  absl::Status CreateDir(const std::string& dirname,
                         TransactionToken* token) override;
  absl::Status Stat(const std::string& fname, TransactionToken* token,
                    FileStatistics* stat) override;
  absl::Status IsDirectory(const std::string& fname,
                           TransactionToken* token) override;
  absl::Status GetFileSize(const std::string& fname, TransactionToken* token,
                           uint64* file_size) override;
  absl::Status RenameFile(const std::string& src, const std::string& target,
                          TransactionToken* token) override;
  absl::Status CopyFile(const std::string& src, const std::string& target,
                        TransactionToken* token) override;
  std::string TranslateName(const std::string& name) const override;
  void FlushCaches(TransactionToken* token) override;
  absl::Status SetOption(const std::string& name,
                         const std::vector<string>& values) override;
  absl::Status SetOption(const std::string& name,
                         const std::vector<int64_t>& values) override;
  absl::Status SetOption(const std::string& name,
                         const std::vector<double>& values) override;

 private:
  std::unique_ptr<TF_Filesystem> filesystem_;
  std::unique_ptr<const TF_FilesystemOps> ops_;
  std::unique_ptr<const TF_RandomAccessFileOps> random_access_file_ops_;
  std::unique_ptr<const TF_WritableFileOps> writable_file_ops_;
  std::unique_ptr<const TF_ReadOnlyMemoryRegionOps>
      read_only_memory_region_ops_;
  std::function<void*(size_t)> plugin_memory_allocate_;
  std::function<void(void*)> plugin_memory_free_;
  ModularFileSystem(const ModularFileSystem&) = delete;
  void operator=(const ModularFileSystem&) = delete;
};

class ModularRandomAccessFile final : public RandomAccessFile {
 public:
  ModularRandomAccessFile(const std::string& filename,
                          std::unique_ptr<TF_RandomAccessFile> file,
                          const TF_RandomAccessFileOps* ops)
      : filename_(filename), file_(std::move(file)), ops_(ops) {}

  ~ModularRandomAccessFile() override { ops_->cleanup(file_.get()); }

  absl::Status Read(uint64 offset, size_t n, StringPiece* result,
                    char* scratch) const override;
  absl::Status Name(StringPiece* result) const override;

 private:
  std::string filename_;
  std::unique_ptr<TF_RandomAccessFile> file_;
  const TF_RandomAccessFileOps* ops_;  // not owned
  ModularRandomAccessFile(const ModularRandomAccessFile&) = delete;
  void operator=(const ModularRandomAccessFile&) = delete;
};

class ModularWritableFile final : public WritableFile {
 public:
  ModularWritableFile(const std::string& filename,
                      std::unique_ptr<TF_WritableFile> file,
                      const TF_WritableFileOps* ops)
      : filename_(filename), file_(std::move(file)), ops_(ops) {}

  ~ModularWritableFile() override { ops_->cleanup(file_.get()); }

  absl::Status Append(StringPiece data) override;
  absl::Status Close() override;
  absl::Status Flush() override;
  absl::Status Sync() override;
  absl::Status Name(StringPiece* result) const override;
  absl::Status Tell(int64_t* position) override;

 private:
  std::string filename_;
  std::unique_ptr<TF_WritableFile> file_;
  const TF_WritableFileOps* ops_;  // not owned
  ModularWritableFile(const ModularWritableFile&) = delete;
  void operator=(const ModularWritableFile&) = delete;
};

class ModularReadOnlyMemoryRegion final : public ReadOnlyMemoryRegion {
 public:
  ModularReadOnlyMemoryRegion(std::unique_ptr<TF_ReadOnlyMemoryRegion> region,
                              const TF_ReadOnlyMemoryRegionOps* ops)
      : region_(std::move(region)), ops_(ops) {}

  ~ModularReadOnlyMemoryRegion() override { ops_->cleanup(region_.get()); };

  const void* data() override { return ops_->data(region_.get()); }
  uint64 length() override { return ops_->length(region_.get()); }

 private:
  std::unique_ptr<TF_ReadOnlyMemoryRegion> region_;
  const TF_ReadOnlyMemoryRegionOps* ops_;  // not owned
  ModularReadOnlyMemoryRegion(const ModularReadOnlyMemoryRegion&) = delete;
  void operator=(const ModularReadOnlyMemoryRegion&) = delete;
};

// Registers a filesystem plugin so that core TensorFlow can use it.
absl::Status RegisterFilesystemPlugin(const std::string& dso_path);

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_MODULAR_FILESYSTEM_H_
