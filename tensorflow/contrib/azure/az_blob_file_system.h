/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_AZURE_AZ_BLOB_FILE_SYSTEM_H_
#define TENSORFLOW_CONTRIB_AZURE_AZ_BLOB_FILE_SYSTEM_H_

#include <memory>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/file_system.h"

#include "blob/blob_client.h"

namespace tensorflow {

class AzBlobFileSystem;

/// Azure Blob Storage implementation of a file system.
class AzBlobFileSystem : public FileSystem {
 public:
  Status NewRandomAccessFile(
      const std::string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const std::string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const std::string& fname) override;

  Status Stat(const std::string& fname, FileStatistics* stat) override;

  Status GetChildren(const std::string& dir,
                     std::vector<string>* result) override;

  Status GetMatchingPaths(const std::string& pattern,
                          std::vector<string>* results) override;

  Status DeleteFile(const std::string& fname) override;

  Status CreateDir(const std::string& dirname) override;

  Status DeleteDir(const std::string& dirname) override;

  Status GetFileSize(const std::string& fname, uint64* file_size) override;

  Status RenameFile(const std::string& src, const std::string& target) override;

  Status IsDirectory(const std::string& fname) override;

  Status RecursivelyCreateDir(const string& dirname) override;

  Status DeleteRecursively(const std::string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs) override;

  void FlushCaches() override;

 private:
  Status ListResources(
      const std::string& dir, const std::string& delimiter,
      microsoft_azure::storage::blob_client_wrapper& blob_client,
      std::vector<std::string>* results) const;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_AZURE_AZ_BLOB_FILE_SYSTEM_H_
