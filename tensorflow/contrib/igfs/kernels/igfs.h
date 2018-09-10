/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CONTRIB_IGFS_KERNELS_IGFS_H_
#define TENSORFLOW_CONTRIB_IGFS_KERNELS_IGFS_H_

#include "igfs_client.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

class IGFS : public FileSystem {
 public:
  IGFS();
  ~IGFS();
  Status NewRandomAccessFile(
      const std::string& file_name,
      std::unique_ptr<RandomAccessFile>* result) override;
  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const std::string& fname,
                           std::unique_ptr<WritableFile>* result) override;
  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;
  Status FileExists(const std::string& fname) override;
  Status GetChildren(const std::string& dir,
                     std::vector<string>* result) override;
  Status GetMatchingPaths(const std::string& pattern,
                          std::vector<string>* results) override;
  Status DeleteFile(const std::string& fname) override;
  Status CreateDir(const std::string& name) override;
  Status DeleteDir(const std::string& name) override;
  Status GetFileSize(const std::string& fname, uint64* size) override;
  Status RenameFile(const std::string& src, const std::string& target) override;
  Status Stat(const std::string& fname, FileStatistics* stat) override;
  string TranslateName(const std::string& name) const override;

 private:
  const std::string host_;
  const int port_;
  const std::string fs_name_;

  std::shared_ptr<IGFSClient> CreateClient() const;
};

}  // namespace tensorflow

#endif
