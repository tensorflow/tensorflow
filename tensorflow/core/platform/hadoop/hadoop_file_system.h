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

#ifndef TENSORFLOW_CORE_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_

#include "tensorflow/core/platform/env.h"

extern "C" {
struct hdfs_internal;
typedef hdfs_internal* hdfsFS;
}

namespace tensorflow {

class LibHDFS;

class HadoopFileSystem : public FileSystem {
 public:
  HadoopFileSystem();
  ~HadoopFileSystem();

  Status NewRandomAccessFile(
      const string& fname,
      std::unique_ptr<RandomAccessFile>*
          result /*, TransactionToken* token = nullptr*/) override;

  Status NewWritableFile(
      const string& fname,
      std::unique_ptr<WritableFile>*
          result /*, TransactionToken* token = nullptr*/) override;

  Status NewAppendableFile(
      const string& fname,
      std::unique_ptr<WritableFile>*
          result /*, TransactionToken* token = nullptr*/) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>*
          result /*, TransactionToken* token = nullptr*/) override;

  Status FileExists(
      const string& fname /*, TransactionToken* token = nullptr*/) override;

  Status GetChildren(
      const string& dir,
      std::vector<string>* result /*, TransactionToken* token = nullptr*/)
      override;

  Status GetMatchingPaths(
      const string& pattern,
      std::vector<string>* results /*, TransactionToken* token = nullptr*/)
      override;

  Status DeleteFile(
      const string& fname /*, TransactionToken* token = nullptr*/) override;

  Status CreateDir(
      const string& name /*, TransactionToken* token = nullptr*/) override;

  Status DeleteDir(
      const string& name /*, TransactionToken* token = nullptr*/) override;

  Status GetFileSize(
      const string& fname,
      uint64* size /*, TransactionToken* token = nullptr*/) override;

  Status RenameFile(
      const string& src,
      const string& target /*, TransactionToken* token = nullptr*/) override;

  Status Stat(
      const string& fname,
      FileStatistics* stat /*, TransactionToken* token = nullptr*/) override;

  string TranslateName(
      const string& name /*, TransactionToken* token = nullptr*/)
      const override;

 private:
  Status Connect(StringPiece fname, hdfsFS* fs);
};

Status SplitArchiveNameAndPath(StringPiece& path, string& nn);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_HADOOP_HADOOP_FILE_SYSTEM_H_
