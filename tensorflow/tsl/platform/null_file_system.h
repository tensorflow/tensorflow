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

#ifndef TENSORFLOW_TSL_PLATFORM_NULL_FILE_SYSTEM_H_
#define TENSORFLOW_TSL_PLATFORM_NULL_FILE_SYSTEM_H_

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/file_system.h"
#include "tensorflow/tsl/platform/file_system_helper.h"

namespace tsl {

// START_SKIP_DOXYGEN

#ifndef SWIG
// Degenerate file system that provides no implementations.
class NullFileSystem : public FileSystem {
 public:
  NullFileSystem() {}

  ~NullFileSystem() override = default;

  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  Status NewRandomAccessFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
    return errors::Unimplemented("NewRandomAccessFile unimplemented");
  }

  Status NewWritableFile(const string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  Status FileExists(const string& fname, TransactionToken* token) override {
    return errors::Unimplemented("FileExists unimplemented");
  }

  Status GetChildren(const string& dir, TransactionToken* token,
                     std::vector<string>* result) override {
    return errors::Unimplemented("GetChildren unimplemented");
  }

  Status GetMatchingPaths(const string& pattern, TransactionToken* token,
                          std::vector<string>* results) override {
    return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
  }

  Status DeleteFile(const string& fname, TransactionToken* token) override {
    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname, TransactionToken* token) override {
    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname, TransactionToken* token) override {
    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, TransactionToken* token,
                     uint64* file_size) override {
    return errors::Unimplemented("GetFileSize unimplemented");
  }

  Status RenameFile(const string& src, const string& target,
                    TransactionToken* token) override {
    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, TransactionToken* token,
              FileStatistics* stat) override {
    return errors::Unimplemented("Stat unimplemented");
  }
};
#endif

// END_SKIP_DOXYGEN

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_NULL_FILE_SYSTEM_H_
