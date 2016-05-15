/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_

#include <string>
#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/cloud/http_request.h"
#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

/// Interface for a provider of HTTP auth bearer tokens.
class AuthProvider {
 public:
  virtual ~AuthProvider() {}
  virtual Status GetToken(string* t) const = 0;
};

/// Google Cloud Storage implementation of a file system.
class GcsFileSystem : public FileSystem {
 public:
  GcsFileSystem();
  GcsFileSystem(std::unique_ptr<AuthProvider> auth_provider,
                std::unique_ptr<HttpRequest::Factory> http_request_factory);

  Status NewRandomAccessFile(const string& fname,
                             RandomAccessFile** result) override;

  Status NewWritableFile(const string& fname, WritableFile** result) override;

  Status NewAppendableFile(const string& fname, WritableFile** result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, ReadOnlyMemoryRegion** result) override;

  bool FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& dirname) override;

  Status DeleteDir(const string& dirname) override;

  Status GetFileSize(const string& fname, uint64* file_size) override;

  Status RenameFile(const string& src, const string& target) override;

 private:
  std::unique_ptr<AuthProvider> auth_provider_;
  std::unique_ptr<HttpRequest::Factory> http_request_factory_;
  TF_DISALLOW_COPY_AND_ASSIGN(GcsFileSystem);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_GCS_FILE_SYSTEM_H_
