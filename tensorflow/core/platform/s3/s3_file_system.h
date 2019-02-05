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

#ifndef TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_
#define TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_

#include <aws/s3/S3Client.h>
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class S3FileSystem : public FileSystem {
 public:
  S3FileSystem();
  ~S3FileSystem();

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override;

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status FileExists(const string& fname) override;

  Status GetChildren(const string& dir, std::vector<string>* result) override;

  Status Stat(const string& fname, FileStatistics* stat) override;

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override;

  Status DeleteFile(const string& fname) override;

  Status CreateDir(const string& name) override;

  Status DeleteDir(const string& name) override;

  Status GetFileSize(const string& fname, uint64* size) override;

  Status RenameFile(const string& src, const string& target) override;

 private:
  // Returns the member S3 client, initializing as-needed.
  // When the client tries to access the object in S3, e.g.,
  //   s3://bucket-name/path/to/object
  // the behavior could be controlled by various environmental
  // variables.
  // By default S3 access regional endpoint, with region
  // controlled by `AWS_REGION`. The endpoint could be overridden
  // explicitly with `S3_ENDPOINT`. S3 uses HTTPS by default.
  // If S3_USE_HTTPS=0 is specified, HTTP is used. Also,
  // S3_VERIFY_SSL=0 could disable SSL verification in case
  // HTTPS is used.
  // This S3 Client does not support Virtual Hosted–Style Method
  // for a bucket.
  std::shared_ptr<Aws::S3::S3Client> GetS3Client();

  std::shared_ptr<Aws::S3::S3Client> s3_client_;
  // Lock held when checking for s3_client_ initialization.
  mutex client_lock_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_
