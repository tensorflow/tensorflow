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

#include <aws/core/utils/StringUtils.h>
#include <aws/core/utils/threading/Executor.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CompletedMultipartUpload.h>
#include <aws/transfer/TransferManager.h>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/retrying_file_system.h"

namespace tensorflow {

struct PartState {
  int partNumber;
  Status status;
};

struct MultiPartCopyAsyncContext : public Aws::Client::AsyncCallerContext {
  int partNumber;
  std::map<int, PartState>* incompletePartStates;
  std::map<int, PartState>* finishedPartStates;
  Aws::String eTag;

  // lock and cv for multi part copy
  std::mutex* multi_part_copy_mutex;
  std::condition_variable* multi_part_copy_cv;
};

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

  Status HasAtomicMove(const string& path, bool* has_atomic_move) override;

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

  // Returns the member transfer manager, initializing as-needed.
  std::shared_ptr<Aws::Transfer::TransferManager> GetTransferManager();
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager_;

  // Returns the member executor for transfer manager, initializing as-needed.
  std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> GetExecutor();
  std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> executor_;

  Status CopyFile(const Aws::String& source_bucket,
                  const Aws::String& source_key,
                  const Aws::String& target_bucket,
                  const Aws::String& target_key);
  Status SimpleCopy(const Aws::String& source, const Aws::String& target_bucket,
                    const Aws::String& target_key);
  Status MultiPartCopy(const Aws::String& source,
                       const Aws::String& target_bucket,
                       const Aws::String& target_key, const int num_parts,
                       const uint64 file_length);
  Status AbortMultiPartCopy(Aws::String target_bucket, Aws::String target_key,
                            Aws::String uploadID);
  Status CompleteMultiPartCopy(
      Aws::String target_bucket, Aws::String target_key, Aws::String uploadId,
      Aws::S3::Model::CompletedMultipartUpload completedMPURequest);
  void MultiPartCopyCallback(
      const Aws::S3::Model::UploadPartCopyRequest& request,
      const Aws::S3::Model::UploadPartCopyOutcome& uploadPartCopyOutcome,
      const std::shared_ptr<const Aws::Client::AsyncCallerContext>&
          multiPartContext);

  // Lock held when checking for s3_client_ and transfer_manager_ initialization
  mutex initialization_lock_;

  // size to split objects during multipart copy
  uint64 multi_part_copy_part_size_;
};

/// S3 implementation of a file system with retry on failures.
class RetryingS3FileSystem : public RetryingFileSystem<S3FileSystem> {
 public:
  RetryingS3FileSystem()
      : RetryingFileSystem(
            std::unique_ptr<S3FileSystem>(new S3FileSystem),
            RetryConfig(100000 /* init_delay_time_us */,
                        32000000 /* max_delay_time_us */, 10 /* max_retries */
                        )) {}
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CONTRIB_S3_S3_FILE_SYSTEM_H_
