/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_HELPER_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_HELPER_H_

#include <aws/s3/S3Client.h>

#include <string>

#include "tensorflow/c/tf_status.h"

namespace tf_s3_filesystem {
#ifdef PLATFORM_WINDOWS
  static const char* kS3SuffixFileSystem = nullptr;
#else
  static const char* kS3SuffixFileSystem = "/tmp/s3_filesystem_XXXXXX";
#endif
  static const char* kS3FileSystemAllocationTag = "S3FileSystemAllocation";
  static const size_t kS3ReadAppendableFileBufferSize = 1024 * 1024;
  static const int kS3GetChildrenMaxKeys = 100;
  static const int kUploadRetries = 5;
  static const int64_t kS3TimeoutMsec = 300000;
  static const uint64_t kS3MultiPartCopyPartSize = 50 * 1024 * 1024;
  static const int kExecutorPoolSize = 5;
  static const char* part_size_str = getenv("S3_MULTI_PART_COPY_PART_SIZE");
  static const uint64_t multi_part_copy_part_size_ = part_size_str ? std::stoull(part_size_str) : kS3MultiPartCopyPartSize;
  static const char* kExecutorTag = "TransferManagerExecutor";
  void TF_SetStatusFromAWSError(TF_Status* status, const Aws::Client::AWSError<Aws::S3::S3Errors>& error);
  void GetParentDir(const char* name, char** parent);
  void GetParentFile(const char* name, char** parent);
  void ParseS3Test(const char* fname, bool object_empty_ok, char** bucket, char** object, TF_Status* status);
}  // namespace tf_s3_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_HELPER_H_
