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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_COPY_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_COPY_H_

#include <aws/core/Aws.h>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/StringUtils.h>
#include <aws/s3/model/CompletedMultipartUpload.h>

#include <string>
#include <memory>

#include "tensorflow/c/tf_status.h"

// TODO(vnvo2409): Replace string by const char*

namespace tf_s3_filesystem {
typedef struct PartState {
  int partNumber;
  TF_Status* status;
} PartState;

struct MultiPartCopyAsyncContext : public Aws::Client::AsyncCallerContext {
  int partNumber;
  std::map<int, PartState>* incompletePartStates;
  std::map<int, PartState>* finishedPartStates;
  Aws::String eTag;

  // lock and cv for multi part copy
  std::mutex* multi_part_copy_mutex;
  std::condition_variable* multi_part_copy_cv;
};

void SimpleCopy(const std::string& source, const std::string& target_bucket, const std::string& target_key, 
                const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status);
void MultiPartCopy(const std::string& source, const std::string& target_bucket, const std::string& target_key, 
                   const int num_parts, const uint64_t file_length, 
                   const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status);
void AbortMultiPartCopy(const Aws::String& target_bucket, const Aws::String& target_key, const Aws::String& uploadID,
                        const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status);
void CompleteMultiPartCopy(const Aws::String& target_bucket, const Aws::String& target_key, const Aws::String& uploadID,
                           const std::shared_ptr<Aws::S3::S3Client>& s3_client,
                           Aws::S3::Model::CompletedMultipartUpload completedMPURequest, TF_Status* status);
void MultiPartCopyCallback(
    const Aws::S3::Model::UploadPartCopyRequest& request,
    const Aws::S3::Model::UploadPartCopyOutcome& uploadPartCopyOutcome,
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context);

}  // namespace tf_s3_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_COPY_H_
