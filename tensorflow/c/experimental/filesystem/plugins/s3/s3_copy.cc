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
#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_copy.h"

#include <aws/core/Aws.h>
#include <aws/core/config/AWSProfileConfigLoader.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/CopyObjectRequest.h>
#include <aws/s3/model/UploadPartCopyRequest.h>
#include <aws/s3/model/CompleteMultipartUploadRequest.h>
#include <aws/s3/model/AbortMultipartUploadRequest.h>
#include <aws/s3/model/CreateMultipartUploadRequest.h>

#include <cstdlib>
#include <memory>
#include <mutex>

#include "tensorflow/c/experimental/filesystem/plugins/s3/s3_helper.h"

namespace tf_s3_filesystem {

void SimpleCopy(const std::string& source, const std::string& target_bucket, const std::string& target_key, 
                const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status) {
  Aws::S3::Model::CopyObjectRequest copyObjectRequest;
  copyObjectRequest.SetBucket(target_bucket.c_str());
  copyObjectRequest.SetKey(target_key.c_str());
  copyObjectRequest.SetCopySource(source.c_str());
  auto copyObjectOutcome = s3_client->CopyObject(copyObjectRequest);
  if (!copyObjectOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, copyObjectOutcome.GetError());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void MultiPartCopyCallback(
    const Aws::S3::Model::UploadPartCopyRequest& request,
    const Aws::S3::Model::UploadPartCopyOutcome& uploadPartCopyOutcome,
    const std::shared_ptr<const Aws::Client::AsyncCallerContext>& context) {

  std::shared_ptr<MultiPartCopyAsyncContext> multiPartContext =
      std::const_pointer_cast<MultiPartCopyAsyncContext>(
          std::static_pointer_cast<MultiPartCopyAsyncContext>(
              std::const_pointer_cast<Aws::Client::AsyncCallerContext>(context)));

  {
    std::unique_lock<std::mutex> lock(*multiPartContext->multi_part_copy_mutex);

    TF_Status* status;
    if (uploadPartCopyOutcome.IsSuccess()) {
      // success
      Aws::String eTag =
          uploadPartCopyOutcome.GetResult().GetCopyPartResult().GetETag();
      multiPartContext->eTag = eTag;
      TF_SetStatus(status, TF_OK, "");
    } else {
      TF_SetStatusFromAWSError(status, uploadPartCopyOutcome.GetError());
    }

    (*multiPartContext->finishedPartStates)[multiPartContext->partNumber] =
        multiPartContext->incompletePartStates->at(
            multiPartContext->partNumber);
    multiPartContext->finishedPartStates->at(multiPartContext->partNumber)
        .status = status;
    multiPartContext->incompletePartStates->erase(multiPartContext->partNumber);
    // Notify the thread that started the operation
    multiPartContext->multi_part_copy_cv->notify_one();
  }
}

void MultiPartCopy(const std::string& source, const std::string& target_bucket, const std::string& target_key, 
                   const int num_parts, const uint64_t file_length, 
                   const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status) {

  Aws::S3::Model::CreateMultipartUploadRequest multipartUploadRequest;
  multipartUploadRequest.SetBucket(target_bucket.c_str());
  multipartUploadRequest.SetKey(target_key.c_str());

  auto multipartUploadOutcome =
      s3_client->CreateMultipartUpload(multipartUploadRequest);
  if (!multipartUploadOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, multipartUploadOutcome.GetError());
    return;
  }

  Aws::String uploadID = multipartUploadOutcome.GetResult().GetUploadId();
  Aws::S3::Model::CompletedMultipartUpload completedMPURequest;

  // passed to each callback keyed by partNumber
  std::map<int, std::shared_ptr<MultiPartCopyAsyncContext>>
      partContexts;
  // keeps track of incompleteParts keyed by partNumber
  std::map<int, PartState> incompletePartStates;
  // S3 API partNumber starts from 1
  for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
    PartState ps;
    ps.partNumber = partNumber;
    incompletePartStates[partNumber] = std::move(ps);
  }

  // keeps track of completed parts keyed by partNumber
  std::map<int, PartState> finishedPartStates;
  // mutex which protects access of the partStates map
  std::mutex multi_part_copy_mutex;
  // condition variable to be used with above mutex for synchronization
  std::condition_variable multi_part_copy_cv;

  int retry_count_ = 3;
  while (retry_count_-- > 0) {
    // queue up parts
    for (std::map<int, PartState>::iterator it = incompletePartStates.begin();
         it != incompletePartStates.end(); it++) {
      int partNumber = it->first;
      uint64_t startPos = (partNumber - 1) * multi_part_copy_part_size_;
      uint64_t endPos = startPos + kS3MultiPartCopyPartSize - 1;
      if (endPos >= file_length) {
        endPos = file_length - 1;
      }

      std::string range = "bytes=" + std::to_string(startPos) + "-" + std::to_string(endPos);

      Aws::S3::Model::UploadPartCopyRequest uploadPartCopyRequest;
      uploadPartCopyRequest.SetBucket(target_bucket.c_str());
      uploadPartCopyRequest.SetKey(target_key.c_str());
      uploadPartCopyRequest.SetCopySource(source.c_str());
      uploadPartCopyRequest.SetCopySourceRange(range.c_str());
      uploadPartCopyRequest.SetPartNumber(partNumber);
      uploadPartCopyRequest.SetUploadId(uploadID);

      auto multiPartContext =
          Aws::MakeShared<MultiPartCopyAsyncContext>(
              "MultiPartCopyContext");

      multiPartContext->partNumber = partNumber;
      multiPartContext->incompletePartStates = &incompletePartStates;
      multiPartContext->finishedPartStates = &finishedPartStates;
      multiPartContext->multi_part_copy_mutex = &multi_part_copy_mutex;
      multiPartContext->multi_part_copy_cv = &multi_part_copy_cv;

      // replace with current context
      partContexts[partNumber] = multiPartContext;

      auto callback =
          [](const Aws::S3::S3Client* client,
                 const Aws::S3::Model::UploadPartCopyRequest& request,
                 const Aws::S3::Model::UploadPartCopyOutcome& outcome,
                 const std::shared_ptr<const Aws::Client::AsyncCallerContext>&
                     context) {
            MultiPartCopyCallback(request, outcome, context);
          };

      s3_client->UploadPartCopyAsync(uploadPartCopyRequest, callback,
                                               multiPartContext);
    }
    // wait till they finish
    {
      std::unique_lock<std::mutex> lock(multi_part_copy_mutex);
      // wait on the mutex until notify is called
      // then check the finished parts as there could be false notifications
      multi_part_copy_cv.wait(lock, [&finishedPartStates, num_parts] {
        return finishedPartStates.size() == num_parts;
      });
    }
    // check if there was any error for any part
    for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
      if (TF_GetCode(finishedPartStates[partNumber].status) != TF_OK) {
        if (retry_count_ <= 0) {
          if (TF_GetCode(finishedPartStates[partNumber].status) != TF_OK) {
            AbortMultiPartCopy(target_bucket.c_str(), target_key.c_str(), uploadID, s3_client, status);
            if(TF_GetCode(status) != TF_OK) return;
            TF_SetStatus(status, TF_GetCode(finishedPartStates[partNumber].status), TF_Message(finishedPartStates[partNumber].status));
            return;
          }
        } else {
          // retry part
          PartState ps;
          ps.partNumber = partNumber;
          incompletePartStates[partNumber] = std::move(ps);
          finishedPartStates.erase(partNumber);
        }
      }
    }
  }
  
  // if there was an error still in any part, it would abort and return in the
  // above loop set the eTag of completed Part to the final CompletedMPURequest
  // note these parts have to be added in order
  for (int partNumber = 1; partNumber <= num_parts; partNumber++) {
    Aws::S3::Model::CompletedPart completedPart;
    completedPart.SetPartNumber(partNumber);
    completedPart.SetETag(partContexts[partNumber]->eTag);
    completedMPURequest.AddParts(completedPart);
  }

  CompleteMultiPartCopy(target_bucket.c_str(), target_key.c_str(), uploadID, s3_client, completedMPURequest, status);
  if(TF_GetCode(status) != TF_OK) {
    AbortMultiPartCopy(target_bucket.c_str(), target_key.c_str(), uploadID, s3_client, status);
    if(TF_GetCode(status) != TF_OK) return;
  }
  return;
}

void CompleteMultiPartCopy(const Aws::String& target_bucket, const Aws::String& target_key, const Aws::String& uploadID,
                           const std::shared_ptr<Aws::S3::S3Client>& s3_client,
                           Aws::S3::Model::CompletedMultipartUpload completedMPURequest, TF_Status* status) {
  Aws::S3::Model::CompleteMultipartUploadRequest completeRequest;
  completeRequest.SetBucket(target_bucket);
  completeRequest.SetKey(target_key);
  completeRequest.SetUploadId(uploadID);
  completeRequest.SetMultipartUpload(completedMPURequest);

  auto completeOutcome =
      s3_client->CompleteMultipartUpload(completeRequest);
  if (!completeOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, completeOutcome.GetError());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

void AbortMultiPartCopy(const Aws::String& target_bucket, const Aws::String& target_key, const Aws::String& uploadID,
                        const std::shared_ptr<Aws::S3::S3Client>& s3_client, TF_Status* status) {
  Aws::S3::Model::AbortMultipartUploadRequest abortRequest;
  abortRequest.WithBucket(target_bucket)
      .WithKey(target_key)
      .WithUploadId(uploadID);
  auto abortOutcome = s3_client->AbortMultipartUpload(abortRequest);
  if (!abortOutcome.IsSuccess()) {
    TF_SetStatusFromAWSError(status, abortOutcome.GetError());
    return;
  }
  TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_s3_filesystem
