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
#ifndef TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_SHARED_H_
#define TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_SHARED_H_

#include <aws/core/utils/threading/Executor.h>
#include <aws/s3/S3Client.h>
#include <aws/transfer/TransferManager.h>

#include <memory>

namespace tf_s3_filesystem {
typedef struct S3Shared {
  std::shared_ptr<Aws::S3::S3Client> s3_client;
  std::shared_ptr<Aws::Transfer::TransferManager> transfer_manager;
  std::shared_ptr<Aws::Utils::Threading::PooledThreadExecutor> thread_executor;
} S3Shared;
void GetS3Client(S3Shared* s3_shared);
void GetTransferManager(S3Shared* s3_shared);
void GetExecutor(S3Shared* s3_shared);
}  // namespace tf_s3_filesystem

#endif  // TENSORFLOW_C_EXPERIMENTAL_FILESYSTEM_PLUGINS_S3_S3_SHARED_H_
