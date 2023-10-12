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

#include "tensorflow/core/kernels/batching_util/threadsafe_status.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

TEST(ThreadSafeStatus, DefaultOk) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(status.status());
}

TEST(ThreadSafeStatus, Update) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(status.status());

  status.Update(errors::FailedPrecondition("original error"));
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);

  status.Update(OkStatus());
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);

  status.Update(errors::Internal("new error"));
  EXPECT_EQ(status.status().code(), error::FAILED_PRECONDITION);
}

TEST(ThreadSafeStatus, Move) {
  ThreadSafeStatus status;
  TF_EXPECT_OK(std::move(status).status());
}

}  // namespace
}  // namespace tensorflow
