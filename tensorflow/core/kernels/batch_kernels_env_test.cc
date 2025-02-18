/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <gmock/gmock.h>
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/kernels/batch_kernel_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {
// Tests that batch kernel initialization returns error when it's configured to
// use adaptive scheduling yet batching thread pool creation fails.
class BatchFunctionKernelEnvTest
    : public test_util::BatchFunctionKernelTestBase {};

TEST_P(BatchFunctionKernelEnvTest, Basic) {
  tensorflow::setenv("TF_NUM_BATCH_THREADS", "0", 1 /* overwrite */);

  const bool adaptive_scheduler_enabled = GetParam();
  absl::Status status = Init(adaptive_scheduler_enabled);
  if (adaptive_scheduler_enabled) {
    EXPECT_THAT(status, tensorflow::testing::StatusIs(
                            error::FAILED_PRECONDITION,
                            "Failed to create batch threads pool"));
  } else {
    // Initialization is ok since batch kernel doesn't use adaptive
    // scheduler.
    TF_EXPECT_OK(status);
  }
}

INSTANTIATE_TEST_SUITE_P(Params, BatchFunctionKernelEnvTest, ::testing::Bool());

}  // namespace
}  // namespace tensorflow
