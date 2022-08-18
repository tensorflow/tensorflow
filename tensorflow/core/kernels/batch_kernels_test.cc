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

#include "tensorflow/core/kernels/batch_kernels.h"

#include "tensorflow/core/kernels/batch_kernel_test_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
class BatchFunctionKernelTest : public BatchFunctionKernelTestBase {};

TEST_P(BatchFunctionKernelTest, EnableAdaptiveScheduler) {
  TF_EXPECT_OK(Init());
  BatchFunctionKernel* batch_kernel =
      dynamic_cast<BatchFunctionKernel*>(op_kernel());
  EXPECT_EQ(internal::BatchFunctionKernelTestAccess(batch_kernel)
                .enable_adaptive_batch_threads(),
            enable_adaptive_scheduler());
}

INSTANTIATE_TEST_SUITE_P(Params, BatchFunctionKernelTest, ::testing::Bool());

}  // namespace tensorflow
