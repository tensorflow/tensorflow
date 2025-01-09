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

#ifndef TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_

#include <gtest/gtest.h>
#include "tensorflow/core/kernels/batch_kernels.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace test_util {

// A test util for accessing private members of `BatchFunctionKernel`.
class BatchFunctionKernelTestAccess {
 public:
  explicit BatchFunctionKernelTestAccess(const BatchFunctionKernel* kernel);

  bool enable_adaptive_batch_threads() const;

 private:
  const BatchFunctionKernel* const kernel_;
};

class BatchFunctionKernelTestBase : public OpsTestBase,
                                    public ::testing::WithParamInterface<bool> {
 public:
  // Init test fixture with a batch kernel instance.
  absl::Status Init(bool enable_adaptive_scheduler);
};

}  // namespace test_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
