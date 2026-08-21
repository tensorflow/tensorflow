/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, banner, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <memory>

#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(XlaOpsTest, ExecutableRunOptionsHasIntraOpThreadPool) {
  // Create a thread pool and eigen device to simulate OpKernelContext setup
  thread::ThreadPool thread_pool(Env::Default(), "test_pool", /*num_threads=*/2);
  Eigen::ThreadPoolDevice eigen_device(thread_pool.AsEigenThreadPool(), 2);

  // Initialize ExecutableRunOptions (default has nullptr thread pool)
  xla::ExecutableRunOptions run_options;
  EXPECT_EQ(run_options.intra_op_thread_pool(), nullptr);

  // Set the intra-op thread pool from the device, mimicking XlaLocalLaunchBase / XlaRunOp
  run_options.set_intra_op_thread_pool(&eigen_device);

  // Assert thread pool pointer is preserved and non-null
  EXPECT_NE(run_options.intra_op_thread_pool(), nullptr);
  EXPECT_EQ(run_options.intra_op_thread_pool(), &eigen_device);
}

}  // namespace
}  // namespace tensorflow
