/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_gpu_executor.h"

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"

namespace stream_executor {
namespace gpu {
namespace {

TEST(GpuExecutorTest, DeviceDescription) {
  ASSERT_FALSE(cuInit(/*Flags=*/0));
  std::unique_ptr<DeviceDescription> d =
      GpuExecutor::CreateDeviceDescription(/*device_ordinal=*/0).value();
  const std::string &name = d->name();
  if (name == "NVIDIA RTX A6000") {
    EXPECT_EQ(d->l2_cache_size(), 6 * 1024 * 1024);
  } else if (name == "Quadro P1000") {
    EXPECT_EQ(d->l2_cache_size(), 1024 * 1024);
  } else if (name == "Tesla P100-SXM2-16GB") {
    EXPECT_EQ(d->l2_cache_size(), 4 * 1024 * 1024);
  } else {
    VLOG(1) << "L2 cache size not tested for " << name << "; reported value is "
            << d->l2_cache_size();
  }
}

}  // namespace
}  // namespace gpu
}  // namespace stream_executor

#endif  // GOOGLE_CUDA
