/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/stream_executor/sycl/sycl_gpu_runtime.h"

#include <gtest/gtest.h>
#include "xla/tsl/platform/status_matchers.h"

namespace stream_executor::sycl {
namespace {

TEST(SyclGpuRuntimeTest, GetDeviceCount) {
  EXPECT_THAT(SyclDevicePool::GetDeviceCount(),
              ::absl_testing::IsOkAndHolds(::testing::Gt(0)));
}

TEST(SyclGpuRuntimeTest, GetDeviceOrdinal) {
  TF_ASSERT_OK_AND_ASSIGN(::sycl::device sycl_device,
                          SyclDevicePool::GetDevice(kDefaultDeviceOrdinal));
  TF_ASSERT_OK_AND_ASSIGN(int device_ordinal,
                          SyclDevicePool::GetDeviceOrdinal(sycl_device));
  EXPECT_EQ(device_ordinal, kDefaultDeviceOrdinal);
}

TEST(SyclGpuRuntimeTest, TestStaticDeviceContext) {
  // Verify that GetDeviceContext returns the same context instance on multiple
  // calls.
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context saved_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  TF_ASSERT_OK_AND_ASSIGN(::sycl::context current_sycl_context,
                          SyclDevicePool::GetDeviceContext());
  EXPECT_EQ(saved_sycl_context, current_sycl_context);
}

}  // namespace
}  // namespace stream_executor::sycl
