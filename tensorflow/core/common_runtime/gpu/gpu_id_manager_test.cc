/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

PlatformGpuId TfToPlatformGpuId(TfGpuId tf) {
  PlatformGpuId platform_gpu_id;
  TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf, &platform_gpu_id));
  return platform_gpu_id;
}

TEST(GpuIdManagerTest, Basics) {
  TfGpuId key_0(0);
  PlatformGpuId value_0(0);
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformGpuIdPair(key_0, value_0));
  EXPECT_EQ(value_0, TfToPlatformGpuId(key_0));

  // Multiple calls to map the same value is ok.
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformGpuIdPair(key_0, value_0));
  EXPECT_EQ(value_0, TfToPlatformGpuId(key_0));

  // Map a different TfGpuId to a different value.
  TfGpuId key_1(3);
  PlatformGpuId value_1(2);
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformGpuIdPair(key_1, value_1));
  EXPECT_EQ(value_1, TfToPlatformGpuId(key_1));

  // Mapping a different TfGpuId to the same value is ok.
  TfGpuId key_2(10);
  TF_ASSERT_OK(GpuIdManager::InsertTfPlatformGpuIdPair(key_2, value_1));
  EXPECT_EQ(value_1, TfToPlatformGpuId(key_2));

  // Mapping the same TfGpuId to a different value.
  ASSERT_FALSE(GpuIdManager::InsertTfPlatformGpuIdPair(key_2, value_0).ok());

  // Getting a nonexistent mapping.
  ASSERT_FALSE(GpuIdManager::TfToPlatformGpuId(TfGpuId(100), &value_0).ok());
}

}  // namespace
}  // namespace tensorflow
