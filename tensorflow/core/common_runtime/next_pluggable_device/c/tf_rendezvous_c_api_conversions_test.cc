/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_conversions.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "tensorflow/c/tf_status.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"
#include "tensorflow/tsl/framework/allocator.h"

namespace tensorflow {

namespace {

TEST(AllocatorAttributes, ToAndFromC) {
  constexpr uint32_t kValue = 0x1234'5678;
  constexpr int32_t kScopeId = 1;

  tsl::AllocatorAttributes in_attributes;
  in_attributes.value = kValue;
  in_attributes.scope_id = kScopeId;

  TFDevice_AllocatorAttributes c_attributes = ToC(in_attributes);
  EXPECT_EQ(kValue, c_attributes.value);
  EXPECT_EQ(kScopeId, c_attributes.scope_id);

  tsl::AllocatorAttributes out_attributes = FromC(c_attributes);
  EXPECT_EQ(kValue, out_attributes.value);
  EXPECT_EQ(kScopeId, out_attributes.scope_id);
}

}  // namespace

}  // namespace tensorflow
