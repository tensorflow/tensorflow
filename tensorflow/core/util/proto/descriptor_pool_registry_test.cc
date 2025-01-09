/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/proto/descriptor_pool_registry.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

struct Value {
  static absl::Status Function(
      tensorflow::protobuf::DescriptorPool const** desc_pool,
      std::unique_ptr<tensorflow::protobuf::DescriptorPool>* owned_desc_pool) {
    return absl::OkStatus();
  }
};

REGISTER_DESCRIPTOR_POOL("TEST POOL 1", Value::Function);
REGISTER_DESCRIPTOR_POOL("TEST POOL 2", Value::Function);
}  // namespace

TEST(DescriptorPoolRegistryTest, TestBasic) {
  EXPECT_EQ(DescriptorPoolRegistry::Global()->Get("NON-EXISTENT"), nullptr);
  auto pool1 = DescriptorPoolRegistry::Global()->Get("TEST POOL 1");
  EXPECT_NE(pool1, nullptr);
  auto pool2 = DescriptorPoolRegistry::Global()->Get("TEST POOL 2");
  EXPECT_NE(pool2, nullptr);
}

}  // namespace tensorflow
