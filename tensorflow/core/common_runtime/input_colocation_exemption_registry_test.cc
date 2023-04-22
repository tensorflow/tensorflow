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
#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"

#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

REGISTER_INPUT_COLOCATION_EXEMPTION("op 1");
REGISTER_INPUT_COLOCATION_EXEMPTION("op 2");

}  // namespace

TEST(RPCFactoryRegistryTest, TestBasic) {
  auto exempt_ops = InputColocationExemptionRegistry::Global()->Get();
  EXPECT_EQ(exempt_ops.size(), 2);
  EXPECT_NE(exempt_ops.find("op 1"), exempt_ops.end());
  EXPECT_NE(exempt_ops.find("op 2"), exempt_ops.end());
  EXPECT_EQ(exempt_ops.find("op 3"), exempt_ops.end());
}

}  // namespace tensorflow
