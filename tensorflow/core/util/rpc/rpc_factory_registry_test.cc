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

#include "tensorflow/core/util/rpc/rpc_factory_registry.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

struct Value {
  static RPCFactory* Function(OpKernelConstruction* ctx, bool fail_fast,
                              int64 timeout_in_ms) {
    return nullptr;
  }
};

REGISTER_RPC_FACTORY("TEST FACTORY 1", Value::Function);
REGISTER_RPC_FACTORY("TEST FACTORY 2", Value::Function);
}  // namespace

TEST(RPCFactoryRegistryTest, TestBasic) {
  EXPECT_EQ(RPCFactoryRegistry::Global()->Get("NON-EXISTENT"), nullptr);
  auto factory1 = RPCFactoryRegistry::Global()->Get("TEST FACTORY 1");
  EXPECT_NE(factory1, nullptr);
  auto factory2 = RPCFactoryRegistry::Global()->Get("TEST FACTORY 2");
  EXPECT_NE(factory2, nullptr);
}

}  // namespace tensorflow
