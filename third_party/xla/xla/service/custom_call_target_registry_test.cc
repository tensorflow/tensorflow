/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/custom_call_target_registry.h"

#include "xla/service/custom_call_status.h"
#include "xla/test.h"

namespace xla {
namespace {

using ::testing::_;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

void custom_call(void*, const void**, XlaCustomCallStatus*) {}
void custom_call2(void*, const void**, XlaCustomCallStatus*) {}

TEST(CustomCallRegistryTest, Registers) {
  CustomCallTargetRegistry registry;
  EXPECT_EQ(registry.Lookup("custom_call", "Host"), nullptr);
  registry.Register("custom_call", reinterpret_cast<void*>(custom_call),
                    "Host");
  EXPECT_EQ(custom_call, registry.Lookup("custom_call", "Host"));
  // A registration with a different name is fine.
  registry.Register("custom_call2", reinterpret_cast<void*>(&custom_call),
                    "Host");

  EXPECT_EQ(registry.Lookup("custom_call", "CUDA"), nullptr);
  // A registration on a different platform is fine.
  registry.Register("custom_call", reinterpret_cast<void*>(custom_call),
                    "CUDA");
  EXPECT_EQ(custom_call, registry.Lookup("custom_call", "CUDA"));

  // A second registration of the same function is fine.
  registry.Register("custom_call", reinterpret_cast<void*>(custom_call),
                    "Host");

  EXPECT_THAT(
      registry.registered_symbols("Host"),
      UnorderedElementsAre(Pair("custom_call", _), Pair("custom_call2", _)));
  EXPECT_THAT(registry.registered_symbols("CUDA"),
              UnorderedElementsAre(Pair("custom_call", _)));
}

TEST(CustomCallRegistryDeathTest, RejectsDuplicateRegistrations) {
  CustomCallTargetRegistry registry;
  registry.Register("custom_call", reinterpret_cast<void*>(custom_call),
                    "Host");
  EXPECT_DEATH(registry.Register("custom_call",
                                 reinterpret_cast<void*>(custom_call2), "Host"),
               "Duplicate custom call");
}

}  // namespace
}  // namespace xla
