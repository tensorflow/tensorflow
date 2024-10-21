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

#include "xla/hlo/ir/hlo_module.h"

#include <memory>
#include <string_view>

#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_module_config.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(HloModuleTest, AbslHashValue) {
  HloModule module1("temp_module", HloModuleConfig());
  HloModule module2("temp_module3", HloModuleConfig());
  EXPECT_EQ(absl::HashOf(module1), absl::HashOf(module2));

  std::string_view hlo = R"(
      HloModule m1
        ENTRY main {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
        ROOT res = f32[] multiply(a, b)
      })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module3,
                          ParseAndReturnUnverifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module4,
                          ParseAndReturnUnverifiedModule(hlo));
  EXPECT_EQ(absl::HashOf(*module3), absl::HashOf(*module4));
  EXPECT_NE(absl::HashOf(module1), absl::HashOf(*module4));
}

TEST(HloModuleTest, MutableOwnedImmutableSharedConfig) {
  HloModuleConfig config1;
  config1.set_device_type("first");
  config1.set_device_memory_size(7);
  HloModule m1("-", config1);
  HloModule m2("-", m1.shared_config(),
               std::make_unique<CompilationEnvironments>());
  EXPECT_EQ(&m1.config(), &m2.config())
      << "Shared config referres to the same object.";
  m1.mutable_config().set_device_type("second");
  EXPECT_NE(&m1.config(), &m2.config()) << "Config is copied on modification.";
  EXPECT_EQ(m1.config().device_type(), "second");
  EXPECT_EQ(m2.config().device_type(), "first");
  EXPECT_EQ(m1.config().device_memory_size(), m2.config().device_memory_size());
}

}  // namespace
}  // namespace xla
