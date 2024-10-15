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

#include <memory>
#include <string_view>

#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using HloModuleTest = HloTestBase;

TEST_F(HloModuleTest, AbslHashValue) {
  std::unique_ptr<VerifiedHloModule> module1 = CreateNewVerifiedModule();
  std::unique_ptr<VerifiedHloModule> module2 = CreateNewVerifiedModule();
  EXPECT_EQ(absl::HashOf(*module1), absl::HashOf(*module2));

  std::string_view hlo = R"(
      HloModule m1
        ENTRY main {
          a = f32[] parameter(0)
          b = f32[] parameter(1)
        ROOT res = f32[] multiply(a, b)
      })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module3,
                          ParseAndReturnVerifiedModule(hlo));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module4,
                          ParseAndReturnVerifiedModule(hlo));
  EXPECT_EQ(absl::HashOf(*module3), absl::HashOf(*module4));
  EXPECT_NE(absl::HashOf(*module1), absl::HashOf(*module4));
}

}  // namespace
}  // namespace xla
