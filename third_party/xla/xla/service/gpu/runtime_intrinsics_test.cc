/* Copyright 2023 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using RuntimeIntrinsicsTest = HloTestBase;

TEST_F(RuntimeIntrinsicsTest, NopReturnTokenWorks) {
  constexpr absl::string_view kHloText = R"(
HloModule m

ENTRY e {
  constant = u32[2]{0} constant({0, 1})
  ROOT nop_return_token = token[] custom-call(constant), custom_call_target="NopReturnToken", custom_call_has_side_effect=true, api_version=API_VERSION_STATUS_RETURNING
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          GetOptimizedModule(kHloText));

  // The parameter of the NopReturnToken is not removed.
  EXPECT_EQ(module->entry_computation()->instruction_count(), 2);
  // Can run.
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
