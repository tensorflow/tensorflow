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

#include "xla/debug_me_context_util.h"

#include <string>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace {

TEST(DebugMeContextUtil, BasicHloPass) {
  // Define a custom HLO pass.
  class TestPass : public HloPassInterface {
   public:
    absl::string_view name() const override { return "hello123123"; }
    absl::StatusOr<bool> Run(HloModule* module,
                             const absl::flat_hash_set<absl::string_view>&
                                 execution_threads) override {
      return false;
    }
    absl::StatusOr<bool> RunOnModuleGroup(
        HloModuleGroup* module_group,
        const absl::flat_hash_set<absl::string_view>& execution_threads)
        override {
      return false;
    }
  };

  TestPass test_pass;
  debug_me_context_util::HloPassDebugMeContext ctx(&test_pass);
  const std::string error_message =
      debug_me_context_util::DebugMeContextToErrorMessageString();

  EXPECT_TRUE(absl::StrContains(error_message, test_pass.name()));
}

}  // namespace
}  // namespace xla
