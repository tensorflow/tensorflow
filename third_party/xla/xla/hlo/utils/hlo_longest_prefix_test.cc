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

#include "xla/hlo/utils/hlo_longest_prefix.h"

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

using HloLongestPrefixTest = HloHardwareIndependentTestBase;

TEST_F(HloLongestPrefixTest, CommandBufferName) {
  constexpr absl::string_view hlo_text = R"(
    HloModule cb_module, is_scheduled=true

    double {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] add(p0, p0), metadata={op_name="cb_module/main/fuse/f0/double"}
    }

    square {
      p0 = f32[2,2] parameter(0)
      ROOT add = f32[2,2] multiply(p0, p0), metadata={op_name="cb_module/main/fuse/f1/square"}
    }

    sum {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT sum = f32[2,2] add(p0, p1), metadata={op_name="cb_module/main/fuse/f3/sum"}
    }

    command_buffer {
      p0 = f32[2,2] parameter(0)
      f0 = f32[2,2] fusion(p0), kind=kLoop, calls=double, metadata={op_name="cb_module/main/fuse/f0"}
      f1 = f32[2,2] fusion(p0), kind=kLoop, calls=square, metadata={op_name="cb_module/main/fuse/f1"}
      ROOT f3 = f32[2,2] fusion(f0, f1), kind=kLoop, calls=sum, metadata={op_name="cb_module/main/fuse/f3"}
    }

    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT call = f32[2,2] call(p0), to_apply=command_buffer
    })";

  auto m = ParseAndReturnUnverifiedModule(hlo_text);
  EXPECT_TRUE(m.ok());
  HloModule* module = m->get();
  auto call_op_name_prefix = ::xla::hlo_longest_prefix::GetLongestOpNamePrefix(
      *module->entry_computation()->root_instruction());
  EXPECT_EQ(call_op_name_prefix, "cb_module/main/fuse");
}

}  // namespace
}  // namespace xla
