/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/hlo/transforms/memory_space_propagation.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class MemorySpacePropagationTest : public HloHardwareIndependentTestBase {
 public:
  MemorySpacePropagationTest()
      : HloHardwareIndependentTestBase(),
        verifier_(/*layout_sensitive=*/false, /*allow_mixed_precision*/ false) {
  }

  absl::Status Verify(HloModule* module) {
    return verifier_.Run(module).status();
  }

 protected:
  // Returns a dataflow analysis for the given module.
  std::unique_ptr<HloDataflowAnalysis> GetDataflowAnalysis(
      const HloModule& module) {
    if (auto status_or =
            HloDataflowAnalysis::Run(module, /*ssa_form=*/false,
                                     /*bitcast_defines_value=*/true);
        status_or.ok()) {
      return std::move(status_or.value());
    }
    return nullptr;
  }

 private:
  HloVerifier verifier_;
};

TEST_F(MemorySpacePropagationTest, NoMemorySpace) {
  absl::string_view hlo_string = R"(
  HloModule NoMemorySpace

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)} copy(%param2)
    %fusion = s32[6]{0:T(128)} fusion(s32[6]{0:T(128)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_FALSE(memory_space_propagation.Run(module.get()).value());
  TF_ASSERT_OK_AND_ASSIGN(auto ref, ParseAndReturnVerifiedModule(hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NonTupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NonTupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)SC(0:3)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, TupleOutput) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)SC(0:3)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)S(1)SC(0:3)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)SC(0:3)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)SC(0:3)} get-tuple-element(%fusion), index=0
    %gte1 = s32[6]{0:T(128)} get-tuple-element(%fusion), index=1
    ROOT %root = s32[6]{0:T(128)} add(%gte0, %gte1)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedInputFusion) {
  // Tests propagating the memory space to nested fusions on the input side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)SC(1:1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)S(1)SC(1:1)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)SC(1:1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, NestedOutputFusion) {
  // Tests propagating the memory space to nested fusions on the output side.
  absl::string_view hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)S(1)SC(1:1)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %add.0 = s32[6]{0:T(128)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)} %param_0.1)
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)SC(1:1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

TEST_F(MemorySpacePropagationTest, BitcastInFusion) {
  absl::string_view hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule TupleOutput

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)S(1)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[6]{0:T(128)S(1)SC(0:3)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)S(1)SC(0:3)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)SC(0:3)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)SC(0:3)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)SC(0:3)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  TF_EXPECT_OK(Verify(module.get()));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests RunOnComputation. The parameters do _not_ get the memory
// space propagated from the operands. The operations in the fusion get the
// memory space propagated from the parameters.
TEST_F(MemorySpacePropagationTest, RunOnComputationPropagateFromParameters) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} negate(%gte_1.3)
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      %param1_copy = s32[6]{0:T(128)S(1)} copy(%param1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1_copy), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)S(1)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} negate(%gte_1.3)
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      %param1_copy = s32[6]{0:T(128)S(1)} copy(%param1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1_copy), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the parameters in nested fusions get the memory space
// propagated from the operands.
TEST_F(MemorySpacePropagationTest, RunOnComputationFromParametersNestedFusion) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %nested_fusion {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      ROOT %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
    }

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} fusion(%gte_1.3), kind=kLoop, calls=%nested_fusion
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }

    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %nested_fusion {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      ROOT %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
    }

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)S(1)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%param_1.3, %param_2.3)
      %gte_1.3 = s32[6]{0:T(128)S(1)} get-tuple-element(%tuple), index=0
      %neg_1.3 = s32[6]{0:T(128)} fusion(%gte_1.3), kind=kLoop, calls=%nested_fusion
      ROOT %root = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }

    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the operations in the fusion get the memory space
// propagated from the output.
TEST_F(MemorySpacePropagationTest, RunOnComputationPropagateFromOutput) {
  absl::string_view hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %neg_1.3 = s32[6]{0:T(128)} negate(%param_1.3)
      ROOT %root = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  absl::string_view expected_hlo_string = R"(
    HloModule NoMemorySpace

    %fused_computation {
      %param_1.3 = s32[6]{0:T(128)} parameter(0)
      %param_2.3 = s32[6]{0:T(128)} parameter(1)
      %neg_1.3 = s32[6]{0:T(128)S(1)} negate(%param_1.3)
      ROOT %root = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%neg_1.3, %param_2.3)
    }
    ENTRY %entry {
      %param0 = s32[6]{0:T(128)} parameter(0)
      %param1 = s32[6]{0:T(128)} parameter(1)
      ROOT %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(%param0, %param1), kind=kLoop, calls=%fused_computation
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));
  auto dataflow_analysis = GetDataflowAnalysis(*module);
  MemorySpacePropagation memory_space_propagation(std::move(dataflow_analysis));
  HloComputation* computation =
      module->GetComputationWithName("fused_computation");
  ASSERT_NE(computation, nullptr);
  EXPECT_TRUE(memory_space_propagation.RunOnComputation(computation));
  TF_ASSERT_OK_AND_ASSIGN(auto ref,
                          ParseAndReturnVerifiedModule(expected_hlo_string));
  EXPECT_EQ(absl::HashOf(*module), absl::HashOf(*ref));
}

// This test tests that the memory space propagation works correctly when there
// is a nested fusion with a shape mismatch. In this test, S(1) must propagate
// from the parameter of the nested fusion fusion.505 to its output shape, and
// from there to the root of the nested fusion %copy.4014.
TEST_F(MemorySpacePropagationTest, NestedFusionShapeMismatchBug) {
  absl::string_view hlo_string =
      R"(HloModule jit_insert.fusion.21.isolated, is_scheduled=true

%copy_fusion.20.clone {
  %input.20 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} parameter(0)
  ROOT %copy.4014 = s4[8,32768,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} copy(%input.20)
}

%fused_computation.434.clone {
  %param_0.777 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} parameter(0)
  %fusion.505 = s4[8,32768,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} fusion(%param_0.777), kind=kLoop, output_to_operand_aliasing={{}: (0, {})}, calls=%copy_fusion.20.clone
  %param_3.751 = pred[]{:T(512)} parameter(3)
  %broadcast.1846 = pred[1,16384,1,256]{3,1,0,2:T(8,128)(4,1)} broadcast(%param_3.751), dimensions={}, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}
  %param_2.1768 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)S(1)} parameter(2)
  %param_1.980 = s32[]{:T(128)S(6)} parameter(1)
  %constant.9791 = s32[]{:T(128)} constant(0), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit"}
  %dynamic-slice.2042 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} dynamic-slice(%fusion.505, %param_1.980, %constant.9791, %constant.9791, %constant.9791), dynamic_slice_sizes={1,16384,1,256}, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}],"is_index_aligned":[true,true,true,true]},"used_scoped_memory_configs":[]}
  %select.912 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} select(%broadcast.1846, %param_2.1768, %dynamic-slice.2042), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}
  ROOT %dynamic-update-slice.455 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} dynamic-update-slice(%fusion.505, %select.912, %param_1.980, %constant.9791, %constant.9791, /*index=5*/%constant.9791), metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"indices_config":{"index_known_bits":[{"zeroes":"0","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"},{"zeroes":"4294967295","ones":"0","bitwidth":"32"}],"is_index_aligned":[true,true,true,true]},"used_scoped_memory_configs":[]}
}

ENTRY %jit_insert.fusion.21.isolated.root {
  %bitcast.1556.hbm = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)} parameter(0)
  %select.32 = s32[]{:T(128)S(6)} parameter(1)
  %collective-permute.56.hbm = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)} parameter(2)
  %and.74 = pred[]{:T(512)} parameter(3)
  %copy = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} copy(%bitcast.1556.hbm)
  %copy.1 = s4[1,16384,1,256]{3,1,0,2:T(8,128)(8,1)E(4)S(1)} copy(%collective-permute.56.hbm)
  %fusion.21 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)S(1)} fusion(%copy, %select.32, %copy.1, %and.74), kind=kLoop, calls=%fused_computation.434.clone, metadata={op_name="jit(insert)/jit(main)/pjit/jit(insert)/jit(main)/jit(insert)/pjit/jit(insert)/jit(main)/jit(insert)/jit(insert)/dynamic_update_slice" stack_frame_id=146}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[{"indices":["0","4"]}]}}
  ROOT %copy.2 = s4[8,32768,1,256]{3,1,0,2:T(64,128)(8,1)E(4)} copy(%fusion.21)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  MemorySpacePropagation memory_space_propagation;
  // %copy.4014 output memory space must get modified to match %fusion.505
  // output shape.
  EXPECT_TRUE(memory_space_propagation.Run(module.get()).value());
  HloComputation* computation =
      module->GetComputationWithName("copy_fusion.20.clone");
  ASSERT_NE(computation, nullptr);
  const HloInstruction* copy = computation->GetInstructionWithName("copy.4014");
  ASSERT_NE(copy, nullptr);
  EXPECT_EQ(copy->shape().layout().memory_space(), 1);
  computation = module->GetComputationWithName("fused_computation.434.clone");
  ASSERT_NE(computation, nullptr);
  const HloInstruction* fusion =
      computation->GetInstructionWithName("fusion.505");
  ASSERT_NE(fusion, nullptr);
  EXPECT_EQ(fusion->shape().layout().memory_space(), 1);
  TF_EXPECT_OK(Verify(module.get()));
}

}  // namespace
}  // namespace xla
