/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/memory_space_propagation.h"

#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class MemorySpacePropagationTest : public HloTestBase {
 public:
  MemorySpacePropagationTest()
      : HloTestBase(),
        verifier_(/*layout_sensitive=*/false, /*allow_mixed_precision*/ false) {
  }

  Status Verify(HloModule* module) { return verifier_.Run(module).status(); }

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
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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
    %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)} get-tuple-element(%fusion), index=0
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
    %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) tuple(%add.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = (s32[6]{0:T(128)S(1)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    %gte0 = s32[6]{0:T(128)S(1)} get-tuple-element(%fusion), index=0
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
    %arg0 = s32[3,2]{0,1:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[6]{0:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[3,2]{0,1:T(128)S(1)} parameter(0)
    ROOT %bitcast = s32[6]{0:T(128)} bitcast(%bf_param)
  }

  %fused_computation {
    %param_1.3 = s32[1]{0:T(128)} parameter(1)
    %constant.2 = s32[]{:T(128)} constant(-2147483648)
    %pad.2 = s32[6]{0:T(128)} pad(s32[1]{0:T(128)} %param_1.3, s32[]{:T(128)} %constant.2), padding=0_5
    %param_2.3 = s32[5]{0:T(128)S(1)} parameter(2)
    %pad.3 = s32[6]{0:T(128)} pad(s32[5]{0:T(128)} %param_2.3, s32[]{:T(128)} %constant.2), padding=1_0
    %maximum.1 = s32[6]{0:T(128)} maximum(s32[6]{0:T(128)} %pad.2, s32[6]{0:T(128)} %pad.3)
    %param_0.1 = s32[3,2]{0,1:T(128)S(1)} parameter(0)
    %fusion.1 = s32[6]{0:T(128)} fusion(%param_0.1), kind=kLoop, calls=bitcast_fusion
    ROOT %add.0 = s32[6]{0:T(128)S(1)} add(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)} %fusion.1)
  }

  ENTRY %entry {
    %param0 = s32[3,2]{0,1:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[3,2]{0,1:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[6]{0:T(128)S(1)} fusion(s32[3,2]{0,1:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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
    %fusion = s32[3,2]{0,1:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
    ROOT %root = s32[3,2]{0,1:T(128)} copy(%fusion)
  }
  )";
  absl::string_view expected_hlo_string = R"(
  HloModule NestedFusion

  %bitcast_fusion {
    %bf_param = s32[6]{0:T(128)} parameter(0)
    ROOT %bitcast = s32[3,2]{0,1:T(128)S(1)} bitcast(%bf_param)
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
    ROOT %fusion.1 = s32[3,2]{0,1:T(128)S(1)} fusion(%add.0), kind=kLoop, calls=bitcast_fusion
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    %fusion = s32[3,2]{0,1:T(128)S(1)} fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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
    %param_0.1 = s32[6]{0:T(128)S(1)} parameter(0)
    %bitcast.0 = s32[6]{0:T(128)} bitcast(s32[6]{0:T(128)S(1)} %param_0.1)
    %multiply.0 = s32[6]{0:T(128)} multiply(s32[6]{0:T(128)} %maximum.1, s32[6]{0:T(128)S(1)} %param_0.1)
    ROOT %tuple = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) tuple(%bitcast.0, %multiply.0)
  }

  ENTRY %entry {
    %param0 = s32[6]{0:T(128)} parameter(0)
    %param1 = s32[1]{0:T(128)} parameter(1)
    %param2 = s32[5]{0:T(128)} parameter(2)
    %arg0 = s32[6]{0:T(128)S(1)} copy(%param0)
    %arg1 = s32[1]{0:T(128)} copy(%param1)
    %arg2 = s32[5]{0:T(128)S(1)} copy(%param2)
    ROOT %fusion = (s32[6]{0:T(128)}, s32[6]{0:T(128)}) fusion(s32[6]{0:T(128)S(1)} %arg0, s32[1]{0:T(128)} %arg1, s32[5]{0:T(128)S(1)} %arg2), kind=kLoop, calls=%fused_computation
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

}  // namespace
}  // namespace xla
