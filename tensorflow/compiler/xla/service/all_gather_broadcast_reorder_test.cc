/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_gather_broadcast_reorder.h"

#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {
namespace {

namespace m = xla::testing::opcode_matchers;

class AllGatherBroadcastReorderTest : public HloTestBase {
 public:
  enum class PassOutput { NoChange, NonUniformAGPattern, UniformAGPattern };
  void RunPass(absl::string_view hlo_module, PassOutput expected_output) {
    TF_ASSERT_OK_AND_ASSIGN(auto module,
                            ParseAndReturnVerifiedModule(hlo_module));
    auto changed = AllGatherBroadcastReorder().Run(module.get());
    ASSERT_TRUE(changed.ok());

    if (expected_output == PassOutput::NoChange) {
      EXPECT_FALSE(changed.value());
    } else {
      EXPECT_TRUE(changed.value());
      if (expected_output == PassOutput::NonUniformAGPattern) {
        EXPECT_THAT(module->entry_computation()->root_instruction(),
                    m::Broadcast(m::AllGather(m::Parameter())));
      } else {
        EXPECT_THAT(
            module->entry_computation()->root_instruction(),
            m::Reshape(m::Broadcast(m::AllGather(m::Reshape(m::Parameter())))));
      }
    }
  }
};

TEST_F(AllGatherBroadcastReorderTest, Simple_GatherAlongNonUniformDim) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY main {
  x = f32[128, 5] parameter(0)
  bc = f32[5, 4, 8, 128] broadcast(x), dimensions={3, 0}
  ROOT ag = f32[5, 4, 8, 256] all-gather(bc), dimensions={3}, replica_groups={{0, 1}}
}
)";
  RunPass(hlo_string, PassOutput::NonUniformAGPattern);
}

TEST_F(AllGatherBroadcastReorderTest, Simple_GatherAlongUniformDim) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY main {
  x = f32[128, 5] parameter(0)
  bc = f32[5, 4, 8, 128] broadcast(x), dimensions={3, 0}
  ROOT ag = f32[5, 12, 8, 128] all-gather(bc), dimensions={1}, replica_groups={{0, 1, 2}}
}
)";
  RunPass(hlo_string, PassOutput::UniformAGPattern);
}

TEST_F(AllGatherBroadcastReorderTest, Simple_GatherBroadcastScalar) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY main {
  x = f32[] parameter(0)
  bc = f32[4, 8] broadcast(x), dimensions={}
  ROOT ag = f32[12, 8] all-gather(bc), dimensions={0}, replica_groups={{0, 1, 2}}
}
)";
  RunPass(hlo_string, PassOutput::UniformAGPattern);
}

TEST_F(AllGatherBroadcastReorderTest, T5Test) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY main {
  x = f32[128] parameter(0)
  bc = f32[1,4,84,128]{3,2,1,0} broadcast(x), dimensions={3}
  ROOT ag = f32[8,4,84,128]{3,2,1,0} all-gather(bc), channel_id=6,
                                     replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
}
)";
  RunPass(hlo_string, PassOutput::UniformAGPattern);
}

TEST_F(AllGatherBroadcastReorderTest, FailedMatch) {
  absl::string_view hlo_string = R"(
HloModule m

ENTRY main {
  x = f32[1,4,84,128] parameter(0)
  ROOT ag = f32[8,4,84,128]{3,2,1,0} all-gather(x), channel_id=6,
                                     replica_groups={{0,1,2,3,4,5,6,7}}, dimensions={0}, use_global_device_ids=true
}
)";
  RunPass(hlo_string, PassOutput::NoChange);
}

}  // namespace
}  // namespace xla
