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

#include "xla/service/gpu/model/coalescing_analysis.h"

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/fusions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::ElementsAre;

class CoalescingTest : public HloTestBase {
 public:
  std::vector<bool> IsReadCoalescedPerOperand(absl::string_view hlo_string) {
    auto module = ParseAndReturnVerifiedModule(hlo_string).value();
    HloInstruction* root = module->entry_computation()->root_instruction();
    return IsReadCoalescedPerOperand(root);
  }

  std::vector<bool> IsReadCoalescedPerOperand(const HloInstruction* root) {
    auto fusion_adaptor = HloFusionAdaptor::ForInstruction(root);
    auto analysis = AnalyzeFusion(*root, device_info_);
    auto emitter = GetFusionEmitter(PreBufferAssignmentFusionInfo{analysis});
    auto fusion = dynamic_cast<KernelFusionInterface*>(emitter.value().get());
    EXPECT_TRUE(emitter.ok());

    CoalescingAnalysis coalescing_analysis(root, root->operands(), analysis,
                                           fusion, &mlir_context_,
                                           /*use_heuristic=*/false);

    std::vector<bool> results;
    for (const HloInstruction* operand : root->operands()) {
      results.push_back(coalescing_analysis.IsReadCoalesced(operand));
    }
    return results;
  }

 protected:
  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext mlir_context_;
};

TEST_F(CoalescingTest, IdentityLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200] parameter(0)
      p1 = f32[100, 200] parameter(1)
      ROOT adthread_x = f32[100, 200] add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200] parameter(0)
      p1 = f32[100, 200] parameter(1)
      ROOT fusion = f32[100, 200] fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

TEST_F(CoalescingTest, RhsTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT exp = f32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT fusion = f32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

TEST_F(CoalescingTest, OutputTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{1, 0} parameter(1)
      ROOT exp = f32[100, 200]{0, 1} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{1, 0} parameter(1)
      ROOT fusion = f32[100, 200]{0, 1} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x * 200)
  // Operand 2: (thread_x) -> (thread_x * 200)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(false, false));
}

TEST_F(CoalescingTest, OutputAndLhsTransposedLayout) {
  absl::string_view ir = R"(
    HloModule m
    fusion {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT exp = f32[100, 200]{1, 0} add(p0, p1)
    }
    ENTRY e {
      p0 = f32[100, 200]{1, 0} parameter(0)
      p1 = f32[100, 200]{0, 1} parameter(1)
      ROOT fusion = f32[100, 200]{1, 0} fusion(p0, p1), kind=kInput, calls=fusion
    }
  )";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  // Operand 2: (thread_x) -> (thread_x * 100)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, false));
}

TEST_F(CoalescingTest, Transpose) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      %input = f32[100, 64, 32] parameter(0)
      ROOT transpose = f32[32, 100, 64] transpose(%input), dimensions={2, 0, 1}
    }

    ENTRY entry {
      %input = f32[100, 64, 32] parameter(0)
      ROOT %fusion = f32[32, 100, 64] fusion(%input), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1:  (thread_x)[s0] -> (thread_x + s0 * 128) for s0 in [0, 7]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, TransposeOnlyOuterDims) {
  absl::string_view ir = R"(
    HloModule module

    fusion {
      %input = f32[100, 32, 64] parameter(0)
      ROOT transpose = f32[32, 100, 64] transpose(%input), dimensions={1, 0, 2}
    }

    ENTRY entry {
      %input = f32[100, 32, 64] parameter(0)
      ROOT %fusion = f32[32, 100, 64] fusion(%input), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1:
  //   (thread_x) -> (thread_x * 4 + s0 + (thread_x floordiv 16) * 1984)
  //   for s0 in [0, 3]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, PadOp) {
  absl::string_view ir = R"(
    HloModule module
    fusion {
      p0 = f32[997, 436] parameter(0)
      p1 = f32[] parameter(1)
      ROOT pad = f32[1024, 512] pad(p0, p1), padding=10_17x24_52
    }
    ENTRY entry {
      p0 = f32[997, 436] parameter(0)
      p1 = f32[] parameter(1)
      ROOT %fusion = f32[1024, 512] fusion(p0, p1), kind=kLoop, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x * 4 + s0 - 4384)
  //   for s0 in [0, 3] and thread_x * 4 + s0 in [24, 459]
  // Operand 2: (thread_x) -> ()
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true, true));
}

TEST_F(CoalescingTest, RowReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,512] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,512] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x + s0 * 32) for s0 in [0, 15]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, MultiRowReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,4] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,64] reduce(%input, %c0), dimensions={2}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,4] parameter(0)
      ROOT %fusion = f32[100,64] fusion(%input), kind=kInput, calls=fusion
  })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x) -> (thread_x)
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, ColumnReduction) {
  absl::string_view ir = R"(
    HloModule module
    add {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT add = f32[] add(p0, p1)
    }
    fusion {
      %input = f32[100,64,32] parameter(0)
      %c0 = f32[] constant(0)
      ROOT reduce = f32[100,32] reduce(%input, %c0),
        dimensions={1}, to_apply=add
    }
    ENTRY entry {
      %input = f32[100,64,32] parameter(0)
      ROOT %fusion = f32[100,32] fusion(%input), kind=kInput, calls=fusion
    })";
  // thread_x to linearized input mapping for thread_x in [0, 31]:
  // Operand 1: (thread_x)[s0] -> (thread_x + s0 * 1024) for s0 in [0, 1]
  EXPECT_THAT(IsReadCoalescedPerOperand(ir), ElementsAre(true));
}

TEST_F(CoalescingTest, VariadicReduceViaLoopEmitter) {
  absl::string_view ir = R"(
    HloModule module
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      max01 = s32[] maximum(p0, p1)
      max23 = s32[] maximum(p2, p3)
      ROOT max = (s32[], s32[]) tuple(max01, max23)
    }
    fusion {
      p0 = s32 [5696,10,4] parameter(0)
      p1 = s32 [5696,10,4] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT reduce = (s32[5696,4], s32[5696,4]) reduce(s32[5696,10,4] p0,
        s32[5696,10,4] p1, s32[] p2, s32[] p3), dimensions={1}, to_apply=max
    }
    ENTRY entry {
      p0 = s32 [5696,10,4] parameter(0)
      p1 = s32 [5696,10,4] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT f = (s32[5696,4], s32[5696,4]) fusion(p0, p1, p2, p3),
          kind=kInput, calls=fusion
    })";
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(true, true, true, true));
}

TEST_F(CoalescingTest, VariadicReduceViaReductionEmitter) {
  absl::string_view ir = R"(
    HloModule module
    max {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      max01 = s32[] maximum(p0, p1)
      max23 = s32[] maximum(p2, p3)
      ROOT max = (s32[], s32[]) tuple(max01, max23)
    }
    fusion {
      p0 = s32[32,40] parameter(0)
      p1 = s32[32,40] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT reduce = (s32[32], s32[32])
        reduce(s32[32,40] p0, s32[32,40] p1, s32[] p2, s32[] p3),
        dimensions={1}, to_apply=max
    }
    ENTRY entry {
      p0 = s32[32,40] parameter(0)
      p1 = s32[32,40] parameter(1)
      p2 = s32[] parameter(2)
      p3 = s32[] parameter(3)
      ROOT f = (s32[32], s32[32]) fusion(p0, p1, p2, p3),
          kind=kInput, calls=fusion
    })";
  EXPECT_THAT(IsReadCoalescedPerOperand(ir),
              ElementsAre(true, true, true, true));
}

TEST_F(CoalescingTest, UnusedParameter) {
  Shape shape = ShapeUtil::MakeShape(F32, {100000});

  auto module = std::make_unique<HloModule>("m", HloModuleConfig{});
  HloComputation::Builder b("b");
  auto p0 = b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto p1 = b.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"));

  HloComputation::Builder sub_builder("subcomp");
  HloInstruction* p0f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "p0f"));
  // p1f is not used.
  HloInstruction* p1f = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "p1f"));
  ASSERT_NE(p1f, nullptr);
  sub_builder.AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kNegate, p0f));

  HloComputation* subcomp = module->AddEmbeddedComputation(sub_builder.Build());
  auto fusion = HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kLoop, {p0, p1}, subcomp);
  b.AddInstruction(std::move(fusion));
  module->AddEntryComputation(b.Build());

  EXPECT_THAT(IsReadCoalescedPerOperand(
                  module->entry_computation()->root_instruction()),
              ElementsAre(true, true));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
