/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dynamic_slice_fusion_rewriter_v2.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/dynamic_slice_annotator.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform_id.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;

using Offset = DynamicSliceFusion::Offset;
using Param = DynamicSliceFusion::Parameter;
using Result = DynamicSliceFusion::Result;

static constexpr absl::string_view kFakeTarget = "fake_target";

DynamicSliceConfig MakeConfig(int64_t loop_index, int64_t offset,
                              int64_t stride) {
  DynamicSliceConfig config;
  config.set_loop_index(loop_index);
  config.set_byte_offset(offset);
  config.set_byte_stride(stride);
  return config;
}

DynamicSliceConfig MakeStaticConfig(int64_t offset) {
  DynamicSliceConfig config;
  config.set_byte_offset(offset);
  config.set_byte_stride(0);
  return config;
}

const HloComputation* FindDsfBody(HloModule* module) {
  for (HloComputation* comp : module->computations()) {
    if (absl::StrContains(comp->name(), "dynamic-slice-fusion")) {
      return comp;
    }
  }
  return nullptr;
}

int64_t CountInstructionsWithOpcode(const HloComputation* computation,
                                    HloOpcode opcode) {
  return absl::c_count_if(computation->instructions(),
                          [opcode](const HloInstruction* instr) {
                            return instr->opcode() == opcode;
                          });
}

class DynamicSliceFusionRewriterV2Test : public HloHardwareIndependentTestBase {
  void SetUp() override {
    auto maybe_name = PlatformUtil::CanonicalPlatformName("gpu");
    CHECK_OK(maybe_name);
    auto maybe_platform_id =
        PlatformUtil::GetPlatformIdFromCanonicalName(maybe_name.value());
    CHECK_OK(maybe_platform_id);
    platform_id_ = maybe_platform_id.value();
  }

 protected:
  stream_executor::PlatformId platform_id() const { return platform_id_; }

  using Options = DynamicSliceFusionRewriterV2::Options;
  using OptLevel = DynamicSliceFusionRewriterV2::OptLevel;

  static Options DefaultOptions(OptLevel opt_level = OptLevel::kO1) {
    Options options;
    options.predicate = [](const HloInstruction* instr) {
      auto* custom_call = DynCast<HloCustomCallInstruction>(instr);
      return custom_call != nullptr &&
             custom_call->custom_call_target() == kFakeTarget;
    };
    options.opt_level = opt_level;
    return options;
  }

  HloPassPipeline MakePipeline(OptLevel opt_level = OptLevel::kO1) {
    return MakePipeline(DefaultOptions(opt_level));
  }

  HloPassPipeline MakePipeline(Options options) {
    HloPassPipeline pipeline("test-pipeline");
    pipeline.AddPass<DynamicSliceAnnotator>();
    pipeline.AddPass<DynamicSliceFusionRewriterV2>(platform_id(),
                                                   std::move(options));
    return pipeline;
  }

 private:
  stream_executor::PlatformId platform_id_;
};

//===----------------------------------------------------------------------===//
// Sliced operand tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, SlicedOperands) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[2,8,8]{2,1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %slice1 = f32[1,8,8]{2,1,0} slice(%p1), slice={[1:2], [0:8], [0:8]}
      %bitcast1 = f32[8,8]{1,0} bitcast(%slice1)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast0, %bitcast1),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   {{.*}} f32[2,8,8]{2,1,0} parameter(0)
    ; CHECK-DAG:   {{.*}} f32[2,8,8]{2,1,0} parameter(1)
    ; CHECK-DAG:   {{.*}} f32[1,8,8]{2,1,0} slice(
    ; CHECK-DAG:   {{.*}} f32[8,8]{1,0} bitcast(
    ; CHECK-DAG:   {{.*}} f32[1,8,8]{2,1,0} slice(
    ; CHECK-DAG:   {{.*}} f32[8,8]{1,0} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0, %p1),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{1, f32_288, f32_188, MakeStaticConfig(256), std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, SlicedOperandsDuplicateSlice) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %slice1 = f32[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
      %bitcast1 = f32[8,8]{1,0} bitcast(%slice1)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast0, %bitcast1),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       {{.*}} = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       {{.*}} = f32[8,8]{1,0} bitcast(
    ; CHECK:       {{.*}} = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[0:1], [0:8], [0:8]}
    ; CHECK:       {{.*}} = f32[8,8]{1,0} bitcast(
    ; CHECK:       ROOT {{.*}} = f32[8,8]{1,0} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} = f32[8,8]{1,0} fusion(%p0),
    ; CHECK:              kind=kCustom
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{0, f32_288, f32_188, MakeStaticConfig(0), std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, NotContiguousSliceNotFused) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[8,8]{1,0} parameter(0)
      %slice0 = f32[4,4]{1,0} slice(%p0), slice={[0:4], [0:4]}
      ROOT %hero = f32[4,4]{1,0} custom-call(%slice0),
        custom_call_target="fake_target"
    }
  )";

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterV2Test, NonNoOpInSliceChainNotFused) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %negate = f32[1,8,8]{2,1,0} negate(%slice0)
      %bitcast0 = f32[8,8]{1,0} bitcast(%negate)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast0),
        custom_call_target="fake_target"
    }
  )";

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), std::nullopt);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DSWithConstantOffset) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[4,8,8]{2,1,0} parameter(0)
      %c0 = s32[] constant(0)
      %c1 = s32[] constant(1)
      %ds = f32[1,8,8]{2,1,0} dynamic-slice(%p0, %c1, %c0, %c0),
        dynamic_slice_sizes={1,8,8}
      %bitcast = f32[8,8]{1,0} bitcast(%ds)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    std::vector<Offset> offsets = {{0, Offset::Constant(1)},
                                   {1, Offset::Constant(0)},
                                   {2, Offset::Constant(0)}};
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeStaticConfig(256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, SlicedOperandWithTupleResult) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[8,8]{1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      ROOT %hero = (f32[8,8]{1,0}, f32[8,8]{1,0}) custom-call(%bitcast0, %p1),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} = f32[2,8,8]{2,1,0} parameter({{.*}})
    ; CHECK:       {{.*}} = f32[1,8,8]{2,1,0} slice({{.*}})
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       {{.*}} = f32[8,8]{1,0} bitcast(
    ; CHECK:       {{.*}} = f32[8,8]{1,0} parameter({{.*}})
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       ROOT {{.*}} tuple(
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0, %p1),
    ; CHECK:              kind=kCustom
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{1, f32_88, f32_88, std::nullopt, std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88,
                                            std::nullopt, std::nullopt},
                                     Result{std::nullopt, 1, f32_88, f32_88,
                                            std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// Dynamic slice operand tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, DynamicSlicedOperands) {
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8] get-tuple-element(p0), index=1
      accum = f32[8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast = f32[8,8] bitcast(ds)
      hero = f32[8,8] custom-call(bitcast, accum),
          custom_call_target="fake_target"
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[8,8]) tuple(next_ivar, input, hero)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8] parameter(0)
      accum = f32[8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[8,8]) tuple(c0, input, accum)
      ROOT while = (s32[], f32[4,8,8], f32[8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  // Captures: input(p0), ivar(p1), accum(p2). Constants sunk into fusion.
  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(Param{0, f32_488, f32_188, MakeConfig(0, 0, 256), offsets},
                    Param{2, f32_88, f32_88, std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// Dynamic update slice result tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, DynamicUpdateSliceResult) {
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8] get-tuple-element(p0), index=1
      output = f32[4,8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast_in = f32[8,8] bitcast(ds)
      hero = f32[8,8] custom-call(bitcast_in),
          custom_call_target="fake_target"
      bitcast_out = f32[1,8,8] bitcast(hero)
      dus = f32[4,8,8] dynamic-update-slice(output, bitcast_out, ivar, c0, c0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[4,8,8]) tuple(next_ivar, input, dus)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8] parameter(0)
      output = f32[4,8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[4,8,8]) tuple(c0, input, output)
      ROOT while = (s32[], f32[4,8,8], f32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  // Captures: input(p0), ivar(p1), output(p2). Constants sunk into fusion.
  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{2, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DUSOnlyNoSlicedInput) {
  // Hero takes an unsliced input and writes output via bitcast → DUS into a
  // stacked buffer. This mirrors the real-world pattern where a transpose
  // fusion produces a gradient slice that is DUS-ed into a stacked buffer.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[8,8] get-tuple-element(p0), index=1
      output = f32[4,8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      hero = f32[8,8] custom-call(input),
          custom_call_target="fake_target"
      bitcast_out = f32[1,8,8] bitcast(hero)
      dus = f32[4,8,8] dynamic-update-slice(output, bitcast_out, ivar, c0, c0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[8,8], f32[4,8,8]) tuple(next_ivar, input, dus)
    }

    condition {
      p0 = (s32[], f32[8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[8,8] parameter(0)
      output = f32[4,8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[8,8], f32[4,8,8]) tuple(c0, input, output)
      ROOT while = (s32[], f32[8,8], f32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  // Captures: input(p0), output(p1), ivar(p2). Constants sunk into fusion.
  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(2)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_88, f32_88, std::nullopt,
                                          std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{1, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DUSWithConstantOffset) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[8,8]{1,0} parameter(0)
      %p1 = f32[4,8,8]{2,1,0} parameter(1)
      %hero = f32[8,8]{1,0} custom-call(%p0),
        custom_call_target="fake_target"
      %bitcast = f32[1,8,8]{2,1,0} bitcast(%hero)
      %c0 = s32[] constant(0)
      ROOT %dus = f32[4,8,8]{2,1,0} dynamic-update-slice(
        %p1, %bitcast, %c0, %c0, %c0)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0, %p1),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_88, f32_88, std::nullopt,
                                          std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    std::vector<Offset> offsets = {{0, Offset::Constant(0)},
                                   {1, Offset::Constant(0)},
                                   {2, Offset::Constant(0)}};
    EXPECT_THAT(results, ElementsAre(Result{1, 0, f32_488, f32_188,
                                            MakeStaticConfig(0), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DUSWithConstantHeroOperand) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[4,8,8]{2,1,0} parameter(0)
      %input = f32[8,8]{1,0} constant({
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1}})
      %hero = f32[8,8]{1,0} custom-call(%input),
        custom_call_target="fake_target"
      %bitcast = f32[1,8,8]{2,1,0} bitcast(%hero)
      %c0 = s32[] constant(0)
      ROOT %dus = f32[4,8,8]{2,1,0} dynamic-update-slice(
        %p0, %bitcast, %c0, %c0, %c0)
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %p0),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_88, f32_88, std::nullopt,
                                          std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    std::vector<Offset> offsets = {{0, Offset::Constant(0)},
                                   {1, Offset::Constant(0)},
                                   {2, Offset::Constant(0)}};
    EXPECT_THAT(results, ElementsAre(Result{1, 0, f32_488, f32_188,
                                            MakeStaticConfig(0), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DUSNotRootNotFused) {
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8] get-tuple-element(p0), index=1
      output = f32[4,8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast_in = f32[8,8] bitcast(ds)
      hero = f32[8,8] custom-call(bitcast_in),
          custom_call_target="fake_target"
      bitcast_out = f32[1,8,8] bitcast(hero)
      dus = f32[4,8,8] dynamic-update-slice(output, bitcast_out, ivar, c0, c0)
      negate = f32[4,8,8] negate(dus)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[4,8,8]) tuple(
        next_ivar, input, negate)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8] parameter(0)
      output = f32[4,8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[4,8,8]) tuple(c0, input, output)
      ROOT while = (s32[], f32[4,8,8], f32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  // Only DS fused (DUS not root). Captures: input(p0), ivar(p1).
  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, MixedSlicedAndUnslicedOperands) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[8,8]{1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast0, %p1),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} = f32[2,8,8]{2,1,0} parameter({{.*}})
    ; CHECK:       {{.*}} = f32[1,8,8]{2,1,0} slice({{.*}})
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       {{.*}} = f32[8,8]{1,0} bitcast(
    ; CHECK:       {{.*}} = f32[8,8]{1,0} parameter({{.*}})
    ; CHECK:       ROOT {{.*}} = f32[8,8]{1,0} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0, %p1),
    ; CHECK:              kind=kCustom
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{1, f32_88, f32_88, std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// Predicate tests
//===----------------------------------------------------------------------===//

// The next four tests use this HLO as input.
constexpr char kSlicedInputAndDusOutputHlo[] = R"(
  HloModule test

  body {
    p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
    ivar = s32[] get-tuple-element(p0), index=0
    input = f32[4,8,8] get-tuple-element(p0), index=1
    output = f32[4,8,8] get-tuple-element(p0), index=2
    c0 = s32[] constant(0)
    ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
        dynamic_slice_sizes={1,8,8}
    bitcast_in = f32[8,8] bitcast(ds)
    hero = f32[8,8] custom-call(bitcast_in),
        custom_call_target="fake_target"
    bitcast_out = f32[1,8,8] bitcast(hero)
    dus = f32[4,8,8] dynamic-update-slice(output, bitcast_out, ivar, c0, c0)
    c1 = s32[] constant(1)
    next_ivar = s32[] add(ivar, c1)
    ROOT result = (s32[], f32[4,8,8], f32[4,8,8]) tuple(next_ivar, input, dus)
  }

  condition {
    p0 = (s32[], f32[4,8,8], f32[4,8,8]) parameter(0)
    ivar = s32[] get-tuple-element(p0), index=0
    c4 = s32[] constant(4)
    ROOT cmp = pred[] compare(ivar, c4), direction=LT
  }

  ENTRY main {
    input = f32[4,8,8] parameter(0)
    output = f32[4,8,8] parameter(1)
    c0 = s32[] constant(0)
    tuple = (s32[], f32[4,8,8], f32[4,8,8]) tuple(c0, input, output)
    ROOT while = (s32[], f32[4,8,8], f32[4,8,8]) while(tuple),
        condition=condition, body=body,
        backend_config={"known_trip_count":{"n":"4"},
                        "known_init_step":{"init":"0","step":"1"},
                        "known_induction_variable":{"tuple_index":"0"}}
  }
)";

TEST_F(DynamicSliceFusionRewriterV2Test, PredicateMismatchNotFused) {
  Options options = DefaultOptions();
  options.predicate = [](const HloInstruction*) { return false; };

  const char* expected = R"(
    ; CHECK: HloModule
    ; CHECK-NOT: dynamic_slice_fusion
  )";

  RunAndFilecheckHloRewrite(kSlicedInputAndDusOutputHlo,
                            MakePipeline(std::move(options)), expected);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       InputAndOutputPredicatesRejectAllSkipsHero) {
  Options options = DefaultOptions();
  options.capture_slice = [](const HloInstruction*, int64_t,
                             const HloInstruction*) { return false; };
  options.capture_update_slice = [](const HloInstruction*,
                                    std::optional<int64_t>,
                                    const HloInstruction*) { return false; };

  const char* expected = R"(
    ; CHECK: HloModule
    ; CHECK-NOT: dynamic_slice_fusion
  )";

  RunAndFilecheckHloRewrite(kSlicedInputAndDusOutputHlo,
                            MakePipeline(std::move(options)), expected);
}

TEST_F(DynamicSliceFusionRewriterV2Test, CanCaptureSlicedInputsOnly) {
  Options options = DefaultOptions();
  options.capture_update_slice = [](const HloInstruction*,
                                    std::optional<int64_t>,
                                    const HloInstruction*) { return false; };

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK-SAME:         custom_call_target="fake_target"
    ; CHECK-NOT:   dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK-SAME:         kind=kCustom
    ; CHECK-SAME:         "name":"dynamic_slice_fusion"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} dynamic-update-slice(
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(kSlicedInputAndDusOutputHlo,
                            MakePipeline(std::move(options)), expected,
                            fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, CanCaptureSlicedOutputsOnly) {
  Options options = DefaultOptions();
  options.capture_slice = [](const HloInstruction*, int64_t,
                             const HloInstruction*) { return false; };

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-NOT:   dynamic-slice(
    ; CHECK:       {{.*}} custom-call(
    ; CHECK-SAME:         custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} fusion(
    ; CHECK-SAME:         kind=kCustom
    ; CHECK-SAME:         "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(2)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_88, f32_88}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{1, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets}));
  };

  RunAndFilecheckHloRewrite(kSlicedInputAndDusOutputHlo,
                            MakePipeline(std::move(options)), expected,
                            fusion_checks);
}

//===----------------------------------------------------------------------===//
// Induction variable offset tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, OffsetAsLinearFunctionOfInductionVar) {
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[8,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      c2 = s32[] constant(2)
      offset = s32[] multiply(ivar, c2)
      ds = f32[1,8,8] dynamic-slice(input, offset, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast = f32[8,8] bitcast(ds)
      hero = f32[8,8] custom-call(bitcast),
          custom_call_target="fake_target"
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[8,8,8]) tuple(next_ivar, input)
    }

    condition {
      p0 = (s32[], f32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[8,8,8]) tuple(c0, input)
      ROOT while = (s32[], f32[8,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[OFFSET:%[^ ]+]] = s32[] multiply(
    ; CHECK:       {{.*}} dynamic-slice({{.*}}, [[OFFSET]],
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  // offset = ivar*2, byte stride dim0 = 256. stride = 2*256 = 512.
  // Captures: input(p0), ivar(p1). Constants sunk into fusion.
  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {
      {0, Offset::Multiply(Offset::Parameter(1), Offset::Constant(2))},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeConfig(0, 0, 512), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       DynamicSliceCapturesSelectOffsetExpression) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      ivar = s32[] parameter(1)
      c0_s32 = s32[] constant(0)
      c1_s32 = s32[] constant(1)
      c2_s32 = s32[] constant(2)
      c7_s32 = s32[] constant(7)
      inc = s32[] add(ivar, c1_s32)
      is_lt = pred[] compare(inc, c2_s32), direction=LT
      dec = s32[] subtract(c7_s32, ivar)
      selected = s32[] select(is_lt, inc, dec)
      ds = f32[1,8,8] dynamic-slice(input, selected, c0_s32, c0_s32),
          dynamic_slice_sizes={1,8,8},
          backend_config={"dynamic_slice_config":{"byte_offset":0,"byte_stride":0}}
      bitcast = f32[8,8] bitcast(ds)
      ROOT hero = f32[8,8] custom-call(bitcast),
          custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[INC:%[^ ]+]] = s32[] add(
    ; CHECK:       [[PRED:%[^ ]+]] = pred[] compare([[INC]],
    ; CHECK-SAME:    direction=LT
    ; CHECK:       [[DEC:%[^ ]+]] = s32[] subtract(
    ; CHECK:       [[SELECT:%[^ ]+]] = s32[] select([[PRED]], [[INC]], [[DEC]])
    ; CHECK:       {{.*}} dynamic-slice({{.*}}, [[SELECT]],
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %ivar),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  auto inc = [] {
    return Offset::Add(Offset::Parameter(1), Offset::Constant(1));
  };
  std::vector<Offset> offsets = {
      {0,
       Offset::Select(
           Offset::Compare(ComparisonDirection::kLt, inc(),
                           Offset::Constant(2)),
           inc(), Offset::Subtract(Offset::Constant(7), Offset::Parameter(1)))},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kAdd), 1);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kCompare), 1);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kSubtract), 1);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kSelect), 1);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeStaticConfig(0), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       RepeatedOffsetExpressionOperandClonedOnce) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      ivar = s32[] parameter(1)
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      offset_base = s32[] add(ivar, c1)
      offset = s32[] subtract(offset_base, offset_base)
      ds = f32[1,8,8] dynamic-slice(input, offset, c0, c0),
          dynamic_slice_sizes={1,8,8},
          backend_config={"dynamic_slice_config":{"byte_offset":0,"byte_stride":0}}
      bitcast = f32[8,8] bitcast(ds)
      ROOT hero = f32[8,8] custom-call(bitcast),
          custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[OFFSET_BASE:%[^ ]+]] = s32[] add(
    ; CHECK:       [[OFFSET:%[^ ]+]] = s32[] subtract([[OFFSET_BASE]], [[OFFSET_BASE]])
    ; CHECK:       {{.*}} dynamic-slice({{.*}}, [[OFFSET]],
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %ivar),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  auto offset_base = [] {
    return Offset::Add(Offset::Parameter(1), Offset::Constant(1));
  };
  std::vector<Offset> offsets = {
      {0, Offset::Subtract(offset_base(), offset_base())},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kAdd), 1);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kSubtract), 1);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeStaticConfig(0), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       DynamicUpdateSliceCapturesOffsetExpression) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      input = f32[8,8] parameter(0)
      output = f32[8,8,8] parameter(1)
      ivar = s32[] parameter(2)
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      hero = f32[8,8] custom-call(input),
          custom_call_target="fake_target"
      bitcast = f32[1,8,8] bitcast(hero)
      offset = s32[] add(ivar, c1)
      ROOT dus = f32[8,8,8] dynamic-update-slice(
          output, bitcast, offset, c0, c0),
          backend_config={"dynamic_slice_config":{"byte_offset":0,"byte_stride":0}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       [[OFFSET:%[^ ]+]] = s32[] add(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice({{.*}}, {{.*}}, [[OFFSET]],
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %output, %ivar),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {
      {0, Offset::Add(Offset::Parameter(2), Offset::Constant(1))},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kAdd), 1);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_88, f32_88, std::nullopt,
                                          std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{1, 0, f32_888, f32_188,
                                            MakeStaticConfig(0), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       SharedOffsetExpressionClonedOnceForDSAndDUS) {
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[8,8,8], f32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[8,8,8] get-tuple-element(p0), index=1
      output = f32[8,8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      offset = s32[] add(ivar, c1)
      ds = f32[1,8,8] dynamic-slice(input, offset, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast_in = f32[8,8] bitcast(ds)
      hero = f32[8,8] custom-call(bitcast_in),
          custom_call_target="fake_target"
      bitcast_out = f32[1,8,8] bitcast(hero)
      dus = f32[8,8,8] dynamic-update-slice(output, bitcast_out, offset, c0, c0)
      ROOT result = (s32[], f32[8,8,8], f32[8,8,8]) tuple(offset, input, dus)
    }

    condition {
      p0 = (s32[], f32[8,8,8], f32[8,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      output = f32[8,8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[8,8,8], f32[8,8,8]) tuple(c0, input, output)
      ROOT while = (s32[], f32[8,8,8], f32[8,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[OFFSET:%[^ ]+]] = s32[] add(
    ; CHECK:       {{.*}} dynamic-slice({{.*}}, [[OFFSET]],
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       ROOT {{.*}} dynamic-update-slice({{.*}}, {{.*}}, [[OFFSET]],
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  std::vector<Offset> offsets = {
      {0, Offset::Add(Offset::Parameter(1), Offset::Constant(1))},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kAdd), 1);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeConfig(0, 256, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{2, 0, f32_888, f32_188,
                                            MakeConfig(0, 256, 256), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       OffsetExpressionStopsAtNestedDynamicSlice) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      indices = s32[4] parameter(1)
      c0 = s32[] constant(0)
      c1 = s32[] constant(1)
      index_slice = s32[1] dynamic-slice(indices, c0), dynamic_slice_sizes={1}
      index_scalar = s32[] reshape(index_slice)
      offset = s32[] add(index_scalar, c1)
      ds = f32[1,8,8] dynamic-slice(input, offset, c0, c0),
          dynamic_slice_sizes={1,8,8},
          backend_config={"dynamic_slice_config":{"byte_offset":0,"byte_stride":0}}
      bitcast = f32[8,8] bitcast(ds)
      ROOT hero = f32[8,8] custom-call(bitcast),
          custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[OFFSET:%[^ ]+]] = s32[] add(
    ; CHECK:       {{.*}} dynamic-slice({{.*}}, [[OFFSET]],
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %index_scalar),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {
      {0, Offset::Add(Offset::Parameter(1), Offset::Constant(1))},
      {1, Offset::Constant(0)},
      {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kDynamicSlice), 1);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kReshape), 0);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kAdd), 1);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeStaticConfig(0), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       UnsupportedOffsetRootIsCapturedAsParameter) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      input = f32[8,8,8] parameter(0)
      ivar = s32[] parameter(1)
      c0 = s32[] constant(0)
      offset = s32[] maximum(ivar, c0)
      ds = f32[1,8,8] dynamic-slice(input, offset, c0, c0),
          dynamic_slice_sizes={1,8,8},
          backend_config={"dynamic_slice_config":{"byte_offset":0,"byte_stride":0}}
      bitcast = f32[8,8] bitcast(ds)
      ROOT hero = f32[8,8] custom-call(bitcast),
          custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%input, %offset),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_888 = ShapeUtil::MakeShape(F32, {8, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    const HloComputation* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    EXPECT_EQ(CountInstructionsWithOpcode(body, HloOpcode::kMaximum), 0);

    auto* hero = DynamicSliceFusion::FindHero(body);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_888, f32_188,
                                          MakeStaticConfig(0), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// Non-parameter source tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, SliceFromNonParameterSource) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %negate = f32[2,8,8]{2,1,0} negate(%p0)
      %slice0 = f32[1,8,8]{2,1,0} slice(%negate), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      ROOT %hero = f32[8,8]{1,0} custom-call(%bitcast0),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       [[B0:%[^ ]+]] = f32[8,8]{1,0} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} custom-call([[B0]]),
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       %negate = f32[2,8,8]{2,1,0} negate(
    ; CHECK:       ROOT {{.*}} = f32[8,8]{1,0} fusion(%negate),
    ; CHECK:              kind=kCustom
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params,
                ElementsAre(Param{0, f32_288, f32_188, MakeStaticConfig(256),
                                  std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// Multiple heroes tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, TwoHeroesSameComputation) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[8,8]{1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[0:1], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %hero0 = f32[8,8]{1,0} custom-call(%bitcast0),
        custom_call_target="fake_target"
      %slice1 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast1 = f32[8,8]{1,0} bitcast(%slice1)
      %hero1 = f32[8,8]{1,0} custom-call(%bitcast1, %p1),
        custom_call_target="fake_target"
      ROOT %tuple = (f32[8,8]{1,0}, f32[8,8]{1,0}) tuple(%hero0, %hero1)
    }
  )";

  const char* expected = R"(
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK-DAG:   {{.*}} fusion(%p0), kind=kCustom
    ; CHECK-DAG:   {{.*}} fusion(%p0, %p1), kind=kCustom
    ; CHECK:     }
  )";

  // Two separate fusions; verify both can be analyzed.
  auto fusion_checks = [](HloModule* module) {
    int dsf_count = 0;
    for (HloComputation* comp : module->computations()) {
      if (!absl::StrContains(comp->name(), "dynamic-slice-fusion")) {
        continue;
      }
      dsf_count++;
      auto* hero = DynamicSliceFusion::FindHero(comp);
      ASSERT_OK_AND_ASSIGN(auto params,
                           DynamicSliceFusion::ResolveParameters(hero));
      EXPECT_FALSE(params.empty());
    }
    EXPECT_EQ(dsf_count, 2);
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, NestedTupleResultNotFused) {
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      ROOT %hero = ((f32[8,8]{1,0}, f32[8,8]{1,0}), f32[8,8]{1,0})
          custom-call(%bitcast0), custom_call_target="fake_target"
    }
  )";

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), std::nullopt);
}

//===----------------------------------------------------------------------===//
// Non-standard layout tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, SlicedOperandColumnMajorLayout) {
  // Layout {0,1,2}: column-major, dim0 most minor, dim2 most major.
  // Byte strides for f32[8,8,2]{0,1,2}: dim0=4, dim1=32, dim2=256.
  // Slice along most-major dim2: [0:8,0:8,1:2] → byte offset = 1*256 = 256.
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[8,8,2]{0,1,2} parameter(0)
      %slice0 = f32[8,8,1]{0,1,2} slice(%p0), slice={[0:8], [0:8], [1:2]}
      %bitcast0 = f32[8,8]{0,1} bitcast(%slice0)
      ROOT %hero = f32[8,8]{0,1} custom-call(%bitcast0),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[8,8,2]{0,1,2} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[8,8,1]{0,1,2} slice([[P0]])
    ; CHECK-SAME:    slice={[0:8], [0:8], [1:2]}
    ; CHECK:       [[B0:%[^ ]+]] = f32[8,8]{0,1} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} = f32[8,8]{0,1} custom-call([[B0]]),
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} = f32[8,8]{0,1} fusion(%p0),
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_882 = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 8, 2}, {0, 1, 2});
  auto f32_881 = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 8, 1}, {0, 1, 2});
  auto f32_88 = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 8}, {0, 1});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params,
                ElementsAre(Param{0, f32_882, f32_881, MakeStaticConfig(256),
                                  std::nullopt}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DynamicSliceNonStandardLayoutWithDUS) {
  // Layout {1,2,0}: minor_to_major=[1,2,0], so dim1 most minor, dim2 next,
  // dim0 most major. Byte strides for f32[4,8,8]{1,2,0}: dim1=4, dim2=32,
  // dim0=256. Slicing dim0 is contiguous because dim0 is most major.
  // DS/DUS along dim0 → config stride = 256.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8]{1,2,0}, f32[4,8,8]{1,2,0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8]{1,2,0} get-tuple-element(p0), index=1
      output = f32[4,8,8]{1,2,0} get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      ds = f32[1,8,8]{1,2,0} dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast_in = f32[8,8]{0,1} bitcast(ds)
      hero = f32[8,8]{0,1} custom-call(bitcast_in),
          custom_call_target="fake_target"
      bitcast_out = f32[1,8,8]{1,2,0} bitcast(hero)
      dus = f32[4,8,8]{1,2,0} dynamic-update-slice(
          output, bitcast_out, ivar, c0, c0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8]{1,2,0}, f32[4,8,8]{1,2,0})
          tuple(next_ivar, input, dus)
    }

    condition {
      p0 = (s32[], f32[4,8,8]{1,2,0}, f32[4,8,8]{1,2,0}) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8]{1,2,0} parameter(0)
      output = f32[4,8,8]{1,2,0} parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8]{1,2,0}, f32[4,8,8]{1,2,0})
          tuple(c0, input, output)
      ROOT while = (s32[], f32[4,8,8]{1,2,0}, f32[4,8,8]{1,2,0})
          while(tuple), condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} dynamic-update-slice(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShapeWithDenseLayout(F32, {4, 8, 8}, {1, 2, 0});
  auto f32_188 = ShapeUtil::MakeShapeWithDenseLayout(F32, {1, 8, 8}, {1, 2, 0});
  auto f32_88 = ShapeUtil::MakeShapeWithDenseLayout(F32, {8, 8}, {0, 1});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{2, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test,
       NonContiguousSliceNonStandardLayoutNotFused) {
  // Layout {1,0,2}: dim1 most minor, dim0 next, dim2 most major.
  // Slicing dim0 [1:2,0:8,0:8] is non-contiguous because dim2 (more major
  // than sliced dim0) has extent 8 != 1.
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{1,0,2} parameter(0)
      %slice0 = f32[1,8,8]{1,0,2} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{0,1} bitcast(%slice0)
      ROOT %hero = f32[8,8]{0,1} custom-call(%bitcast0),
        custom_call_target="fake_target"
    }
  )";

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), std::nullopt);
}

//===----------------------------------------------------------------------===//
// Mixed DUS/passthrough tuple output tests
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, TupleOutputOneDUS) {
  // Hero produces (f32[8,8], f32[8,8]). Only the first output flows through
  // DUS; the second is returned directly (passthrough). The fusion output
  // must be a tuple containing both the DUS result and the passthrough GTE.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = f32[4,8,8] get-tuple-element(p0), index=1
      prev = f32[8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      hero = (f32[8,8], f32[8,8]) custom-call(),
          custom_call_target="fake_target"
      gte0 = f32[8,8] get-tuple-element(hero), index=0
      gte1 = f32[8,8] get-tuple-element(hero), index=1
      bc0 = f32[1,8,8] bitcast(gte0)
      dus = f32[4,8,8] dynamic-update-slice(buf, bc0, ivar, c0, c0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[8,8]) tuple(next_ivar, dus, gte1)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      buf = f32[4,8,8] parameter(0)
      prev = f32[8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[8,8]) tuple(c0, buf, prev)
      ROOT while = (s32[], f32[4,8,8], f32[8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK-SAME:         custom_call_target="fake_target"
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} dynamic-update-slice(
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       ROOT {{.*}} tuple(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK-SAME:         kind=kCustom
    ; CHECK-SAME:         "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    auto* hero = DynamicSliceFusion::FindHero(body);
    ASSERT_NE(hero, nullptr);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, IsEmpty());

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{0, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets},
                                     Result{std::nullopt, 1, f32_88, f32_88,
                                            std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, TupleOutputNoDUS) {
  // Hero produces (f32[8,8], f32[8,8]). Neither output has DUS — both are
  // passthrough. With a sliced input, the rewriter should still fuse.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8] get-tuple-element(p0), index=1
      prev0 = f32[8,8] get-tuple-element(p0), index=2
      prev1 = f32[8,8] get-tuple-element(p0), index=3
      c0 = s32[] constant(0)
      ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bc = f32[8,8] bitcast(ds)
      hero = (f32[8,8], f32[8,8]) custom-call(bc),
          custom_call_target="fake_target"
      gte0 = f32[8,8] get-tuple-element(hero), index=0
      gte1 = f32[8,8] get-tuple-element(hero), index=1
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[8,8], f32[8,8])
          tuple(next_ivar, input, gte0, gte1)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8] parameter(0)
      prev0 = f32[8,8] parameter(1)
      prev1 = f32[8,8] parameter(2)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[8,8], f32[8,8])
          tuple(c0, input, prev0, prev1)
      ROOT while = (s32[], f32[4,8,8], f32[8,8], f32[8,8])
          while(tuple), condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} custom-call(
    ; CHECK-SAME:         custom_call_target="fake_target"
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       ROOT {{.*}} tuple(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK-SAME:         kind=kCustom
    ; CHECK-SAME:         "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    auto* hero = DynamicSliceFusion::FindHero(body);
    ASSERT_NE(hero, nullptr);

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88,
                                            std::nullopt, std::nullopt},
                                     Result{std::nullopt, 1, f32_88, f32_88,
                                            std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, TupleOutputOneDUSOneDeadOutput) {
  // Hero produces (f32[8,8], f32[8,8]). Only the first output is used (via
  // DUS). The second output has no GTE users — it is dead. The fusion must
  // still include both in its output tuple so that buffer allocation has a
  // slot for the second output.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      buf = f32[4,8,8] get-tuple-element(p0), index=1
      c0 = s32[] constant(0)
      hero = (f32[8,8], f32[8,8]) custom-call(),
          custom_call_target="fake_target"
      gte0 = f32[8,8] get-tuple-element(hero), index=0
      bc0 = f32[1,8,8] bitcast(gte0)
      dus = f32[4,8,8] dynamic-update-slice(buf, bc0, ivar, c0, c0)
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8]) tuple(next_ivar, dus)
    }

    condition {
      p0 = (s32[], f32[4,8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      buf = f32[4,8,8] parameter(0)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8]) tuple(c0, buf)
      ROOT while = (s32[], f32[4,8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} custom-call(
    ; CHECK-SAME:         custom_call_target="fake_target"
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       {{.*}} dynamic-update-slice(
    ; CHECK:       {{.*}} get-tuple-element(
    ; CHECK:       ROOT {{.*}} tuple(
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK-SAME:         kind=kCustom
    ; CHECK-SAME:         "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* body = FindDsfBody(module);
    ASSERT_NE(body, nullptr);
    auto* hero = DynamicSliceFusion::FindHero(body);
    ASSERT_NE(hero, nullptr);

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{0, 0, f32_488, f32_188,
                                            MakeConfig(0, 0, 256), offsets},
                                     Result{std::nullopt, 1, f32_88, f32_88,
                                            std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, DynamicSliceOnlyNoResult) {
  const char* hlo = R"(
    HloModule test

    %body {
      %p = (s32[], f32[4,8,8]{2,1,0}) parameter(0)
      %ivar = s32[] get-tuple-element(%p), index=0
      %input = f32[4,8,8]{2,1,0} get-tuple-element(%p), index=1
      %zero = s32[] constant(0)
      %ds = f32[1,8,8]{2,1,0} dynamic-slice(%input, %ivar, %zero, %zero),
        dynamic_slice_sizes={1,8,8}
      %bc = f32[8,8]{1,0} bitcast(%ds)
      %hero = f32[8,8]{1,0} custom-call(%bc),
        custom_call_target="fake_target"
      %one = s32[] constant(1)
      %next = s32[] add(%ivar, %one)
      ROOT %tuple = (s32[], f32[4,8,8]{2,1,0}) tuple(%next, %input)
    }

    %cond {
      %p = (s32[], f32[4,8,8]{2,1,0}) parameter(0)
      %i = s32[] get-tuple-element(%p), index=0
      %limit = s32[] constant(4)
      ROOT %cmp = pred[] compare(%i, %limit), direction=LT
    }

    ENTRY main {
      %zero = s32[] constant(0)
      %buf = f32[4,8,8]{2,1,0} parameter(0)
      %init = (s32[], f32[4,8,8]{2,1,0}) tuple(%zero, %buf)
      ROOT %while = (s32[], f32[4,8,8]{2,1,0}) while(%init),
        body=%body, condition=%cond,
        backend_config={"known_trip_count":{"n":"4"},
                        "known_init_step":{"init":"0","step":"1"},
                        "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[4,8,8]{2,1,0} parameter(0)
    ; CHECK:       [[DS:%[^ ]+]] = f32[1,8,8]{2,1,0} dynamic-slice([[P0]]
    ; CHECK:       [[BC:%[^ ]+]] = f32[8,8]{1,0} bitcast([[DS]])
    ; CHECK:       ROOT {{.*}} = f32[8,8]{1,0} custom-call([[BC]]),
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     %body{{.*}} {
    ; CHECK:       {{.*}} = f32[8,8]{1,0} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(params, ElementsAre(Param{0, f32_488, f32_188,
                                          MakeConfig(0, 0, 256), offsets}));

    ASSERT_OK_AND_ASSIGN(auto results,
                         DynamicSliceFusion::ResolveResults(hero));
    EXPECT_THAT(results, ElementsAre(Result{std::nullopt, 0, f32_88, f32_88}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(), expected, fusion_checks);
}

//===----------------------------------------------------------------------===//
// O2 mode tests — looking through tuples
//===----------------------------------------------------------------------===//

TEST_F(DynamicSliceFusionRewriterV2Test, O2LooksThroughTupleGte) {
  // Pattern: Slice → bitcast → tuple → GTE → hero. In O2 mode the rewriter
  // should look through the tuple/GTE barrier and fuse the Slice. The tuple
  // and GTE are NOT captured into the fusion.
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[8,8]{1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %tuple = (f32[8,8]{1,0}, f32[8,8]{1,0}) tuple(%bitcast0, %p1)
      %gte0 = f32[8,8]{1,0} get-tuple-element(%tuple), index=0
      %gte1 = f32[8,8]{1,0} get-tuple-element(%tuple), index=1
      ROOT %hero = f32[8,8]{1,0} custom-call(%gte0, %gte1),
        custom_call_target="fake_target"
    }
  )";

  // In O1 mode: GTE→tuple blocks the search, no fusion (only Slice has no
  // DynamicSliceConfig, so the annotator doesn't change the module either).
  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO1), std::nullopt);

  // In O2 mode: look through tuple→GTE, find slice, fuse it.
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[2,8,8]{2,1,0} parameter({{.*}})
    ; CHECK:       [[S0:%[^ ]+]] = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       [[B0:%[^ ]+]] = f32[8,8]{1,0} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{1, f32_88, f32_88, std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO2), expected,
                            fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, O2DynamicSliceThroughTupleGte) {
  // Pattern in while body: DS → bitcast → tuple → GTE → hero, with dynamic
  // offsets. O2 mode looks through the tuple barrier and fuses the DS.
  const char* hlo = R"(
    HloModule test

    body {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = f32[4,8,8] get-tuple-element(p0), index=1
      accum = f32[8,8] get-tuple-element(p0), index=2
      c0 = s32[] constant(0)
      ds = f32[1,8,8] dynamic-slice(input, ivar, c0, c0),
          dynamic_slice_sizes={1,8,8}
      bitcast = f32[8,8] bitcast(ds)
      barrier = (f32[8,8], f32[8,8]) tuple(bitcast, accum)
      gte_sliced = f32[8,8] get-tuple-element(barrier), index=0
      gte_accum = f32[8,8] get-tuple-element(barrier), index=1
      hero = f32[8,8] custom-call(gte_sliced, gte_accum),
          custom_call_target="fake_target"
      c1 = s32[] constant(1)
      next_ivar = s32[] add(ivar, c1)
      ROOT result = (s32[], f32[4,8,8], f32[8,8]) tuple(next_ivar, input, hero)
    }

    condition {
      p0 = (s32[], f32[4,8,8], f32[8,8]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c4 = s32[] constant(4)
      ROOT cmp = pred[] compare(ivar, c4), direction=LT
    }

    ENTRY main {
      input = f32[4,8,8] parameter(0)
      accum = f32[8,8] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], f32[4,8,8], f32[8,8]) tuple(c0, input, accum)
      ROOT while = (s32[], f32[4,8,8], f32[8,8]) while(tuple),
          condition=condition, body=body,
          backend_config={"known_trip_count":{"n":"4"},
                          "known_init_step":{"init":"0","step":"1"},
                          "known_induction_variable":{"tuple_index":"0"}}
    }
  )";

  // O1 mode: blocked by tuple barrier — annotator still runs and annotates
  // the DS, but no fusion is created.
  const char* o1_expected = R"(
    ; CHECK:     body
    ; CHECK-NOT:   fusion(
    ; CHECK:     ENTRY
  )";
  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO1), o1_expected);

  // O2 mode: looks through tuple, fuses DS.
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       {{.*}} dynamic-slice(
    ; CHECK:       {{.*}} bitcast(
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     body
    ; CHECK:       {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_488 = ShapeUtil::MakeShape(F32, {4, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});
  std::vector<Offset> offsets = {{0, Offset::Parameter(1)},
                                 {1, Offset::Constant(0)},
                                 {2, Offset::Constant(0)}};

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(Param{0, f32_488, f32_188, MakeConfig(0, 0, 256), offsets},
                    Param{2, f32_88, f32_88, std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO2), expected,
                            fusion_checks);
}

TEST_F(DynamicSliceFusionRewriterV2Test, O2NestedTupleGteLookthrough) {
  // Nested tuple barrier: DS → bitcast → tuple → tuple → GTE → GTE → hero.
  // O2 should look through both levels.
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %inner = (f32[8,8]{1,0}) tuple(%bitcast0)
      %outer = ((f32[8,8]{1,0})) tuple(%inner)
      %gte_outer = (f32[8,8]{1,0}) get-tuple-element(%outer), index=0
      %gte_inner = f32[8,8]{1,0} get-tuple-element(%gte_outer), index=0
      ROOT %hero = f32[8,8]{1,0} custom-call(%gte_inner),
        custom_call_target="fake_target"
    }
  )";

  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK:       [[P0:%[^ ]+]] = f32[2,8,8]{2,1,0} parameter(0)
    ; CHECK:       [[S0:%[^ ]+]] = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       [[B0:%[^ ]+]] = f32[8,8]{1,0} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} custom-call([[B0]]),
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(%p0),
    ; CHECK:              kind=kCustom
    ; CHECK:     }
  )";

  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO2), expected);
}

TEST_F(DynamicSliceFusionRewriterV2Test, O2LooksThroughOptBarrier) {
  // Pattern: Slice → bitcast → tuple → opt-barrier → GTE → hero.
  // O2 should look through the optimization barrier and tuple together.
  const char* hlo = R"(
    HloModule test

    ENTRY main {
      %p0 = f32[2,8,8]{2,1,0} parameter(0)
      %p1 = f32[8,8]{1,0} parameter(1)
      %slice0 = f32[1,8,8]{2,1,0} slice(%p0), slice={[1:2], [0:8], [0:8]}
      %bitcast0 = f32[8,8]{1,0} bitcast(%slice0)
      %tuple = (f32[8,8]{1,0}, f32[8,8]{1,0}) tuple(%bitcast0, %p1)
      %barrier = (f32[8,8]{1,0}, f32[8,8]{1,0}) opt-barrier(%tuple)
      %gte0 = f32[8,8]{1,0} get-tuple-element(%barrier), index=0
      %gte1 = f32[8,8]{1,0} get-tuple-element(%barrier), index=1
      ROOT %hero = f32[8,8]{1,0} custom-call(%gte0, %gte1),
        custom_call_target="fake_target"
    }
  )";

  // O1 mode: blocked by opt-barrier + tuple.
  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO1), std::nullopt);

  // O2 mode: looks through opt-barrier and tuple, finds the slice.
  const char* expected = R"(
    ; CHECK:     %dynamic-slice-fusion{{.*}} {
    ; CHECK-DAG:   [[P0:%[^ ]+]] = f32[2,8,8]{2,1,0} parameter({{.*}})
    ; CHECK:       [[S0:%[^ ]+]] = f32[1,8,8]{2,1,0} slice([[P0]])
    ; CHECK-SAME:    slice={[1:2], [0:8], [0:8]}
    ; CHECK:       [[B0:%[^ ]+]] = f32[8,8]{1,0} bitcast([[S0]])
    ; CHECK:       ROOT {{.*}} custom-call(
    ; CHECK:              custom_call_target="fake_target"
    ; CHECK:     }
    ; CHECK:     ENTRY %main{{.*}} {
    ; CHECK:       ROOT {{.*}} fusion(
    ; CHECK:              kind=kCustom
    ; CHECK:              "name":"dynamic_slice_fusion"
    ; CHECK:     }
  )";

  auto f32_288 = ShapeUtil::MakeShape(F32, {2, 8, 8});
  auto f32_188 = ShapeUtil::MakeShape(F32, {1, 8, 8});
  auto f32_88 = ShapeUtil::MakeShape(F32, {8, 8});

  auto fusion_checks = [&](HloModule* module) {
    auto* hero = DynamicSliceFusion::FindHero(FindDsfBody(module));

    ASSERT_OK_AND_ASSIGN(auto params,
                         DynamicSliceFusion::ResolveParameters(hero));
    EXPECT_THAT(
        params,
        ElementsAre(
            Param{0, f32_288, f32_188, MakeStaticConfig(256), std::nullopt},
            Param{1, f32_88, f32_88, std::nullopt, std::nullopt}));
  };

  RunAndFilecheckHloRewrite(hlo, MakePipeline(OptLevel::kO2), expected,
                            fusion_checks);
}

}  // namespace
}  // namespace xla::gpu
