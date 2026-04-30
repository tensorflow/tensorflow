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

#include <gtest/gtest.h>
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla.pb.h"

namespace xla::cpu {
namespace {

class YnnE2eTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_CONVOLUTION);
    debug_options.clear_xla_cpu_experimental_ynn_fusion_type();
    return debug_options;
  }
};

TEST_F(YnnE2eTest, DoNotDegroupConvolutionFeatures) {
  const char* matmul_module_str = R"(
  HloModule convolution

  ENTRY %main {
    %lhs = f32[1,7,8,9] parameter(0)
    %rhs = f32[1,5,3,9] parameter(1)
    ROOT %conv = f32[1,4,8,9] convolution(%lhs, %rhs),
        window={size=1x5 stride=2x1 pad=0_0x2_2}, dim_labels=b01f_01io->b01f,
        feature_group_count=3
  })";

  // If the convolution feature group is de-grouped, the shape will change to:
  //   f32[1,4,8,3,3]{4,3,2,1,0}
  // This convolution is supported by YNNPACK, so the shape should not change.
  MatchOptimizedHlo(matmul_module_str,
                    "CHECK: f32[1,4,8,9]{3,2,1,0} convolution");
}

class YnnReduceTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_REDUCE);
    return debug_options;
  }
};

TEST_F(YnnReduceTest, ReduceWindowFollowedByReduce) {
  const char* hlo_text = R"(
  HloModule reduce_window_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    rw = f32[256,256] reduce-window(input, init), window={size=2x2 stride=2x2}, to_apply=add
    ROOT result = f32[] reduce(rw, init), dimensions={0,1}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reduce-window
    CHECK: reduce
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReduceReshape) {
  const char* hlo_text = R"(
  HloModule reduce_reshape

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reduced = f32[512] reduce(input, init), dimensions={1}, to_apply=add
    ROOT result = f32[1,512] reshape(reduced)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reduce
    CHECK: reshape
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReshapeReduce) {
  const char* hlo_text = R"(
  HloModule reshape_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reshaped = f32[262144] reshape(input)
    ROOT result = f32[] reduce(reshaped, init), dimensions={0}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: reshape
    CHECK: reduce
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ReduceConvert) {
  const char* hlo_text = R"(
  HloModule reduce_convert

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = f32[512,512] parameter(0)
    init = f32[] constant(0)
    reduced = f32[512] reduce(input, init), dimensions={1}, to_apply=add
    ROOT result = bf16[512] convert(reduced)
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[reduce:.+]] = {{.+}} reduce({{.+}})
    CHECK: ROOT {{.+}} = {{.+}} convert(%[[reduce]])
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

TEST_F(YnnReduceTest, ConvertReduce) {
  const char* hlo_text = R"(
  HloModule convert_reduce

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY main {
    input = bf16[512,512] parameter(0)
    init = f32[] constant(0)
    converted = f32[512,512] convert(input)
    ROOT result = f32[] reduce(converted, init), dimensions={0,1}, to_apply=add
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: %[[convert:.+]] = {{.+}} convert({{.+}})
    CHECK: ROOT {{.+}} = {{.+}} reduce-window(%[[convert]], {{.+}})
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

class YnnDotTest : public HloPjRtTestBase {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = HloPjRtTestBase::GetDebugOptionsForTest();
    debug_options.add_xla_cpu_experimental_ynn_fusion_type(
        DebugOptions::LIBRARY_FUSION_TYPE_INDIVIDUAL_DOT);
    return debug_options;
  }
};

TEST_F(YnnDotTest, SingleDot) {
  const char* hlo_text = R"(
  HloModule single_dot

  ENTRY main {
    %lhs = f32[256,128] parameter(0)
    %rhs = f32[128,512] parameter(1)
    ROOT %out = f32[256,512] dot(%lhs, %rhs), lhs_contracting_dims={1},
                                              rhs_contracting_dims={0}
  }
  )";

  MatchOptimizedHlo(hlo_text, R"(
    CHECK: dot
    CHECK: ENTRY
    CHECK: kind=kCustom
    CHECK: "kind":"__ynn_fusion"
  )");
}

}  // namespace
}  // namespace xla::cpu
