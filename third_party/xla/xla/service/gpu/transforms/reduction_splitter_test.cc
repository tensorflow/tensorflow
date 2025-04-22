/* Copyright 2016 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/reduction_splitter.h"

#include <cstdint>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

auto MakeDeviceDescription() {
  stream_executor::DeviceDescription device_description{
      stream_executor::GpuDeviceInfoProto{}};
  device_description.set_threads_per_warp(32);
  return device_description;
}

class ReductionSplitterTest : public HloHardwareIndependentTestBase {
 public:
  auto MakeReductionSplitter(bool ignore_small_dims) const {
    return ReductionSplitter{device_description_,
                             /*ignore_small_dims=*/ignore_small_dims};
  }

 private:
  const stream_executor::DeviceDescription device_description_{
      MakeDeviceDescription()};
};

TEST_F(ReductionSplitterTest, SplitReductionAtDimensionTwo) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f16[6,16,512,64]{3,2,1,0} parameter(0)
    transpose.1781 = f16[6,512,16,64]{3,1,2,0} transpose(param_0), dimensions={0,2,1,3}
    convert.6986 = f32[6,512,16,64]{3,1,2,0} convert(transpose.1781)
    bitcast.2136 = f32[6,16,512,64]{3,2,1,0} bitcast(convert.6986)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[16,64]{1,0} reduce(bitcast.2136, constant_11111), dimensions={0,2}, to_apply=add_computation
  }
  )")
                    .value();
  ASSERT_TRUE(MakeReductionSplitter(/*ignore_small_dims=*/true)
                  .Run(module.get())
                  .value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root_reduction =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(root_reduction,
              GmockMatch(m::Reduce(m::Reduce(), m::Constant())));

  auto* pre_reduction = root_reduction->operand(0);
  EXPECT_THAT(pre_reduction->dimensions(), std::vector<int64_t>({2}));
  EXPECT_THAT(pre_reduction->shape(), ShapeUtil::MakeShape(F32, {6, 16, 64}));
  EXPECT_THAT(root_reduction->dimensions(), std::vector<int64_t>({0}));
  EXPECT_THAT(root_reduction->shape(), ShapeUtil::MakeShape(F32, {16, 64}));
}

TEST_F(ReductionSplitterTest, SplitReductionAtDimensionZero) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[1024,16,512,64,128]{4,3,2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[16,64]{1,0} reduce(param_0, constant_11111), dimensions={2,0,4}, to_apply=add_computation
  }
  )")
                    .value();
  ASSERT_TRUE(MakeReductionSplitter(/*ignore_small_dims=*/false)
                  .Run(module.get())
                  .value());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root_reduction =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(root_reduction,
              GmockMatch(m::Reduce(m::Reduce(), m::Constant())));

  auto* pre_reduction = root_reduction->operand(0);
  EXPECT_THAT(pre_reduction->dimensions(), std::vector<int64_t>({0}));
  EXPECT_THAT(pre_reduction->shape(),
              ShapeUtil::MakeShape(F32, {16, 512, 64, 128}));
  EXPECT_THAT(root_reduction->dimensions(), std::vector<int64_t>({1, 3}));
  EXPECT_THAT(root_reduction->shape(), ShapeUtil::MakeShape(F32, {16, 64}));
}

TEST_F(ReductionSplitterTest, DontSplitReductionWithSmallDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[16,8,1024,8]{3,2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    ROOT reduce.982 = f32[16,1024]{1,0} reduce(param_0, constant_11111), dimensions={3,1}, to_apply=add_computation
  }
  )")
                    .value();
  EXPECT_FALSE(MakeReductionSplitter(/*ignore_small_dims=*/true)
                   .Run(module.get())
                   .value());
  EXPECT_TRUE(MakeReductionSplitter(/*ignore_small_dims=*/false)
                  .Run(module.get())
                  .value());
}

TEST_F(ReductionSplitterTest, DontSplitReductionsWithContiguousDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  add_computation {
    x = f32[] parameter(0)
    y = f32[] parameter(1)
    ROOT add = f32[] add(x, y)
  }

  ENTRY entry_computation {
    param_0 = f32[128,128,64,128]{3,2,1,0} parameter(0)
    constant_11111 = f32[] constant(0)
    // The dimenstions to keep (1 and 2) are contiguous.
    ROOT reduce.982 = f32[128,64]{1,0} reduce(param_0, constant_11111), dimensions={3,0}, to_apply=add_computation
  }
  )")
                    .value();
  EXPECT_FALSE(MakeReductionSplitter(/*ignore_small_dims=*/false)
                   .Run(module.get())
                   .value());
}

}  // namespace
}  // namespace gpu
}  // namespace xla
