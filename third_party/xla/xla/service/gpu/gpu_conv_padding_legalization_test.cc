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

#include "xla/service/gpu/gpu_conv_padding_legalization.h"

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

using GpuConvPaddingLegalizationTest = HloTestBase;

TEST_F(GpuConvPaddingLegalizationTest, BackwardInputConvolve) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule convolution_module
ENTRY %convolution (operand f64[2,2,2,3]{3,2,1,0}) -> (f64[2,2,4,4]{3,2,1,0}, u8[0]) {
  %operand = f64[2,2,2,3]{3,2,1,0} parameter(0)
  %kernel = f64[2,3,2,3]{3,2,1,0} constant(
  {
    { /*i0=0*/
    { /*i1=0*/
      { 0.29629629629629628, 0.30246913580246915, 0.30864197530864196 },
      { 0.31481481481481483, 0.32098765432098764, 0.3271604938271605 }
    },
    { /*i1=1*/
      { 0.25925925925925924, 0.26543209876543211, 0.27160493827160492 },
      { 0.27777777777777779, 0.2839506172839506, 0.29012345679012347 }
    },
    { /*i1=2*/
      { 0.22222222222222221, 0.22839506172839505, 0.23456790123456789 },
      { 0.24074074074074073, 0.24691358024691357, 0.25308641975308643 }
    }
    },
    { /*i0=1*/
    { /*i1=0*/
      { 0.18518518518518517, 0.19135802469135801, 0.19753086419753085 },
      { 0.20370370370370369, 0.20987654320987653, 0.21604938271604937 }
    },
    { /*i1=1*/
      { 0.14814814814814814, 0.15432098765432098, 0.16049382716049382 },
      { 0.16666666666666666, 0.1728395061728395, 0.17901234567901234 }
    },
    { /*i2=2*/
      { 0.1111111111111111, 0.11728395061728394, 0.12345679012345678 },
      { 0.12962962962962962, 0.13580246913580246, 0.1419753086419753 }
    }
    }
  })
  %reverse = f64[2,3,2,3]{3,2,1,0} reverse(%kernel), dimensions={0,1}
  ROOT %custom-call = (f64[2,2,4,4]{3,2,1,0}, u8[0]{0}) custom-call(f64[2,2,2,3]{3,2,1,0} %operand, f64[2,3,2,3]{3,2,1,0} %reverse), window={size=2x3 stride=2x2 pad=0_0x0_1}, dim_labels=bf01_01io->b01f, custom_call_target="__cudnn$convBackwardInput", backend_config="{\"algorithm\":\"0\",\"tensor_ops_enabled\":false,\"conv_result_scale\":1,\"activation_mode\":\"0\",\"side_input_scale\":0}"
}
                                               )")
                    .value();
  ASSERT_TRUE(GpuConvPaddingLegalization().Run(module.get()).value());
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(
                        m::Slice(m::GetTupleElement(
                            m::CustomCall({kCudnnConvBackwardInputCallTarget},
                                          m::Op(), m::Reverse(m::Constant())),
                            0)),
                        m::GetTupleElement())));
  auto slice = root->operand(0);
  Shape expected_slice_shape = ShapeUtil::MakeShape(F64, {2, 2, 4, 4});
  EXPECT_TRUE(ShapeUtil::Equal(slice->shape(), expected_slice_shape));
  auto conv = slice->operand(0);
  Shape expected_conv_shape = ShapeUtil::MakeShape(F64, {2, 2, 4, 5});
  EXPECT_TRUE(ShapeUtil::Equal(conv->shape(), expected_conv_shape));
}

}  // anonymous namespace
}  // namespace gpu
}  // namespace xla
