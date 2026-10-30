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

#include <string>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "xla/primitive_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

class ConvSyclTest : public HloInterpreterReferenceMixin<HloTestBase>,
                     public ::testing::WithParamInterface<PrimitiveType> {
 protected:
  PrimitiveType dtype_;
  std::string dtypeString_;
  float atol_;
  float rtol_;

  ConvSyclTest() {
    dtype_ = GetParam();
    atol_ = rtol_ = (dtype_ == F32) ? 1e-4 : 1e-2;
    dtypeString_ = primitive_util::LowercasePrimitiveTypeName(dtype_);
  }
};

TEST_P(ConvSyclTest, Simple2DTest1) {
  const absl::string_view outline = R"(
  HloModule convolution.test

  ENTRY convolution.test {
    arg.0 = $dtype[1,22,22,1] parameter(0)
    arg.1 = $dtype[8,8,1,1] parameter(1)
    ROOT convolution.0 = $dtype[1,11,11,1] convolution(arg.0, arg.1),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, SimpleScalarTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test

  ENTRY convolution.test {
    arg.0 = $dtype[1,22,22,1] parameter(0)
    arg.1 = $dtype[1] parameter(1)
    reshape.1 = $dtype[1,1,1,1] reshape(arg.1)
    ROOT convolution.0 = $dtype[1,14,14,1] convolution(arg.0, reshape.1),
          window={size=1x1 stride=2x2 pad=3_3x3_3}, dim_labels=b01f_01io->b01f
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv2DWithSmallBiasTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.constant.bias

  ENTRY convolution.test.with.bias {
    arg.0 = $dtype[1,32,224,224] parameter(0)
    arg.1 = $dtype[64,32,3,3] parameter(1)
    conv = $dtype[1,64,222,222] convolution(arg.0, arg.1),
          window={size=3x3}, dim_labels=bf01_oi01->bf01
    bias = $dtype[64] constant({...})
    broadcasted_bias = $dtype[1,64,222,222] broadcast(bias), dimensions={1}
    ROOT add = $dtype[1,64,222,222] add(conv, broadcasted_bias)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv2DWithBiasAndReluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.bias.relu.test

  ENTRY convolution.bias.relu.test {
    arg0.1 = $dtype[1,1,22,22] parameter(0)
    arg0.2 = $dtype[10,1,8,8] parameter(1)
    convolution.0 = $dtype[1,10,11,11] convolution(arg0.1, arg0.2),
          window={size=8x8 stride=2x2 pad=3_3x3_3}, dim_labels=bf01_oi01->bf01
    const.0 = $dtype[10] constant(15)
    bcast.1 = $dtype[1,10,11,11] broadcast(const.0), dimensions={1}
    add.0 = $dtype[1,10,11,11] add(convolution.0, bcast.1)
    const.1 = $dtype[] constant(0)
    bcast.2 = $dtype[1,10,11,11] broadcast(const.1), dimensions={}
    ROOT maximum.1 = $dtype[1,10,11,11] maximum(add.0, bcast.2)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Simple3DTest1) {
  const absl::string_view outline = R"(
  HloModule convolution.test
  ENTRY convolution.test {
    p0 = $dtype[8,1,4,5,5] parameter(0)
    p1 = $dtype[32,1,3,3,3] parameter(1)
    ROOT conv = $dtype[8,32,4,5,5] convolution(p0, p1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=bf012_oi012->bf012
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv3DWithBiasTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.bias

  ENTRY convolution.test.with.bias {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    ROOT add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
})";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv3DReluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.with.relu

  ENTRY convolution.test.with.relu {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    const.1 = $dtype[] constant(0)
    bcast.2 = $dtype[15,4,5,5,64] broadcast(const.1), dimensions={}
    ROOT maximum.1 = $dtype[15,4,5,5,64] maximum(conv, bcast.2)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv3DWithBiasAndEluTest) {
  const absl::string_view outline = R"(
  HloModule convolution.test.bias.elu

  ENTRY convolution.test.bias.elu {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
    const.0 = $dtype[] constant(0)
    broadcast.0 = $dtype[15,4,5,5,64] broadcast(const.0), dimensions={}
    compare.0 = pred[15,4,5,5,64] compare(add, broadcast.0), direction=GT
    exp-min-one.0 = $dtype[15,4,5,5,64] exponential-minus-one(add)
    ROOT select.0 = $dtype[15,4,5,5,64] select(compare.0, add, exp-min-one.0)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, Conv3DWithBiasAndRelu6Test) {
  const absl::string_view outline = R"(
  HloModule convolution.test.bias.relu6

  ENTRY convolution.test.bias.relu6 {
    arg.0 = $dtype[15,4,5,5,28] parameter(0)
    arg.1 = $dtype[3,3,3,28,64] parameter(1)
    conv = $dtype[15,4,5,5,64] convolution(arg.0, arg.1),
          window={size=3x3x3 pad=1_1x1_1x1_1}, dim_labels=b012f_012io->b012f
    bias = $dtype[64] parameter(2)
    broadcasted_bias = $dtype[15,4,5,5,64] broadcast(bias), dimensions={4}
    add = $dtype[15,4,5,5,64] add(conv, broadcasted_bias)
    const.0 = $dtype[] constant(0)
    broadcast.0 = $dtype[15,4,5,5,64] broadcast(const.0), dimensions={}
    const.1 = $dtype[] constant(6)
    broadcast.1 = $dtype[15,4,5,5,64] broadcast(const.1), dimensions={}
    ROOT clamp.0 = $dtype[15,4,5,5,64] clamp(broadcast.0, add, broadcast.1)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(outline, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, FuseAlpha) {
  const std::string module_str = R"(
    HloModule Test

    ENTRY Test {
      input = $dtype[1,5,4,4] parameter(0)
      filter = $dtype[2,2,5,8] parameter(1)
      alpha = $dtype[] constant(6)
      alpha_broadcast = $dtype[1,8,5,5] broadcast(alpha), dimensions={}
      conv = $dtype[1,8,5,5] convolution(input, filter),
               window={size=2x2 pad=1_1x1_1},
               dim_labels=bf01_01io->bf01
      ROOT root = multiply(conv, alpha_broadcast)
    })";

  const std::string hlo_module =
      absl::StrReplaceAll(module_str, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

TEST_P(ConvSyclTest, BackwardPass) {
  const absl::string_view module_str = R"(
  HloModule Test

  ENTRY Test {
    arg.0 = $dtype[8,32,32,16]{3,2,1,0} parameter(0)
    arg.1 = $dtype[3,3,16,32]{3,2,1,0} parameter(1)
    conv.0 = $dtype[8,32,32,32]{3,2,1,0} convolution(arg.0, arg.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
    arg.2 = $dtype[] parameter(2)
    broadcast = $dtype[8,32,32,32]{3,2,1,0} broadcast(arg.2), dimensions={}
    mul.0 = $dtype[8,32,32,32]{3,2,1,0} multiply(conv.0, broadcast)
    mul.1 = $dtype[8,32,32,32]{3,2,1,0} multiply(broadcast, conv.0)
    add.1 = $dtype[8,32,32,32]{3,2,1,0} add(mul.0, mul.1)
    reverse.1 = $dtype[3,3,16,32]{3,2,1,0} reverse(arg.1), dimensions={0,1}
    conv.1 = $dtype[8,32,32,16]{3,2,1,0} convolution(add.1, reverse.1), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01oi->b01f
    conv.2 = $dtype[3,3,16,32]{3,2,1,0} convolution(arg.0, add.1), window={size=32x32 pad=1_1x1_1}, dim_labels=f01b_i01o->01bf
    ROOT tuple.1 = ($dtype[8,32,32,16]{3,2,1,0}, $dtype[3,3,16,32]{3,2,1,0}) tuple(conv.1, conv.2)
  })";

  const std::string hlo_module =
      absl::StrReplaceAll(module_str, {{"$dtype", dtypeString_}});
  EXPECT_TRUE(RunAndCompare(hlo_module, ErrorSpec{atol_, rtol_}));
}

INSTANTIATE_TEST_SUITE_P(
    SyclConvSyclTestSuite, ConvSyclTest, ::testing::Values(F32, BF16, F16),
    [](const ::testing::TestParamInfo<ConvSyclTest::ParamType>& info) {
      auto test_name = primitive_util::LowercasePrimitiveTypeName(info.param);
      absl::AsciiStrToUpper(&test_name);
      return test_name;
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
