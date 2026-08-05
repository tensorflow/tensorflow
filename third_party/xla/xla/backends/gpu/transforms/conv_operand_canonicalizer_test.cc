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

#include "xla/backends/gpu/transforms/conv_operand_canonicalizer.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

class ConvOperandCanonicalizerTest : public HloHardwareIndependentTestBase {};

TEST_F(ConvOperandCanonicalizerTest, CanonicalizesS32ConstantToS8) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = s32[1,14,14,4] parameter(0)
      w0 = s32[3,3,4,8] constant(1) // Values in range [-128, 127], i.e. fits in s8
      ROOT conv = s32[1,12,12,8] convolution(p0, w0), window={size=3x3}, dim_labels=b01f_01io->b01f
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(auto pass_result,
                       RunHloPass(ConvOperandCanonicalizer(), module.get()));
  EXPECT_TRUE(pass_result);

  const char* expected = R"(
      CHECK: %[[W8:.*]] = s8[3,3,4,8]{{.*}} constant
      CHECK: %[[W32:.*]] = s32[3,3,4,8]{{.*}} convert(%[[W8]])
      CHECK: %[[CONV:.*]] = s32[1,12,12,8]{{.*}} convolution(%[[P0:.*]], %[[W32]])
  )";
  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), expected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(ConvOperandCanonicalizerTest, CommutesSpatialOpsWithConvert) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = s8[1,14,14,4] parameter(0)
      c0 = s32[1,14,14,4] convert(p0)
      r0 = s32[1,14,14,4] reshape(c0)
      w0 = s32[3,3,4,8] convert(s8[3,3,4,8] parameter(1))
      ROOT conv = s32[1,12,12,8] convolution(r0, w0), window={size=3x3}, dim_labels=b01f_01io->b01f
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(auto pass_result,
                       RunHloPass(ConvOperandCanonicalizer(), module.get()));
  EXPECT_TRUE(pass_result);

  const char* expected = R"(
      CHECK: %[[P0:.*]] = s8[1,14,14,4]{{.*}} parameter(0)
      CHECK: %[[R8:.*]] = s8[1,14,14,4]{{.*}} reshape(%[[P0]])
      CHECK: %[[CONV_INPUT:.*]] = s32[1,14,14,4]{{.*}} convert(%[[R8]])
      CHECK: %[[CONV:.*]] = s32[1,12,12,8]{{.*}} convolution(%[[CONV_INPUT]], %[[W0:.*]])
  )";
  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), expected));
  EXPECT_TRUE(filecheck_matched);
}

TEST_F(ConvOperandCanonicalizerTest, SimplifiesRedundantConverts) {
  const char* hlo_text = R"hlo(
    HloModule test

    ENTRY test {
      p0 = s8[1,14,14,4] parameter(0)
      c_inner = s32[1,14,14,4] convert(p0)
      c_outer = s32[1,14,14,4] convert(c_inner)
      w0 = s32[3,3,4,8] convert(s8[3,3,4,8] parameter(1))
      ROOT conv = s32[1,12,12,8] convolution(c_outer, w0), window={size=3x3}, dim_labels=b01f_01io->b01f
    }
  )hlo";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(auto pass_result,
                       RunHloPass(ConvOperandCanonicalizer(), module.get()));
  EXPECT_TRUE(pass_result);

  const char* expected = R"(
      CHECK: %[[P0:.*]] = s8[1,14,14,4]{{.*}} parameter(0)
      CHECK: %[[W0:.*]] = s32[3,3,4,8]{{.*}} convert
      CHECK: %[[CONV:.*]] = s32[1,12,12,8]{{.*}} convolution(%[[CLEAN_CONVERT:.*]], %[[W0]])
  )";
  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), expected));
  EXPECT_TRUE(filecheck_matched);
}

}  // namespace
}  // namespace xla::gpu
