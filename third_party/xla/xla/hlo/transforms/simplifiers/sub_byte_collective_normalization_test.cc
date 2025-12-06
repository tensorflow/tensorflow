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

#include "xla/hlo/transforms/simplifiers/sub_byte_collective_normalization.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {

class SubByteCollectiveNormalizationTest
    : public HloHardwareIndependentTestBase {};

TEST_F(SubByteCollectiveNormalizationTest, SkipNonSubByteTypes) {
  ASSERT_OK(RunAndCheckHloRewrite(R"(
e {
 a = s8[4,8]{1,0} parameter(0)
 b = s8[8,8]{1,0} all-gather(a), dimensions={0}
})",
                                  SubByteCollectiveNormalization(),
                                  /*expect_change=*/false));
}

TEST_F(SubByteCollectiveNormalizationTest, SkipNonPacked) {
  ASSERT_OK(RunAndCheckHloRewrite(R"(
e {
 a = s2[16,32]{1,0} parameter(0)
 b = s2[32,32]{1,0} all-gather(a), dimensions={0}
})",
                                  SubByteCollectiveNormalization(),
                                  /*expect_change=*/false));
}

TEST_F(SubByteCollectiveNormalizationTest, SkipOddElementCount) {
  ASSERT_OK(RunAndCheckHloRewrite(R"(
e {
 a = s4[4,9]{1,0:E(4)} parameter(0)
 b = s4[8,9]{1,0:E(4)} all-gather(a), dimensions={0}
})",
                                  SubByteCollectiveNormalization(),
                                  /*expect_change=*/false));
}

TEST_F(SubByteCollectiveNormalizationTest, SkipVariadic) {
  ASSERT_OK(RunAndCheckHloRewrite(R"(
e {
 a = s4[2]{0:E(4)} parameter(0)
 b = s4[2]{0:E(4)} parameter(1)
 c = (s4[2]{0:E(4)}, s4[2]{0:E(4)}) all-to-all(a, b), replica_groups={{2,1},{3,0}}
})",
                                  SubByteCollectiveNormalization(),
                                  /*expect_change=*/false));
}

TEST_F(SubByteCollectiveNormalizationTest, TransformS4AllGather) {
  RunAndFilecheckHloRewrite(R"(
e {
 a = s4[4,8]{1,0:E(4)} parameter(0)
 b = s4[8,8]{1,0:E(4)} all-gather(a), dimensions={0}
})",
                            SubByteCollectiveNormalization(), R"(
CHECK: s4[4,8]{1,0:E(4)} parameter
CHECK-NEXT: s4[4,4,2]{2,1,0:E(4)} bitcast
CHECK-NEXT: s8[4,4]{1,0} bitcast-convert
CHECK-NEXT: s8[8,4]{1,0} all-gather
CHECK-NEXT: s4[8,4,2]{2,1,0:E(4)} bitcast-convert
CHECK-NEXT: s4[8,8]{1,0:E(4)} bitcast
)");
}

TEST_F(SubByteCollectiveNormalizationTest,
       TransformAllGatherWithNonMinorMostLastDim) {
  RunAndFilecheckHloRewrite(R"(
e {
 a = s4[32,16]{0,1:E(4)} parameter(0)
 b = s4[32,16]{0,1:E(4)} all-gather(a), dimensions={0}
})",
                            SubByteCollectiveNormalization(), R"(
CHECK: s4[32,16]{0,1:E(4)} parameter
CHECK-NEXT: s4[16,16,2]{2,0,1:E(4)} bitcast
CHECK-NEXT: s8[16,16]{0,1} bitcast-convert
CHECK-NEXT: s8[16,16]{0,1} all-gather
CHECK-NEXT: s4[16,16,2]{2,0,1:E(4)} bitcast-convert
CHECK-NEXT: s4[32,16]{0,1:E(4)} bitcast
)");
}

TEST_F(SubByteCollectiveNormalizationTest, SkipTinyAllToAll) {
  ASSERT_OK(RunAndCheckHloRewrite(R"(
HloModule m, replica_count=2
e {
  a = u4[2]{0:E(4)} parameter(0)
  b = u4[2]{0:E(4)} all-to-all(a), dimensions={0}
})",
                                  SubByteCollectiveNormalization(),
                                  /*expect_change=*/false));
}

TEST_F(SubByteCollectiveNormalizationTest, TransformF4AllToAll) {
  RunAndFilecheckHloRewrite(R"(
e {
 a = f4e2m1fn[3,6,10]{2,1,0:E(4)} parameter(0)
 b = f4e2m1fn[3,6,10]{2,1,0:E(4)} all-to-all(a), dimensions={1}
})",
                            SubByteCollectiveNormalization(), R"(
CHECK: f4e2m1fn[3,6,10]{2,1,0:E(4)} parameter
CHECK-NEXT: f4e2m1fn[3,6,5,2]{3,2,1,0:E(4)} bitcast
CHECK-NEXT: s8[3,6,5]{2,1,0} bitcast-convert
CHECK-NEXT: s8[3,6,5]{2,1,0} all-to-all
CHECK-NEXT: f4e2m1fn[3,6,5,2]{3,2,1,0:E(4)} bitcast-convert
CHECK-NEXT: f4e2m1fn[3,6,10]{2,1,0:E(4)} bitcast
)");
}

TEST_F(SubByteCollectiveNormalizationTest, TransformU2CollectiveBroadcast) {
  RunAndFilecheckHloRewrite(R"(
e {
 a = u2[5,9,8]{2,0,1:E(2)} parameter(0)
 b = u2[5,9,8]{2,0,1:E(2)} collective-broadcast(a), replica_groups={}
})",
                            SubByteCollectiveNormalization(), R"(
CHECK: u2[5,9,8]{2,0,1:E(2)} parameter
CHECK-NEXT: u2[5,9,2,4]{3,2,0,1:E(2)} bitcast
CHECK-NEXT: s8[5,9,2]{2,0,1} bitcast-convert
CHECK-NEXT: s8[5,9,2]{2,0,1} collective-broadcast
CHECK-NEXT: u2[5,9,2,4]{3,2,0,1:E(2)} bitcast-convert
CHECK-NEXT: u2[5,9,8]{2,0,1:E(2)} bitcast
)");
}
}  // namespace xla
