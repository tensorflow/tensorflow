
/* Copyright 2022 The OpenXLA Authors.

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
#include "xla/hlo/transforms/simplifiers/broadcast_canonicalizer.h"

#include <optional>

#include <gtest/gtest.h>
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla {
namespace {

class BroadcastCanonicalizerTest : public HloHardwareIndependentTestBase {};

TEST_F(BroadcastCanonicalizerTest, ReshapeBroadcast) {
  const char* hlo = R"(
HloModule fusion.1644

ENTRY fusion.1644 {
  parameter.2 = f32[2,3,2]{2,1,0} parameter(0)
  %broadcast.399 = f32[3,2,8,2]{3,2,1,0} broadcast(%parameter.2), dimensions={1,0,3}
  ROOT %reshape.43 = f32[3,16,1,2]{3,2,1,0} reshape(f32[3,2,8,2]{3,2,1,0} %broadcast.399)
}
)";

  RunAndFilecheckHloRewrite(hlo, BroadcastCanonicalizer{}, R"(
// CHECK: [[parameter_2_0:%[^ ]+]] = f32[2,3,2]{2,1,0} parameter(0)
// CHECK: [[transpose_0:%[^ ]+]] = f32[3,2,2]{2,1,0} transpose([[parameter_2_0]]), dimensions={1,0,2}
// CHECK: [[broadcast_1:%[^ ]+]] = f32[3,2,8,2]{3,2,1,0} broadcast([[transpose_0]]), dimensions={0,1,3}
// CHECK-NOT: transpose
// CHECK: ROOT [[reshape_43_3:%[^ ]+]] = f32[3,16,1,2]{3,2,1,0} reshape([[broadcast_1]])
      )");
}

TEST_F(BroadcastCanonicalizerTest, ReshapeBroadcast22) {
  const char* hlo = R"(
HloModule fusion.1644

ENTRY fusion.1644 {
  parameter.2 = f32[5,6,7]{2,1,0} parameter(0)
  %broadcast.399 = f32[8,7,9,5,6]{4,3,2,1,0} broadcast(%parameter.2), dimensions={3,4,1}
  ROOT %reshape.43 = f32[8,7,45,1,6]{4,3,2,1,0} reshape(%broadcast.399)
}
)";

  RunAndFilecheckHloRewrite(hlo, BroadcastCanonicalizer{}, R"(
// CHECK: [[transpose_0:%[^ ]+]] = f32[7,5,6]{2,1,0} transpose([[parameter_2_1:%[^ ]+]]), dimensions={2,0,1}
// CHECK: [[broadcast_0:%[^ ]+]] = f32[8,7,9,5,6]{4,3,2,1,0} broadcast([[transpose_0]]), dimensions={1,3,4}
// CHECK-NOT: transpose
// CHECK: ROOT [[reshape_43_3:%[^ ]+]] = f32[8,7,45,1,6]{4,3,2,1,0} reshape([[broadcast_0]])
      )");
}

TEST_F(BroadcastCanonicalizerTest, Rank1OperandNoTranspose) {
  const char* hlo = R"(
HloModule fusion.1644

ENTRY fusion.1644 {
  parameter.2 = f32[2]{0} parameter(0)
  ROOT %broadcast.399 = f32[2,3,8,2]{3,2,1,0} broadcast(%parameter.2), dimensions={0}
}
)";

  auto status_or_module = RunAndCheckHloRewrite(hlo, BroadcastCanonicalizer{},
                                                /*expect_change=*/false);
  EXPECT_TRUE(status_or_module.ok());
}

}  // namespace
}  // namespace xla
