
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
#include "xla/service/broadcast_canonicalizer.h"

#include <functional>
#include <memory>
#include <optional>

#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace {

class BroadcastCanonicalizerTest : public HloTestBase {};

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
// CHECK: [[broadcast_1:%[^ ]+]] = f32[2,3,8,2]{3,2,1,0} broadcast([[parameter_2_0]]), dimensions={0,1,3}
// CHECK: [[transpose_2:%[^ ]+]] = f32[3,2,8,2]{3,2,0,1} transpose([[broadcast_1]]), dimensions={1,0,2,3}
// CHECK: ROOT [[reshape_43_3:%[^ ]+]] = f32[3,16,1,2]{3,2,1,0} reshape([[transpose_2]])
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
// CHECK: [[broadcast_0:%[^ ]+]] = f32[8,5,9,6,7]{4,3,2,1,0} broadcast([[parameter_2_1:%[^ ]+]]), dimensions={1,3,4}
// CHECK: [[transpose_2:%[^ ]+]] = f32[8,7,9,5,6]{1,4,2,3,0} transpose([[broadcast_0]]), dimensions={0,4,2,1,3}
// CHECK: ROOT [[reshape_43_3:%[^ ]+]] = f32[8,7,45,1,6]{4,3,2,1,0} reshape([[transpose_2]])
      )");
}

}  // namespace
}  // namespace xla
