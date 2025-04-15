/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/cudnn_custom_call_converter.h"

#include <gtest/gtest.h>
#include "xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {
namespace {

using ConverterTest = HloTestBase;

TEST_F(ConverterTest, CustomCallGetsConvertedToCustomFusion) {
  RunAndFilecheckHloRewrite(R"(
f {
  a = s8[] parameter(0)
  ROOT r = s8[] add(a, a)
}

ENTRY e {
  b = s8[] parameter(0)
  ROOT c = s8[] custom-call(b),
    custom_call_target="__cudnn$fusion", called_computations={f}
})",
                            CuDnnCustomCallConverter(), R"(
; CHECK: ROOT %fusion = s8[] fusion(%b), kind=kCustom, calls=%f
; CHECK-SAME: "fusion_backend_config":{"kind":"__cudnn$fusion"}
                          )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
