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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/error_spec.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

struct YnnFusionTestParams {
  std::string in_dtype;
  std::string out_dtype;  // Only used for mixed input/output types.
};

class YnnFusionTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>,
      public ::testing::WithParamInterface<YnnFusionTestParams> {
 public:
  static std::string Name(
      const ::testing::TestParamInfo<YnnFusionTestParams>& info) {
    return absl::StrCat(info.param.in_dtype, "x", info.param.out_dtype);
  }

 protected:
  void RunTest(absl::string_view hlo_template) {
    YnnFusionTestParams params = GetParam();
    std::string hlo_text =
        absl::StrReplaceAll(hlo_template, {{"$dtype", params.in_dtype},
                                           {"$in_dtype", params.in_dtype},
                                           {"$out_dtype", params.out_dtype}});
    bool bf16_compute = params.in_dtype == "bf16" || params.out_dtype == "bf16";
    double tolerance = bf16_compute ? 1e-2 : 1e-7;
    EXPECT_TRUE(RunAndCompareNoHloPasses(
        hlo_text, ErrorSpec{/*aabs=*/tolerance, /*arel=*/tolerance}));
  }
};

TEST_P(YnnFusionTest, AddAndMultiply) {
  constexpr absl::string_view kModuleStr = R"(
    HloModule add_and_multiply

    ynn_fusion {
      %lhs = $dtype[100] parameter(0)
      %rhs = $dtype[100] parameter(1)
      %add = $dtype[100] add(%lhs, %rhs)
      ROOT %mul = $in_dtype[100] multiply(%add, %add)
    }

    ENTRY entry {
      %p0 = $dtype[100] parameter(0)
      %p1 = $dtype[100] parameter(1)
      ROOT %fusion = $dtype[100] fusion(%p0, %p1), kind=kCustom, calls=ynn_fusion,
        backend_config={"fusion_config": {kind: "__ynn_fusion"}}
    })";

  RunTest(kModuleStr);
}

std::vector<YnnFusionTestParams> GetSameTypeTestCases() {
  return std::vector<YnnFusionTestParams>({
      YnnFusionTestParams{"bf16", "bf16"},
      YnnFusionTestParams{"f32", "f32"},
  });
}

INSTANTIATE_TEST_SUITE_P(YnnFusionTestInstantiation, YnnFusionTest,
                         ::testing::ValuesIn(GetSameTypeTestCases()),
                         YnnFusionTest::Name);

}  // namespace
}  // namespace xla::cpu
