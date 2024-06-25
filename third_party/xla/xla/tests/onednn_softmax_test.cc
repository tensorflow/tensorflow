/* Copyright 2023 The OpenXLA Authors.

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
#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#include <utility>

#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "xla/literal.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/service/cpu/onednn_ops_rewriter.h"
#include "xla/service/cpu/onednn_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace cpu {

std::string TestParamsToString(
    const ::testing::TestParamInfo<std::tuple<PrimitiveType, int>>& data) {
  PrimitiveType data_type;
  int batch_size;
  std::tie(data_type, batch_size) = data.param;
  return absl::StrCat(primitive_util::LowercasePrimitiveTypeName(data_type),
                      "_BatchSize", std::to_string(batch_size));
}

class OneDnnSoftmaxTest
    : public HloTestBase,
      public ::testing::WithParamInterface<std::tuple<PrimitiveType, int>> {
 protected:
  const char* onednn_softmax_ =
      R"(
  ; CHECK: custom_call_target="__onednn$softmax"
  )";

  // Test pattern match with OneDnnOpsRewriter pass
  void TestSoftmax(std::string input_hlo_string, int expected_softmax_axis) {
    TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                            ParseAndReturnVerifiedModule(input_hlo_string));
    OneDnnOpsRewriter softmax_rewrite_pass;
    HloInstruction* onednn_softmax;
    OneDnnSoftmaxConfig softmax_config;
    TF_ASSERT_OK_AND_ASSIGN(
        bool changed, this->RunHloPass(&softmax_rewrite_pass, module.get()));
    EXPECT_TRUE(changed);
    EXPECT_THAT(module->entry_computation()->root_instruction(),
                GmockMatch(::xla::match::CustomCall(&onednn_softmax,
                                                    {"__onednn$softmax"})));

    auto backend_config = onednn_softmax->backend_config<BackendConfig>();
    softmax_config.CopyFrom(backend_config->onednn_softmax_config());
    int axis_after_rewrite = softmax_config.softmax_axis();
    EXPECT_EQ(expected_softmax_axis, axis_after_rewrite);
  }
};

// Softmax test with last dimension as axis. In this case, axis = 2
TEST_P(OneDnnSoftmaxTest, SoftmaxGenericTest) {
  PrimitiveType data_type;
  int batch_size;
  std::tie(data_type, batch_size) = GetParam();
  if (!IsSupportedType(data_type)) {
    GTEST_SKIP() << "CPU does not support "
                 << primitive_util::LowercasePrimitiveTypeName(data_type);
  }

  const std::string softmax_hlo_template_string = R"(
        HloModule softmax_module
        region_max {
            Arg_0 = $0[] parameter(0)
            Arg_1 = $0[] parameter(1)
            ROOT maximum = $0[] maximum(Arg_0, Arg_1)
        }
        region_add {
            Arg_0 = $0[] parameter(0)
            Arg_1 = $0[] parameter(1)
            ROOT add = $0[] add(Arg_0, Arg_1)
        }
        ENTRY main {
            Arg_0 = $0[$1,128,30522]{2,1,0} parameter(0)
            neg_inf = $0[] constant(-inf)
            reduce_max = $0[$1,128]{1,0} reduce(Arg_0, neg_inf), dimensions={2}, to_apply=region_max
            reshape.0 = $0[$1,128,1]{2,1,0} reshape(reduce_max)
            broadcast.0 = $0[$1,128,1]{2,1,0} broadcast(reshape.0), dimensions={0,1,2}
            reshape.1 = $0[$1,128]{1,0} reshape(broadcast.0)
            broadcast.1 = $0[$1,128,30522]{2,1,0} broadcast(reshape.1), dimensions={0,1}
            subtract.0 = $0[$1,128,30522]{2,1,0} subtract(Arg_0, broadcast.1)
            exponential = $0[$1,128,30522]{2,1,0} exponential(subtract.0)
            const_zero = $0[] constant(0)
            reduce_add = $0[$1,128]{1,0} reduce(exponential, const_zero), dimensions={2}, to_apply=region_add
            reshape.2 = $0[$1,128,1]{2,1,0} reshape(reduce_add)
            broadcast.2 = $0[$1,128,1]{2,1,0} broadcast(reshape.2), dimensions={0,1,2}
            reshape.3 = $0[$1,128]{1,0} reshape(broadcast.2)
            broadcast.3 = $0[$1,128,30522]{2,1,0} broadcast(reshape.3), dimensions={0,1}
            ROOT divide = $0[$1,128,30522]{2,1,0} divide(exponential, broadcast.3)
        }
    )";

  const std::string softmax_hlo_string = absl::Substitute(
      softmax_hlo_template_string,
      primitive_util::LowercasePrimitiveTypeName(data_type), batch_size);

  TestSoftmax(softmax_hlo_string, /*expected_softmax_axis*/ 2);
}

INSTANTIATE_TEST_SUITE_P(OneDnnSoftmaxTestSuite, OneDnnSoftmaxTest,
                         ::testing::Combine(::testing::ValuesIn({F32, BF16,
                                                                 F16}),
                                            ::testing::Values(1, 16)),
                         TestParamsToString);

TEST_F(OneDnnSoftmaxTest, SoftmaxFP32OnAxisZero) {
  const std::string softmax_hlo_string = R"(
        HloModule softmax_module
        region_max {
          Arg_0 = f32[] parameter(0)
          Arg_1 = f32[] parameter(1)
          ROOT maximum = f32[] maximum(Arg_0, Arg_1)
        }
        region_add {
          Arg_0 = f32[] parameter(0)
          Arg_1 = f32[] parameter(1)
          ROOT add = f32[] add(Arg_0, Arg_1)
        }
        ENTRY main {
          Arg_0 = f32[3,1,1]{2,1,0} parameter(0)
          neg_inf = f32[] constant(-inf)
          reduce_max = f32[1,1]{1,0} reduce(Arg_0, neg_inf), dimensions={0}, to_apply=region_max
          neg_inf.1 = f32[1,1]{1,0} constant({ {-inf} })
          maximum = f32[1,1]{1,0} maximum(reduce_max, neg_inf.1)
          reshape.0 = f32[1,1,1]{2,1,0} reshape(maximum)
          broadcast.0 = f32[1,1,1]{2,1,0} broadcast(reshape.0), dimensions={0,1,2}
          reshape.1 = f32[1,1]{1,0} reshape(broadcast.0)
          broadcast.1 = f32[3,1,1]{2,1,0} broadcast(reshape.1), dimensions={1,2}
          subtract = f32[3,1,1]{2,1,0} subtract(Arg_0, broadcast.1)
          exponential = f32[3,1,1]{2,1,0} exponential(subtract)
          const_zero = f32[] constant(0)
          reduce_add = f32[1,1]{1,0} reduce(exponential, const_zero), dimensions={0}, to_apply=region_add
          reshape.2 = f32[1,1,1]{2,1,0} reshape(reduce_add)
          broadcast.2 = f32[1,1,1]{2,1,0} broadcast(reshape.2), dimensions={0,1,2}
          reshape.3 = f32[1,1]{1,0} reshape(broadcast.2)
          broadcast.3 = f32[3,1,1]{2,1,0} broadcast(reshape.3), dimensions={1,2}
          ROOT divide = f32[3,1,1]{2,1,0} divide(exponential, broadcast.3)
        }
    )";

  TestSoftmax(softmax_hlo_string, /*expected_softmax_axis*/ 0);
}

TEST_F(OneDnnSoftmaxTest, SoftmaxWithBF16ConvertOutputFP32Pattern) {
  if (!IsSupportedType(PrimitiveType::BF16)) {
    GTEST_SKIP() << "CPU does not support BF16.";
  }

  const std::string softmax_hlo_string = R"(
        HloModule softmax_module
        region_max {
            Arg_0 = f32[] parameter(0)
            Arg_1 = f32[] parameter(1)
            ROOT maximum = f32[] maximum(Arg_0, Arg_1)
        }
        region_add {
            Arg_0 = f32[] parameter(0)
            Arg_1 = f32[] parameter(1)
            ROOT add = f32[] add(Arg_0, Arg_1)
        }
        ENTRY main {
            Arg_0 = f32[16,128,30522]{2,1,0} parameter(0)
            neg_inf = f32[] constant(-inf)
            reduce_max = f32[16,128]{1,0} reduce(Arg_0, neg_inf), dimensions={2}, to_apply=region_max
            reshape.0 = f32[16,128,1]{2,1,0} reshape(reduce_max)
            broadcast.0 = f32[16,128,1]{2,1,0} broadcast(reshape.0), dimensions={0,1,2}
            reshape.1 = f32[16,128]{1,0} reshape(broadcast.0)
            broadcast.1 = f32[16,128,30522]{2,1,0} broadcast(reshape.1), dimensions={0,1}
            subtract = f32[16,128,30522]{2,1,0} subtract(Arg_0, broadcast.1)
            exponential = f32[16,128,30522]{2,1,0} exponential(subtract)
            const_zero = f32[] constant(0)
            reduce_add = f32[16,128]{1,0} reduce(exponential, const_zero), dimensions={2}, to_apply=region_add
            reshape.2 = f32[16,128,1]{2,1,0} reshape(reduce_add)
            broadcast.2 = f32[16,128,1]{2,1,0} broadcast(reshape.2), dimensions={0,1,2}
            reshape.3 = f32[16,128]{1,0} reshape(broadcast.2)
            broadcast.3 = f32[16,128,30522]{2,1,0} broadcast(reshape.3), dimensions={0,1}
            divide = f32[16,128,30522]{2,1,0} divide(exponential, broadcast.3)
            ROOT convert = bf16[16,128,30522]{2,1,0} convert(divide)
        }
    )";

  TestSoftmax(softmax_hlo_string, /*expected_softmax_axis=*/2);
}

}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
