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

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace xla_compile {
namespace {

class XlaCompileTest : public HloPjRtTestBase {
 public:
  void LoadAndRunExecutable(absl::string_view path_to_serialized_aot_result,
                            absl::Span<const Literal* const> args,
                            const Literal& expected) {
    std::string path = tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "service",
                                         path_to_serialized_aot_result);
    std::string serialized_aot_result;
    TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), path,
                                       &serialized_aot_result));

    auto* pjrt_runner = absl::down_cast<HloRunnerPjRt*>(&test_runner());
    ASSERT_TRUE(pjrt_runner != nullptr);
    ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<OpaqueExecutable> executable,
        pjrt_runner->DeserializeExecutable(serialized_aot_result));

    // Run loaded executable.
    ASSERT_OK_AND_ASSIGN(Literal result, pjrt_runner->ExecuteWithExecutable(
                                             executable.get(), args));
    EXPECT_EQ(expected, result);
  }
};

class XlaAotCompileTest
    : public XlaCompileTest,
      public ::testing::WithParamInterface<absl::string_view> {};

TEST_P(XlaAotCompileTest, LoadGpuExecutable) {
  Literal input1 = LiteralUtil::CreateR1<double>({0.0f, 1.0f, 2.0f});
  Literal input2 = LiteralUtil::CreateR1<double>({1.0f, 2.0f, 4.0f});
  Literal expected = LiteralUtil::CreateR1<double>({1.0f, 3.0f, 6.0f});
  LoadAndRunExecutable(GetParam(), {&input1, &input2}, expected);
}

INSTANTIATE_TEST_SUITE_P(
    TestingAotFormats, XlaAotCompileTest,
    ::testing::Values("xla_aot_compile_test_gpu_executable",
                      "xla_aot_compile_test_gpu_executable_hlo"));

TEST_F(XlaCompileTest, LoadGpuExecutableWithConstant) {
  Literal input = LiteralUtil::CreateR1<double>({3.0f, 3.0f, 3.0f});
  Literal expected = LiteralUtil::CreateR1<double>({4.0f, 5.0f, 6.0f});
  LoadAndRunExecutable("xla_aot_compile_test_gpu_executable_constant", {&input},
                       expected);
}

// Should also cover the case of loading a GPU executable with a GEMM.
TEST_F(XlaCompileTest, LoadGpuExecutableWithConvolution) {
  Literal input1 = LiteralUtil::CreateR4<float>(
      {{{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}},
        {{11.0, 12.0}, {13.0, 14.0}, {15.0, 16.0}, {17.0, 18.0}},
        {{21.0, 22.0}, {23.0, 24.0}, {25.0, 26.0}, {27.0, 28.0}},
        {{31.0, 32.0}, {33.0, 34.0}, {35.0, 36.0}, {37.0, 38.0}}}});
  Literal input2 =
      LiteralUtil::CreateR4<float>({{{{1.0}, {2.0}}, {{3.0}, {4.0}}},
                                    {{{5.0}, {6.0}}, {{7.0}, {8.0}}},
                                    {{{9.0}, {10.0}}, {{11.0}, {12.0}}}});
  Literal expected = LiteralUtil::CreateR4<float>({{
      {{1310.0}, {1466.0}, {1622.0}},
      {{2090.0}, {2246.0}, {2402.0}},
  }});
  LoadAndRunExecutable("xla_aot_compile_test_gpu_executable_convolution",
                       {&input1, &input2}, expected);
}

}  // namespace
}  // namespace xla_compile
}  // namespace xla
