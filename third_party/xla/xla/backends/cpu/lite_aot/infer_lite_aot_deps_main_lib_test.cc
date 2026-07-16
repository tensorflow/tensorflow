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

#include "xla/backends/cpu/lite_aot/infer_lite_aot_deps_main_lib.h"

#include <functional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace cpu {
namespace {

struct ThunkTestCase {
  std::string name;
  std::function<void(ThunkProto&)> setup_thunk;
  std::string expected_dep;
};

class InferLiteAotDepsParamTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<ThunkTestCase> {};

TEST_P(InferLiteAotDepsParamTest, InferredDependencyMatches) {
  const ThunkTestCase& test_case = GetParam();
  CompilationResultProto compilation_result_proto;
  auto* thunk = compilation_result_proto.mutable_thunk_sequence()->add_thunks();
  test_case.setup_thunk(*thunk);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> deps,
                          InferLiteAotDeps(compilation_result_proto));

  EXPECT_THAT(deps, ::testing::Contains(test_case.expected_dep));
}

INSTANTIATE_TEST_SUITE_P(
    InferLiteAotDepsTests, InferLiteAotDepsParamTest,
    ::testing::ValuesIn<ThunkTestCase>({
        // LINT.IfChange
        {"Collective", [](ThunkProto& t) { t.mutable_collective_thunk(); },
         SerDesPath("collective")},
        {"Convolution", [](ThunkProto& t) { t.mutable_convolution_thunk(); },
         SerDesPath("convolution")},
        {"Copy", [](ThunkProto& t) { t.mutable_copy_thunk(); },
         SerDesPath("copy")},
        {"CustomCall", [](ThunkProto& t) { t.mutable_custom_call_thunk(); },
         SerDesPath("custom_call")},
        {"Dot", [](ThunkProto& t) { t.mutable_dot_thunk(); },
         SerDesPath("dot")},
        {"Fft", [](ThunkProto& t) { t.mutable_fft_thunk(); },
         SerDesPath("fft")},
        {"YnnFusion", [](ThunkProto& t) { t.mutable_ynn_fusion_thunk(); },
         SerDesPath("ynn_fusion")},
        // LINT.ThenChange(//third_party/tensorflow/compiler/xla/backends/cpu/lite_aot/infer_lite_aot_deps_main_lib.cc)
    }),
    [](const ::testing::TestParamInfo<InferLiteAotDepsParamTest::ParamType>&
           info) { return info.param.name; });

TEST(InferLiteAotDepsMainLibTest, NoMatchingThunkReturnsEmpty) {
  CompilationResultProto compilation_result_proto;
  auto* thunk = compilation_result_proto.mutable_thunk_sequence()->add_thunks();
  thunk->mutable_kernel_thunk();

  TF_ASSERT_OK_AND_ASSIGN(std::vector<std::string> deps,
                          InferLiteAotDeps(compilation_result_proto));

  EXPECT_TRUE(deps.empty());
}

}  // namespace
}  // namespace cpu
}  // namespace xla
