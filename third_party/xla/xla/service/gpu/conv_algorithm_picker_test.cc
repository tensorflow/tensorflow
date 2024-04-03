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

#include "xla/service/gpu/conv_algorithm_picker.h"

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/gpu_conv_rewriter.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/service/platform_util.h"
#include "xla/service/tuple_simplifier.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuConvAlgorithmPickerTest : public HloTestBase {
 public:
  GpuConvAlgorithmPickerTest() { AutotunerUtil::ClearAutotuneResults(); }
};

TEST_F(GpuConvAlgorithmPickerTest, SetAlgorithm) {
  constexpr absl::string_view kHlo = R"(
HloModule module

ENTRY main {
  %arg0 = f32[3,56,56,16]{2,1,0,3} parameter(0)
  %arg1 = f32[3,3,3,64]{2,1,0,3} parameter(1)
  ROOT %conv = f32[54,54,16,64]{1,0,3,2} convolution(%arg0, %arg1), window={size=3x3}, dim_labels=f01b_i01o->01bf
})";
  TF_ASSERT_OK_AND_ASSIGN(auto m, ParseAndReturnVerifiedModule(kHlo));

  se::Platform* platform = PlatformUtil::GetDefaultPlatform().value();
  TF_ASSERT_OK_AND_ASSIGN(std::vector<se::StreamExecutor*> executors,
                          PlatformUtil::GetStreamExecutors(platform));
  ASSERT_GT(executors.size(), 0);
  se::StreamExecutor* stream_exec = executors[0];

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(), m.get()));
  changed = false;
  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();

  AutotuneConfig cfg{DeviceConfig{stream_exec, nullptr}, opts};
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(AutotunerUtil::SerializeAutotuneResults(&results));
  ASSERT_EQ(results.results_size(), 1);
  auto& result = *results.mutable_results(0)->mutable_result();
  int64_t old_scratch_bytes = result.scratch_bytes();
  int64_t new_scratch_bytes = old_scratch_bytes + 1;
  result.set_scratch_bytes(new_scratch_bytes);

  AutotunerUtil::ClearAutotuneResults();
  TF_ASSERT_OK(AutotunerUtil::LoadAutotuneResults(results));

  // Now send the same module through GpuConvAlgorithmPicker again.  The conv
  // should have the new scratch bytes.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(), m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed,
                          RunHloPass(GpuConvAlgorithmPicker(cfg), m.get()));
  ASSERT_TRUE(changed);

  // TupleSimplifier cleans this up a bit before we pattern-match
  TF_ASSERT_OK(RunHloPass(TupleSimplifier(), m.get()).status());

  SCOPED_TRACE(m->ToString());
  HloInstruction* conv;
  ASSERT_THAT(m->entry_computation()->root_instruction(),
              GmockMatch(m::GetTupleElement(m::CustomCall(&conv))));
  EXPECT_THAT(
      conv->shape(),
      GmockMatch(m::Shape().WithSubshape(
          {1}, m::Shape().WithElementType(U8).WithDims({new_scratch_bytes}))));
}

}  // namespace
}  // namespace xla::gpu
