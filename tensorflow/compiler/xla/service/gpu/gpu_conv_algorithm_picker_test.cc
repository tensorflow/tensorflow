/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_algorithm_picker.h"

#include "tensorflow/compiler/xla/service/gpu/gpu_conv_rewriter.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/service/tuple_simplifier.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla::gpu {
namespace {

namespace m = ::xla::match;

class GpuConvAlgorithmPickerTest : public HloTestBase {
 public:
  GpuConvAlgorithmPickerTest() {
    GpuConvAlgorithmPicker::ClearAutotuneResults();
  }
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
  DeviceConfig device_config{stream_exec,
                             /*allocator=*/nullptr};

  bool changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(), m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GpuConvAlgorithmPicker(device_config), m.get()));
  ASSERT_TRUE(changed);

  AutotuneResults results;
  TF_ASSERT_OK(GpuConvAlgorithmPicker::WriteAutotuneResults(&results));
  ASSERT_EQ(results.convs_size(), 1);
  auto& result = *results.mutable_convs(0)->mutable_result();
  int64_t old_scratch_bytes = result.scratch_bytes();
  int64_t new_scratch_bytes = old_scratch_bytes + 1;
  result.set_scratch_bytes(new_scratch_bytes);

  GpuConvAlgorithmPicker::ClearAutotuneResults();
  TF_ASSERT_OK(GpuConvAlgorithmPicker::LoadAutotuneResults(results));

  // Now send the same module through GpuConvAlgorithmPicker again.  The conv
  // should have the new scratch bytes.
  TF_ASSERT_OK_AND_ASSIGN(m, ParseAndReturnVerifiedModule(kHlo));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(changed, RunHloPass(GpuConvRewriter(), m.get()));
  changed = false;
  TF_ASSERT_OK_AND_ASSIGN(
      changed, RunHloPass(GpuConvAlgorithmPicker(device_config), m.get()));
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
