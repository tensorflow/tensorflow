/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/hlo_cost_analysis_wrapper.h"

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/tsl/platform/test.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(TpuHloCostAnalysisTest, GetInputBitwidths) {
  absl::string_view hlo_text = R"hlo(
HloModule test_module
ENTRY test {
   x = bf16[2,4]{1,0} parameter(0)
   y = s4[2,4]{1,0} parameter(1)
   ROOT add = f32[2,4]{1,0} convolution(x,y), dim_labels=012_012->012
   }
)hlo";
  auto hlo_module = xla::ParseAndReturnUnverifiedModule(hlo_text).value();
  const xla::HloInstruction* root =
      hlo_module->entry_computation()->root_instruction();
  EXPECT_THAT(GetInputBitwidths(*root), ::testing::ElementsAre(16, 4));
}
}  // namespace
}  // namespace profiler
}  // namespace tensorflow
