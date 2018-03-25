/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gather_expander.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tools/parser/hlo_parser.h"

namespace xla {
namespace {
TEST(GatherExpanderTest, ErrorStatusOnTooManyIndices) {
  const string hlo_text = R"(
HloModule TensorFlowGatherMultipleBatchDims

ENTRY main {
  operand = s32[3,3] parameter(0)
  indices = s32[2147483647,5] parameter(1)
  ROOT gather = s32[2147483647,3,5] gather(operand, indices),
      output_window_dims={1},
      elided_window_dims={1},
      gather_dims_to_operand_dims={1},
      index_vector_dim=2,
      window_bounds={3, 1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          tools::Parse(hlo_text));

  Status status = GatherExpander{}.Run(module.get()).status();
  EXPECT_EQ(status.code(), tensorflow::error::UNIMPLEMENTED);

  ASSERT_THAT(
      status.error_message(),
      ::testing::HasSubstr("Gather operations with more than 2147483647 gather "
                           "indices are not supported."));
}

}  // namespace
}  // namespace xla
