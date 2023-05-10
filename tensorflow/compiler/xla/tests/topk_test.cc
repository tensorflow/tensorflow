/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"

namespace xla {
namespace {

class TopkTest : public HloTestBase {};

XLA_TEST_F(TopkTest, SimpleTopK) {
  absl::string_view hlo = R"(
HloModule topk

compare {
  p.0.lhs = bf16[] parameter(0)
  p.0.rhs = bf16[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=GT
}

ENTRY TopK {
  x = bf16[10,10] parameter(0)
  ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, to_apply=compare
}
)";
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-5, 1e-5}));
}

XLA_TEST_F(TopkTest, SimpleTopKReverseDirection) {
  absl::string_view hlo = R"(
HloModule topk

compare {
  p.0.lhs = bf16[] parameter(0)
  p.0.rhs = bf16[] parameter(1)
  ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
}

ENTRY TopK {
  x = bf16[10,10] parameter(0)
  ROOT topk = (bf16[10,2], s32[10,2]) topk(x), k=2, to_apply=compare
}
)";
  EXPECT_TRUE(RunAndCompare(hlo, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace xla
