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

#include "xla/service/gpu/transforms/gpusolver_rewriter.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/cuda/cuda_solver_context.h"

namespace xla {
namespace gpu {
namespace {

namespace m = ::xla::match;

class GpusolverRewriterTest : public HloHardwareIndependentTestBase {
 public:
  GpusolverRewriter gpusolver_rewriter_{
      stream_executor::CudaSolverContext::Create};
};

TEST_F(GpusolverRewriterTest, CholeskyTest) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule CholeskyTest

  ENTRY entry_computation {
    input = f32[1,256,256] parameter(0)
    ROOT decomp = f32[1,256,256] cholesky(input)
  }
)")
                    .value();

  EXPECT_TRUE(gpusolver_rewriter_.Run(module.get()).value());

  const HloInstruction* entry_root =
      module->entry_computation()->root_instruction();
  ASSERT_THAT(
      entry_root,
      GmockMatch(m::Select(
          m::Broadcast(
              m::Compare(m::GetTupleElement(), m::Broadcast(m::Constant()))),
          m::GetTupleElement(m::CustomCall()), m::Broadcast(m::Constant()))));
}
}  // namespace
}  // namespace gpu
}  // namespace xla
