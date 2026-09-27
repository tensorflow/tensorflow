/* Copyright 2021 The OpenXLA Authors.

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
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla::gpu {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::ContainsRegex;

class TooManyBlocksTest : public GpuPjRtCodegenTest {};

TEST_F(TooManyBlocksTest, FailsWithInvalidStatus) {
  // This test ensures that invalid (too large) launch grids are caught
  // somewhere in the pipeline. The practical relevance is low, since as of
  // 2024, the inputs or outputs have to be way too large to fit on any GPU
  // anyway.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                       ParseAndReturnVerifiedModule(R"(
    HloModule primitive_computation_mul.8

    ENTRY primitive_computation_mul.8 {
      p.1 = s8[65536] parameter(0)
      p.2 = s8[65536] parameter(1)
      bcast.3 = s8[65536,65536,65536,128,16] broadcast(p.1), dimensions={0}
      bcast.4 = s8[65536,65536,65536,128,16] broadcast(p.2), dimensions={1}
      ROOT multiply.5 = s8[65536,65536,65536,128,16] multiply(bcast.3, bcast.4)
    }
  )"));
  EXPECT_THAT(CompileToExecutable(std::move(hlo_module),
                                  /*run_optimization_passes=*/true),
              StatusIs(_, ContainsRegex(
                              "Kernel '.*fusion.*' launch needs more blocks")));
}

}  // namespace
}  // namespace xla::gpu
