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

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class TooManyBlocksTest : public GpuCodegenTest {};

TEST_F(TooManyBlocksTest, FailsWithInvalidStatus) {
  // This test ensures that invalid (too large) launch grids are caught
  // somewhere in the pipeline. The practical relevance is low, since as of
  // 2024, the inputs or outputs have to be way too large to fit on any GPU
  // anyway.
  const char* hlo_text = R"(
HloModule primitive_computation_mul.8

ENTRY primitive_computation_mul.8 {
  parameter.1 = f32[4,1048576,1,1]{3,2,1,0} parameter(0)
  reshape.3 = f32[4,1048576,1]{2,1,0} reshape(parameter.1)
  broadcast.4 = f32[4,1048576,1048576,1]{3,2,1,0} broadcast(reshape.3), dimensions={0,1,3}
  parameter.2 = f32[4,1,1048576,1]{3,2,1,0} parameter(1)
  reshape.5 = f32[4,1048576,1]{2,1,0} reshape(parameter.2)
  broadcast.6 = f32[4,1048576,1048576,1]{3,2,1,0} broadcast(reshape.5), dimensions={0,2,3}
  ROOT multiply.7 = f32[4,1048576,1048576,1]{3,2,1,0} multiply(broadcast.4, broadcast.6)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_module,
                          GetOptimizedModule(hlo_text));

  absl::StatusOr<std::unique_ptr<Executable>> failed_executable =
      backend().compiler()->RunBackend(
          std::move(optimized_module), backend().default_stream_executor(),
          backend().default_stream_executor()->GetAllocator());

  EXPECT_FALSE(failed_executable.ok());
  EXPECT_THAT(
      failed_executable.status().ToString(),
      ::testing::ContainsRegex("Kernel '.*fusion.*' launch needs more blocks"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
