/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

namespace {

class TooManyBlocksTest : public GpuCodegenTest {};

TEST_F(TooManyBlocksTest, FailsWithInvalidStatus) {
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

  StatusOr<std::unique_ptr<Executable>> failed_executable =
      backend().compiler()->RunBackend(
          std::move(optimized_module), backend().default_stream_executor(),
          backend().default_stream_executor()->GetAllocator());

  EXPECT_FALSE(failed_executable.ok());
  EXPECT_THAT(failed_executable.status().ToString(),
              ::testing::HasSubstr("Kernel launch needs more blocks"));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
