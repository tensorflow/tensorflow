/* Copyright 2023 The OpenXLA Authors.

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

#include <optional>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuInt4Test : public GpuCodegenTest {};

TEST_F(GpuInt4Test, TestInt4ParameterSize) {
  const std::string hlo_text = R"(
  HloModule Reshape
  ENTRY main {
    x = s4[4] parameter(0)
    ROOT y = s8[4] convert(x)
  })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfigForTest()).value();

  // The input should be 2 bytes and the output should be 4 bytes
  auto expected_ir = R"(
; CHECK: define KERNEL_ANNOTATION {{.*}} dereferenceable(2){{.*}} dereferenceable(4)
)";
  CompileAndVerifyIr(std::move(hlo_module),
                     MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true);
  EXPECT_TRUE(RunAndCompare(hlo_text, /*error=*/std::nullopt));
}

TEST_F(GpuInt4Test, TestInt4OutputSize) {
  const std::string hlo_text = R"(
  HloModule Reshape
  ENTRY main {
    x = s8[4] parameter(0)
    ROOT y = s4[4] convert(x)
  })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfigForTest()).value();

  // The input should be 4 bytes and the output should be 2 bytes
  auto expected_ir = R"(
; CHECK: define KERNEL_ANNOTATION {{.*}} dereferenceable(4){{.*}} dereferenceable(2)
)";
  CompileAndVerifyIr(std::move(hlo_module),
                     MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true);
  EXPECT_TRUE(RunAndCompare(hlo_text, /*error=*/std::nullopt));
}

TEST_F(GpuInt4Test, TestConstantSize) {
  const std::string hlo_text = R"(
  HloModule Reshape
  ENTRY main {
    x = s4[4] constant({1, 2, 3, 4})
    ROOT y = s8[4] convert(x)
  })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfigForTest()).value();

  // The constant should be 2 bytes and the output should be 4 bytes
  auto expected_ir = R"(
; CHECK: define KERNEL_ANNOTATION {{.*}} dereferenceable(2){{.*}} dereferenceable(4)
)";
  CompileAndVerifyIr(std::move(hlo_module),
                     MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/true);
  EXPECT_TRUE(RunAndCompare(hlo_text, /*error=*/std::nullopt));
}

TEST_F(GpuInt4Test, TestOddElements) {
  const std::string hlo_text = R"(
  HloModule TestOddElements
  ENTRY main {
    x = s8[5] constant({1, 2, 3, 4, 5})
    ROOT y = s4[5] convert(x)
  })";
  auto hlo_module =
      ParseAndReturnVerifiedModule(hlo_text, GetModuleConfigForTest()).value();

  // A conditional branch should check if the index is in bounds within the
  // unrolled loop
  absl::string_view expected_ir = R"(
      ; CHECK: %[[in_bounds:.*]] = icmp sle i32 %{{.*}}, 1
      ; CHECK-NEXT: br i1 %[[in_bounds]], label %[[in_bounds_true:.*]], label %[[in_bounds_after:.*]]
      ; CHECK: [[in_bounds_true]]:
      ; CHECK: %{{.*}} = load i8, ptr %{{.*}}, align 1
      ; CHECK: cmpxchg ptr %{{.*}}
      ; CHECK: br label %[[in_bounds_after]]
      ; CHECK: [[in_bounds_after]]:
      ; CHECK-NEXT: ret void)";
  CompileAndVerifyIr(std::move(hlo_module),
                     MakePlatformSpecificLlvm(expected_ir),
                     /*match_optimized_ir=*/false);
  EXPECT_TRUE(RunAndCompare(hlo_text, /*error=*/std::nullopt));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
