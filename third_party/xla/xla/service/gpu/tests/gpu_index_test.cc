/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

// This file tests the index expressions used to reference source tensors. When
// the destination tensor and source tensor have compatible shapes, the linear
// index is used to access the source tensor. Otherwise, dimensional indices
// computed from the linear index are used to access the source tensor.

class GpuIndexTest : public GpuCodegenTest {};

TEST_F(GpuIndexTest, CompatibleUseLinearIndex) {
  HloComputation::Builder builder(TestName());

  auto param_shape = ShapeUtil::MakeShape(F32, {5, 7, 2});
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "x"));
  HloInstruction* param_y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "y"));
  builder.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {5, 7, 2}), param_x, param_y,
      ComparisonDirection::kGe));

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  // Check the optimized IR as the unoptimized IR contains dead udiv and urem.
  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithReshape) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY CompatibleUseLinearIndexWithReshape {
      x = f32[5,7,2]{2,1,0} parameter(0)
      y = f32[5,14]{1,0} parameter(1)
      reshape = f32[5,7,2]{2,1,0} reshape(y)
      ROOT gte = pred[5,7,2]{2,1,0} compare(x, reshape), direction=GE
    })")
                    .value();

  // Check the optimized IR as the unoptimized IR contains dead udiv and urem.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithReshapeAndBroadcast) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY CompatibleUseLinearIndexWithReshape {
      x = f32[5,7,2]{2,1,0} parameter(0)
      y = f32[14]{0} parameter(1)
      reshape = f32[7,2]{1,0} reshape(y)
      broadcast = f32[5,7,2]{2,1,0} broadcast(reshape), dimensions={1,2}
      ROOT gte = pred[5,7,2]{2,1,0} compare(x, broadcast), direction=GE
    })")
                    .value();

  // Check the optimized IR reuses the linear index by calculating modulo 14.

  // In the IR generated for AMDGPUs, we do not seem to have the
  // the addrspace(1) attribute for the lines being checked by the following
  // patterns.
  // need to investigate why that is the case, and whether or not it is ok
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK: %[[urem1:.*]] = urem i{{[0-9]*}} %[[linear_index:.*]], 14
; CHECK: %[[idx1:.*]] = zext nneg i{{[0-9]*}} %[[urem1]] to i64
; CHECK: getelementptr inbounds{{( nuw)?}} float, ptr{{( addrspace\(1\))?}} %[[alloc:.*]], i64 %[[idx1]]
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithSizeOneDimensions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule  test_module

    ENTRY CompatibleUseLinearIndexWithSizeOneDimensions  {
      x = f32[1,1024,1,256]{3,2,1,0} parameter(0)
      ROOT y = f16[1,1024,1,256]{3,2,1,0} convert(x)
    })")
                    .value();

  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithBitcastTranspose) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule  test_module

    ENTRY CompatibleUseLinearIndexWithTranspose  {
      x = f32[1024,2,256,3]{2,3,0,1} parameter(0)
      y = f32[2,1024,3,256]{3,2,1,0} parameter(1)
      transpose = f32[2,1024,3,256]{3,2,1,0} transpose(x), dimensions={1,0,3,2}
      ROOT gte = pred[2,1024,3,256]{3,2,1,0} compare(transpose, y), direction=GE
    })")
                    .value();
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

}  // namespace gpu
}  // namespace xla
