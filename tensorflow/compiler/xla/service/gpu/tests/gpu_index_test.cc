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

#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

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
  HloModuleConfig config;
  config.set_debug_options(HloTestBase::GetDebugOptionsForTest());
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY CompatibleUseLinearIndexWithReshape {
      x = f32[5,7,2]{2,1,0} parameter(0)
      y = f32[5,14]{1,0} parameter(1)
      reshape = f32[5,7,2]{2,1,0} reshape(y)
      ROOT gte = pred[5,7,2]{2,1,0} compare(x, reshape), direction=GE
    })",
                                             config)
                    .ValueOrDie();

  // Check the optimized IR as the unoptimized IR contains dead udiv and urem.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

TEST_F(GpuIndexTest,
       ReuseMultidimIndexWithTrivialReshapeAndNonContiguousBroadcast) {
  HloModuleConfig config;
  config.set_debug_options(HloTestBase::GetDebugOptionsForTest());
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY CompatibleUseLinearIndexWithReshape {
      x = f32[1,7,2,5,3]{4,3,2,1,0} parameter(0)
      y = f32[2,1,3]{2,1,0} parameter(1)
      reshape = f32[1,2,3]{2,1,0} reshape(y)
      broadcast = f32[1,7,2,5,3]{4,3,2,1,0} broadcast(reshape), dimensions={0,2,4}
      ROOT gte = pred[1,7,2,5,3]{4,3,2,1,0} compare(x, broadcast), direction=GE
    })",
                                             config)
                    .ValueOrDie();
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK: %[[tmp4:.*]] = udiv i32 %[[linear_index:.*]], 1
; CHECK: %[[dim4:.*]] = urem i32 %[[tmp4]], 3
; CHECK: %[[tmp3:.*]] = udiv i32 %[[linear_index]], 3
; CHECK: %[[dim3:.*]] = urem i32 %[[tmp3]], 5
; CHECK: %[[tmp2:.*]] = udiv i32 %[[linear_index]], 15
; CHECK: %[[dim2:.*]] = urem i32 %[[tmp2]], 2
; CHECK: %[[tmp1:.*]] = udiv i32 %[[linear_index]], 30
; CHECK: %[[dim1:.*]] = urem i32 %[[tmp1]], 7
; CHECK: %[[dim0:.*]] = udiv i32 %[[linear_index]], 210
; CHECK: %{{.*}} = getelementptr inbounds [2 x [1 x [3 x float]]], [2 x [1 x [3 x float]]]* %{{.*}}, i32 0, i32 %[[dim2]], i32 0, i32 %[[dim4]]
      )",
                     /*match_optimized_ir=*/false);
}

#if TENSORFLOW_USE_ROCM
#else
TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithReshapeAndBroadcast) {
  HloModuleConfig config;
  config.set_debug_options(HloTestBase::GetDebugOptionsForTest());
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY CompatibleUseLinearIndexWithReshape {
      x = f32[5,7,2]{2,1,0} parameter(0)
      y = f32[14]{0} parameter(1)
      reshape = f32[7,2]{1,0} reshape(y)
      broadcast = f32[5,7,2]{2,1,0} broadcast(reshape), dimensions={1,2}
      ROOT gte = pred[5,7,2]{2,1,0} compare(x, broadcast), direction=GE
    })",
                                             config)
                    .ValueOrDie();

  // Check the optimized IR reuses the linear index by calculating modulo 14.

  // In the IR generated for AMDGPUs, we do not seem to have the
  // the addrspace(1) attribute for the lines being checked by the following
  // patterns.
  // need to investigate why that is the case, and whether or not it is ok
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK: %[[urem1:.*]] = urem i{{[0-9]*}} %[[linear_index:.*]], 14
; CHECK: %[[bitcast:.*]] = bitcast i8{{( addrspace\(1\))?}}* %[[alloc:.*]] to float{{( addrspace\(1\))?}}*
; CHECK: %[[idx1:.*]] = zext i{{[0-9]*}} %[[urem1]] to i64
; CHECK: getelementptr inbounds float, float{{( addrspace\(1\))?}}* %[[bitcast]], i64 %[[idx1]]
      )",
                     /*match_optimized_ir=*/true);
}
#endif

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithSizeOneDimensions) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  config.set_debug_options(debug_options);

  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule  test_module

    ENTRY CompatibleUseLinearIndexWithSizeOneDimensions  {
      x = f32[1,1024,1,256]{3,2,1,0} parameter(0)
      ROOT y = f16[1,1024,1,256]{2,3,1,0} convert(x)
    })",
                                             config)
                    .ValueOrDie();

  // Check that the unoptimized IR reuses the linear index.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-LABEL: @fusion
; CHECK: udiv i32 %[[linear_index:.*]], 262144
; CHECK: %[[ld_addr:.*]] = getelementptr inbounds float, float* {{.*}}, i32 %[[linear_index]]
; CHECK: load float, float* %[[ld_addr]]
; CHECK: %[[st_addr:.*]] = getelementptr inbounds half, half* {{.*}}, i32 %[[linear_index]]
; CHECK: store half {{.*}}, half* %[[st_addr]]
      )",
                     /*match_optimized_ir=*/false);
}

TEST_F(GpuIndexTest, CompatibleUseLinearIndexWithTranspose) {
  HloModuleConfig config;
  auto debug_options = HloTestBase::GetDebugOptionsForTest();
  debug_options.set_xla_gpu_max_kernel_unroll_factor(1);
  config.set_debug_options(debug_options);

  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule  test_module

    ENTRY CompatibleUseLinearIndexWithTranspose  {
      x = f32[2,1024,3,256]{3,2,1,0} parameter(0)
      y = f32[1024,2,256,3]{2,3,0,1} parameter(1)
      transpose = f32[1024,2,256,3]{3,2,1,0} transpose(x), dimensions={1,0,3,2}
      ROOT gte = pred[1024,2,256,3]{2,3,0,1} compare(transpose, y), direction=GE
    })",
                                             config)
                    .ValueOrDie();
  // Check the optimized IR contains no udiv and urem.
  CompileAndVerifyIr(std::move(module),
                     R"(
; CHECK-NOT: udiv
; CHECK-NOT: urem
      )",
                     /*match_optimized_ir=*/true);
}

}  // namespace gpu
}  // namespace xla
