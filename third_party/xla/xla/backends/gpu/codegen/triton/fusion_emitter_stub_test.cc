/* Copyright 2024 The OpenXLA Authors.

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

#include <gtest/gtest.h>
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/triton/compilation_pipeline.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter_legacy_matmul.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/hlo_module_config.h"

namespace mlir::triton::nvidia_gpu {
// We define ClusterInfo here in order to avoid having to import a GPU-only
// header.
struct ClusterInfo {};

}  // namespace mlir::triton::nvidia_gpu

namespace xla::gpu {
namespace {

TEST(TritonStub, CallStubApi) {
  mlir::MLIRContext context;

  LoadMlirDialectsForTriton(context);
  EXPECT_FALSE(TritonWrapper({}, nullptr, {}, {}, {}, nullptr, context).ok());
  EXPECT_FALSE(CreateTritonModule({}, nullptr, {}, {}, context).ok());
  EXPECT_FALSE(CompileTritonToLLVM("", HloModule("test", HloModuleConfig()), {},
                                   {}, {}, nullptr, context,
                                   /*is_xla_fusion=*/true, {})
                   .ok());

  mlir::OpPassManager pm;
  ::mlir::triton::nvidia_gpu::ClusterInfo cluster_info;

  EXPECT_FALSE(CreateTritonPipeline(&pm, "", 1, 1, 1, cluster_info,
                                    /*is_xla_fusion=*/true)
                   .ok());
  EXPECT_EQ(GetLibdevicePath({}, {}), "");

  EmitterLocOpBuilder builder(mlir::UnknownLoc::get(&context), &context);

  EXPECT_TRUE(
      ir_emitter_triton_internal::ComputeDelinearizedTileIndex(builder, {})
          .empty());

  HloConstantInstruction constant(LiteralUtil::CreateR1<int>({1, 1}));
  auto tiled_hlo = TiledHloInstruction::Create(&constant, {}, {1}, {1}, {});
  EXPECT_TRUE(tiled_hlo.ok());

  EXPECT_FALSE(ir_emitter_triton_internal::CreateMakeTensorPtrOp(
                   builder, {}, *tiled_hlo.value(), {})
                   .ok());
}

TEST(TritonStub, CallLegacyMatMulApis) {
  HloConstantInstruction constant(Literal{});
  auto adaptor = HloFusionAdaptor::ForInstruction(&constant);
  EXPECT_FALSE(GetMatMulLaunchDimensions({}, *adaptor.get(), {}, {}).ok());

  mlir::MLIRContext context;
  EmitterLocOpBuilder builder(mlir::UnknownLoc::get(&context), &context);
  EXPECT_FALSE(EmitMatMul(builder, {}, {}, nullptr, {}, {}).ok());
}

}  // namespace

}  // namespace xla::gpu
