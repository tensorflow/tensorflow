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
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "xla/backends/gpu/codegen/triton/fusion_emitter.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"

namespace xla::gpu {
namespace {

TEST(TritonStub, CallStubApi) {
  mlir::MLIRContext mlir_context;

  LoadMlirDialectsForTriton(mlir_context);
  EXPECT_FALSE(
      TritonWrapper({}, nullptr, {}, {}, {}, nullptr, mlir_context).ok());
  EXPECT_FALSE(CreateTritonModule({}, nullptr, {}, {}, mlir_context).ok());
  EXPECT_FALSE(CompileTritonToLLVM("", HloModule("test", HloModuleConfig()), {},
                                   {}, {}, nullptr, mlir_context,
                                   /*is_xla_fusion=*/true, {})
                   .ok());

  mlir::OpPassManager pm;

  EXPECT_EQ(GetLibdevicePath({}, {}), "");

  HloConstantInstruction constant(LiteralUtil::CreateR1<int>({1, 1}));
  auto tiled_hlo = TiledHloInstruction::Create(
      &constant, /*operands=*/{}, /*runtime_variables=*/{}, /*tile_sizes=*/{1},
      /*tile_strides=*/{1}, /*tile_offsets_indexing=*/{});
  EXPECT_TRUE(tiled_hlo.ok());
}

}  // namespace

}  // namespace xla::gpu
