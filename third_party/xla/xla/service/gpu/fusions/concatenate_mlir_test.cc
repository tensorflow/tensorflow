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

#include "xla/service/gpu/fusions/concatenate_mlir.h"

#include <gtest/gtest.h>
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/GPU/IR/GPUDialect.h"  // from @llvm-project
#include "mlir/Dialect/Math/IR/Math.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class MlirConcatenateFusionTest : public HloTestBase {
 public:
  MlirConcatenateFusionTest() {
    context_.loadDialect<mlir::tensor::TensorDialect, mlir::func::FuncDialect,
                         mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                         mlir::math::MathDialect, mlir::scf::SCFDialect,
                         mlir::mhlo::MhloDialect, mlir::gpu::GPUDialect>();
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    context_.appendDialectRegistry(registry);
  }

  stream_executor::DeviceDescription device_info_ =
      TestGpuDeviceInfo::RTXA6000DeviceInfo();
  mlir::MLIRContext context_;
};

TEST_F(MlirConcatenateFusionTest, StandAloneConcatenate) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT concat = f32[256] concatenate(param0, param1), dimensions={0}
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT fusion = f32[256] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  // TODO: Add support for parameter operands.
  EXPECT_FALSE(MlirConcatenateFusion::IsSupported(analysis));
}

TEST_F(MlirConcatenateFusionTest, ConcatenateElementwise) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    log = f32[128] log(param0)
    exp = f32[128] exponential(param1)
    ROOT concat = f32[256] concatenate(log, exp), dimensions={0}
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT fusion = f32[256] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  MlirConcatenateFusion fusion(analysis);
  TF_ASSERT_OK_AND_ASSIGN(
      auto mlir_module,
      fusion.CreateMLIRModule(context_, *Cast<HloFusionInstruction>(root),
                              "fused_computation", nullptr));

  std::string out;
  llvm::raw_string_ostream os(out);
  mlir_module->print(os);
  ASSERT_TRUE(RunFileCheck(out, R"(
// CHECK-LABEL: fused_computation
// CHECK:       %[[C_128:.*]] = arith.constant 128

// CHECK:       %[[THREAD_ID:.*]] = gpu.thread_id x
// CHECK:       %[[VAL_1:.*]] = call @fused_computation_log({{.*}}, %[[THREAD_ID]])
// CHECK:       %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1:.*]] into {{.*}}[%[[THREAD_ID]]]

// CHECK:       %[[VAL_2:.*]] = call @fused_computation_exp({{.*}}, %[[THREAD_ID]])
// CHECK:       %[[INDEX_2:.*]] = arith.addi %[[THREAD_ID]], %[[C_128]]
// CHECK:       %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2:.*]] into {{.*}}[%[[INDEX_2]]]

// CHECK:       return %[[INSERTED_2]]

// CHECK: func.func @fused_computation_log
// CHECK: func.func @fused_computation_exp
)")
                  .value());
}

TEST_F(MlirConcatenateFusionTest, DifferentDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

  fused_computation {
    param0 = f32[400] parameter(0)
    param1 = f32[200] parameter(1)
    log = f32[400] log(param0)
    exp = f32[200] exponential(param1)
    ROOT concat = f32[600] concatenate(log, exp), dimensions={0}
  }

  ENTRY main {
    param0 = f32[400] parameter(0)
    param1 = f32[200] parameter(1)
    ROOT fusion = f32[600] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  // TODO: Add support for different operand sizes.
  EXPECT_FALSE(MlirConcatenateFusion::IsSupported(analysis));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
