/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/xla/experimental/conv_emitter/conv_emitter.h"

#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"  // from @llvm-project
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"  // from @llvm-project
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"  // from @llvm-project
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace experimental {
namespace {

std::string CompileHloConvAndGetMlir(absl::string_view hlo_text) {
  xla::HloModuleConfig hlo_config;
  VerifiedHloModule hlo_module(
      "Conv", hlo_config, /*verifier_layout_sensitive=*/false,
      /*allow_mixed_precision_in_hlo_verifier=*/true,
      /*shape_size_function=*/ShapeUtil::ByteSizeOfElements);
  TF_CHECK_OK(hlo_module.ParseHloStringAndVerifyModule(hlo_text));
  xla::HloInstruction* conv =
      hlo_module.entry_computation()->root_instruction();

  mlir::MLIRContext context;
  context.loadDialect<mlir::AffineDialect, mlir::arith::ArithmeticDialect,
                      mlir::memref::MemRefDialect, mlir::func::FuncDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));

  mlir::func::FuncOp function =
      EmitConvolutionForwardAsMlir(conv, "Conv", &context).ValueOrDie();

  mlir_module->push_back(function);
  (void)mlir_module->verifyInvariants();

  std::string mlir_text;
  {
    llvm::raw_string_ostream strstream(mlir_text);
    function.print(strstream);
  }
  VLOG(1) << mlir_text;

  {
    mlir::PassManager pm(mlir_module->getContext());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createMemRefToLLVMPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    CHECK(mlir::succeeded(pm.run(*mlir_module)));
  }

  return mlir_text;
}

// TODO(timshen): integrate this with mlir's testing infrastructure.
TEST(ConvEmitterTest, TestDefault) {
  std::string hlo_text = R"(HloModule TestModule
ENTRY %TestComputation {
  %param_0 = f16[128,4,224,224]{1,3,2,0} parameter(0)
  %param_1 = f16[7,7,64,4]{3,1,0,2} parameter(1)
  ROOT %custom-call.1 = (f16[128,64,112,112]{1,3,2,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=bf01_01oi->bf01, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})";

  std::string expected_mlir_pattern =
      R"(
CHECK: func @Conv(%arg0: memref<128x112x112x64xf16>, %arg1: memref<128x224x224x4xf16>, %arg2: memref<64x7x7x4xf16>) {
CHECK-NEXT:   affine.for %arg3 = 0 to 128 {
CHECK-NEXT:     affine.for %arg4 = 0 to 2 {
CHECK-NEXT:       affine.for %arg5 = 0 to 112 {
CHECK-NEXT:         affine.for %arg6 = 0 to 7 {
CHECK-NEXT:           %0 = memref.alloc() : memref<32x16xf32>
CHECK-NEXT:           affine.for %arg7 = 0 to 32 {
CHECK-NEXT:             affine.for %arg8 = 0 to 16 {
CHECK-NEXT:               %cst = arith.constant 0.000000e+00 : f32
CHECK-NEXT:               affine.store %cst, %0[%arg7, %arg8] : memref<32x16xf32>
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:           affine.for %arg7 = 0 to 1 {
CHECK-NEXT:             affine.for %arg8 = 0 to 7 {
CHECK-NEXT:               affine.for %arg9 = 0 to 7 {
CHECK-NEXT:                 affine.for %arg10 = 0 to 32 {
CHECK-NEXT:                   affine.for %arg11 = 0 to 16 {
CHECK-NEXT:                     affine.for %arg12 = 0 to 4 {
CHECK-NEXT:                       %1 = affine.load %arg1[%arg3, %arg5 * 2 + %arg8 - 3, (%arg6 * 16 + %arg11) * 2 + %arg9 - 3, %arg7 * 4 + %arg12] : memref<128x224x224x4xf16>
CHECK-NEXT:                       %2 = arith.extf %1 : f16 to f32
CHECK-NEXT:                       %3 = affine.load %arg2[%arg4 * 32 + %arg10, %arg8, %arg9, %arg7 * 4 + %arg12] : memref<64x7x7x4xf16>
CHECK-NEXT:                       %4 = arith.extf %3 : f16 to f32
CHECK-NEXT:                       %5 = affine.load %0[%arg10, %arg11] : memref<32x16xf32>
CHECK-NEXT:                       %6 = arith.mulf %2, %4 : f32
CHECK-NEXT:                       %7 = arith.addf %5, %6 : f32
CHECK-NEXT:                       affine.store %7, %0[%arg10, %arg11] : memref<32x16xf32>
CHECK-NEXT:                     }
CHECK-NEXT:                   }
CHECK-NEXT:                 }
CHECK-NEXT:               }
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:           affine.for %arg7 = 0 to 32 {
CHECK-NEXT:             affine.for %arg8 = 0 to 16 {
CHECK-NEXT:               %1 = affine.load %0[%arg7, %arg8] : memref<32x16xf32>
CHECK-NEXT:               %2 = arith.truncf %1 : f32 to f16
CHECK-NEXT:               affine.store %2, %arg0[%arg3, %arg5, %arg6 * 16 + %arg8, %arg4 * 32 + %arg7] : memref<128x112x112x64xf16>
CHECK-NEXT:             }
CHECK-NEXT:           }
CHECK-NEXT:         }
CHECK-NEXT:       }
CHECK-NEXT:     }
CHECK-NEXT:   }
CHECK-NEXT:   return
CHECK-NEXT: }
)";

  EXPECT_TRUE(
      RunFileCheck(CompileHloConvAndGetMlir(hlo_text), expected_mlir_pattern)
          .ValueOrDie());
}

}  // namespace
}  // namespace experimental
}  // namespace xla
