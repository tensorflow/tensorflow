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

#include "tensorflow/compiler/xla/service/mlir_gpu/experimental/conv_emitter/conv_emitter.h"

#include <vector>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/verified_hlo_module.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu_mlir {
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
  mlir::OwningModuleRef mlir_module(
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));

  mlir::FuncOp function =
      xla::mlir_gpu::EmitConvolutionForwardAsMlir(conv, "Conv", &context)
          .ValueOrDie();

  mlir_module->push_back(function);
  mlir_module->verify();

  std::string mlir_text;
  llvm::raw_string_ostream strstream(mlir_text);
  function.print(strstream);
  return mlir_text;
}

// TODO(timshen): integrate this with mlir_gpu's testing infrastructure.
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
CHECK-NEXT:   "affine.for"() ( {
CHECK-NEXT:   ^bb0(%arg3: index):
CHECK-NEXT:     "affine.for"() ( {
CHECK-NEXT:     ^bb0(%arg4: index):
CHECK-NEXT:       "affine.for"() ( {
CHECK-NEXT:       ^bb0(%arg5: index):
CHECK-NEXT:         "affine.for"() ( {
CHECK-NEXT:         ^bb0(%arg6: index):
CHECK-NEXT:           %0 = alloc() : memref<32x16xf32>
CHECK-NEXT:           "affine.for"() ( {
CHECK-NEXT:           ^bb0(%arg7: index):
CHECK-NEXT:             "affine.for"() ( {
CHECK-NEXT:             ^bb0(%arg8: index):
CHECK-NEXT:               %cst = constant 0.000000e+00 : f32
CHECK-NEXT:               "affine.store"(%cst, %0, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (f32, memref<32x16xf32>, index, index) -> ()
CHECK-NEXT:               "affine.terminator"() : () -> ()
CHECK-NEXT:             }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (16)} : () -> ()
CHECK-NEXT:             "affine.terminator"() : () -> ()
CHECK-NEXT:           }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()
CHECK-NEXT:           "affine.for"() ( {
CHECK-NEXT:           ^bb0(%arg7: index):
CHECK-NEXT:             "affine.for"() ( {
CHECK-NEXT:             ^bb0(%arg8: index):
CHECK-NEXT:               "affine.for"() ( {
CHECK-NEXT:               ^bb0(%arg9: index):
CHECK-NEXT:                 "affine.for"() ( {
CHECK-NEXT:                 ^bb0(%arg10: index):
CHECK-NEXT:                   "affine.for"() ( {
CHECK-NEXT:                   ^bb0(%arg11: index):
CHECK-NEXT:                     "affine.for"() ( {
CHECK-NEXT:                     ^bb0(%arg12: index):
CHECK-NEXT:                       %1 = "affine.load"(%arg1, %arg3, %arg9, %arg5, %arg6, %arg10, %arg11, %arg12, %arg8) {map = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2 * 2 + d4 - 3, (d3 * 16 + d7) * 2 + d5 - 3, d1 * 4 + d6)} : (memref<128x224x224x4xf16>, index, index, index, index, index, index, index, index) -> f16
CHECK-NEXT:                       %2 = fpext %1 : f16 to f32
CHECK-NEXT:                       %3 = "affine.load"(%arg2, %arg4, %arg9, %arg10, %arg11, %arg12, %arg7) {map = (d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d5, d2, d3, d1 * 4 + d4)} : (memref<64x7x7x4xf16>, index, index, index, index, index, index) -> f16
CHECK-NEXT:                       %4 = fpext %3 : f16 to f32
CHECK-NEXT:                       %5 = "affine.load"(%0, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (memref<32x16xf32>, index, index) -> f32
CHECK-NEXT:                       %6 = mulf %2, %4 : f32
CHECK-NEXT:                       %7 = addf %5, %6 : f32
CHECK-NEXT:                       "affine.store"(%7, %0, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (f32, memref<32x16xf32>, index, index) -> ()
CHECK-NEXT:                       "affine.terminator"() : () -> ()
CHECK-NEXT:                     }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (4)} : () -> ()
CHECK-NEXT:                     "affine.terminator"() : () -> ()
CHECK-NEXT:                   }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (7)} : () -> ()
CHECK-NEXT:                   "affine.terminator"() : () -> ()
CHECK-NEXT:                 }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (7)} : () -> ()
CHECK-NEXT:                 "affine.terminator"() : () -> ()
CHECK-NEXT:               }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (1)} : () -> ()
CHECK-NEXT:               "affine.terminator"() : () -> ()
CHECK-NEXT:             }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (16)} : () -> ()
CHECK-NEXT:             "affine.terminator"() : () -> ()
CHECK-NEXT:           }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()
CHECK-NEXT:           "affine.for"() ( {
CHECK-NEXT:           ^bb0(%arg7: index):
CHECK-NEXT:             "affine.for"() ( {
CHECK-NEXT:             ^bb0(%arg8: index):
CHECK-NEXT:               %1 = "affine.load"(%0, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (memref<32x16xf32>, index, index) -> f32
CHECK-NEXT:               %2 = fptrunc %1 : f32 to f16
CHECK-NEXT:               "affine.store"(%2, %arg0, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) {map = (d0, d1, d2, d3, d4, d5) -> (d0, d2, d3 * 16 + d5, d1 * 32 + d4)} : (f16, memref<128x112x112x64xf16>, index, index, index, index, index, index) -> ()
CHECK-NEXT:               "affine.terminator"() : () -> ()
CHECK-NEXT:             }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (16)} : () -> ()
CHECK-NEXT:             "affine.terminator"() : () -> ()
CHECK-NEXT:           }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (32)} : () -> ()
CHECK-NEXT:           "affine.terminator"() : () -> ()
CHECK-NEXT:         }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (7)} : () -> ()
CHECK-NEXT:         "affine.terminator"() : () -> ()
CHECK-NEXT:       }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (112)} : () -> ()
CHECK-NEXT:       "affine.terminator"() : () -> ()
CHECK-NEXT:     }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (2)} : () -> ()
CHECK-NEXT:     "affine.terminator"() : () -> ()
CHECK-NEXT:   }) {lower_bound = () -> (0), step = 1 : index, upper_bound = () -> (128)} : () -> ()
CHECK-NEXT:   return
CHECK-NEXT: }
)";

  EXPECT_TRUE(
      RunFileCheck(CompileHloConvAndGetMlir(hlo_text), expected_mlir_pattern)
          .ValueOrDie());
}

}  // namespace
}  // namespace gpu_mlir
}  // namespace xla
