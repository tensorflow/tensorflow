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

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // TF:local_config_mlir
#include "mlir/IR/AffineMap.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Function.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/MLIRContext.h"  // TF:local_config_mlir
#include "mlir/IR/Module.h"  // TF:local_config_mlir
#include "mlir/IR/StandardTypes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace gpu_mlir {
namespace {

mlir::Type ShapeToMlirType(const xla::Shape& shape, mlir::Builder* builder) {
  CHECK(shape.IsArray());
  mlir::Type element_type = [&] {
    switch (shape.element_type()) {
      case xla::F16:
        return builder->getF16Type();
      case xla::F32:
        return builder->getF32Type();
      default:
        break;
    }
    CHECK(false);
  }();
  std::vector<int64_t> physical_layout;
  std::vector<mlir::AffineExpr> major_to_minor;
  for (int i = shape.layout().minor_to_major_size() - 1; i >= 0; i--) {
    major_to_minor.push_back(
        builder->getAffineDimExpr(shape.layout().minor_to_major(i)));
    physical_layout.push_back(
        shape.dimensions(shape.layout().minor_to_major(i)));
  }
  return builder->getMemRefType(
      physical_layout, element_type,
      {builder->getAffineMap(major_to_minor.size(), 0, major_to_minor)});
}

std::string CompileHloConvAndGetMlir(absl::string_view hlo_text) {
  xla::HloModuleConfig hlo_config;
  xla::HloModule hlo_module("Conv", hlo_config);
  TF_CHECK_OK(xla::ParseHloString(hlo_text, &hlo_module));
  xla::HloInstruction* conv =
      hlo_module.entry_computation()->root_instruction();

  mlir::MLIRContext context;
  mlir::Location location = mlir::UnknownLoc::get(&context);
  auto mlir_module = mlir::ModuleOp::create(location);
  mlir::OpBuilder builder(&context);

  xla::Shape input_shape = conv->operand(0)->shape();
  xla::Shape filter_shape = conv->operand(1)->shape();
  xla::Shape output_shape = conv->shape().tuple_shapes(0);

  auto function = mlir::FuncOp::create(
      location, "Conv",
      builder.getFunctionType({ShapeToMlirType(output_shape, &builder),
                               ShapeToMlirType(input_shape, &builder),
                               ShapeToMlirType(filter_shape, &builder)},
                              {builder.getNoneType()}));
  mlir_module.push_back(function);

  auto* entry_block = function.addEntryBlock();
  builder.setInsertionPointToStart(entry_block);

  TF_CHECK_OK(xla::mlir_gpu::EmitConvolutionForwardAsMlir(
      conv, entry_block->getArgument(1), entry_block->getArgument(2),
      entry_block->getArgument(0), builder));

  mlir_module.verify();

  std::string mlir_text;
  llvm::raw_string_ostream strstream(mlir_text);
  function.print(strstream);
  return mlir_text;
}

// TODO(timshen): integrate this with mlir_gpu's testing infrastructure.
TEST(ConvEmtiterTest, TestDefault) {
  std::string hlo_text = R"(HloModule TestModule
ENTRY %TestComputation {
  %param_0 = f16[128,4,224,224]{1,3,2,0} parameter(0)
  %param_1 = f16[7,7,64,4]{3,1,0,2} parameter(1)
  ROOT %custom-call.1 = (f16[128,64,112,112]{1,3,2,0}, u8[0]{0}) custom-call(%param_0, %param_1), window={size=7x7 stride=2x2 pad=3_3x3_3}, dim_labels=bf01_01oi->bf01, custom_call_target="__cudnn$convForward", backend_config="{conv_result_scale:1}"
})";

  std::string expected_mlir_pattern =
      R"(
CHECK: func @Conv(%arg0: memref<128x112x112x64xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>, %arg1: memref<128x224x224x4xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>, %arg2: memref<64x7x7x4xf16, (d0, d1, d2, d3) -> (d2, d0, d1, d3)>) -> none {
CHECK-NEXT:   %0 = memref_cast %arg2 : memref<64x7x7x4xf16, (d0, d1, d2, d3) -> (d2, d0, d1, d3)> to memref<64x7x7x4xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>
CHECK-NEXT:   "affine.for"() ( {
CHECK-NEXT:   ^bb0(%arg3: index):
CHECK-NEXT:     "affine.for"() ( {
CHECK-NEXT:     ^bb0(%arg4: index):
CHECK-NEXT:       "affine.for"() ( {
CHECK-NEXT:       ^bb0(%arg5: index):
CHECK-NEXT:         "affine.for"() ( {
CHECK-NEXT:         ^bb0(%arg6: index):
CHECK-NEXT:           %1 = alloc() : memref<32x16xf32>
CHECK-NEXT:           "affine.for"() ( {
CHECK-NEXT:           ^bb0(%arg7: index):
CHECK-NEXT:             "affine.for"() ( {
CHECK-NEXT:             ^bb0(%arg8: index):
CHECK-NEXT:               %cst = constant 0.000000e+00 : f32
CHECK-NEXT:               "affine.store"(%cst, %1, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (f32, memref<32x16xf32>, index, index) -> ()
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
CHECK-NEXT:                       %2 = "affine.load"(%arg1, %arg3, %arg9, %arg5, %arg6, %arg10, %arg11, %arg12, %arg8) {map = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 * 4 + d6, d2 * 2 + d4 - 3, (d3 * 16 + d7) * 2 + d5 - 3)} : (memref<128x224x224x4xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>, index, index, index, index, index, index, index, index) -> f16
CHECK-NEXT:                       %3 = "affine.load"(%0, %arg4, %arg9, %arg10, %arg11, %arg12, %arg7) {map = (d0, d1, d2, d3, d4, d5) -> (d0 * 32 + d5, d1 * 4 + d4, d2, d3)} : (memref<64x7x7x4xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>, index, index, index, index, index, index) -> f16
CHECK-NEXT:                       %4 = "affine.load"(%1, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (memref<32x16xf32>, index, index) -> f32
CHECK-NEXT:                       %5 = mulf %2, %3 : f16
CHECK-NEXT:                       %6 = "std.addf"(%4, %5) : (f32, f16) -> f32
CHECK-NEXT:                       "affine.store"(%6, %1, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (f32, memref<32x16xf32>, index, index) -> ()
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
CHECK-NEXT:               %2 = "affine.load"(%1, %arg7, %arg8) {map = (d0, d1) -> (d0, d1)} : (memref<32x16xf32>, index, index) -> f32
CHECK-NEXT:               "affine.store"(%2, %arg0, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8) {map = (d0, d1, d2, d3, d4, d5) -> (d0, d1 * 32 + d4, d2, d3 * 16 + d5)} : (f32, memref<128x112x112x64xf16, (d0, d1, d2, d3) -> (d0, d2, d3, d1)>, index, index, index, index, index, index) -> ()
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
CHECK-NEXT: }
)";

  EXPECT_TRUE(
      RunFileCheck(CompileHloConvAndGetMlir(hlo_text), expected_mlir_pattern)
          .ValueOrDie());
}

}  // namespace
}  // namespace gpu_mlir
}  // namespace xla
