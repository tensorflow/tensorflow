// RUN: mlir_fusions_opt %s -split-input-file -inline='default-pipeline=''' | FileCheck %s

module {
  func.func private @mul(%a: f32, %b: f32) -> f32 {
    %ret = arith.mulf %a, %b : f32
    return %ret : f32
  }

  func.func private @add(%a: f32, %b: f32) -> f32 {
    %add = arith.addf %a, %b : f32
    %ret = xla_gpu.pure_call @mul(%add, %add) : (f32, f32) -> (f32)
    return %ret : f32
  }

  func.func @caller(%a: f32, %b: f32) -> f32 {
    %ret = xla_gpu.pure_call @add(%a, %b) : (f32, f32) -> (f32)
    return %ret : f32
  }
}
// CHECK: @caller
// CHECK-NOT: xla_gpu.pure_call @add
// CHECK: arith.addf
// CHECK-NOT: xla_gpu.pure_call @mul
// CHECK: arith.mulf

// -----

module {
  func.func @fused_computation(%arg0: tensor<2xf32> {xla.slice_index = 0 : index}, %arg1: tensor<2xf32> {xla.slice_index = 1 : index}, %arg2: tensor<2xf32> {xla.slice_index = 2 : index}) -> tensor<2xf32> attributes {xla.entry} {
    %0 = gpu.thread_id  x {xla.range = [0 : index, 1 : index]}
    %1 = xla_gpu.pure_call @fused_computation_atan2(%arg0, %arg1, %0) : (tensor<2xf32>, tensor<2xf32>, index) -> f32
    %inserted = tensor.insert %1 into %arg2[%0] : tensor<2xf32>
    return %inserted : tensor<2xf32>
  }
  func.func private @fused_computation_atan2(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>, %arg2: index {xla.range = [0 : index, 1 : index]}) -> f32 attributes {llvm.linkage = #llvm.linkage<internal>} {
    %extracted = tensor.extract %arg0[%arg2] : tensor<2xf32>
    %extracted_0 = tensor.extract %arg1[%arg2] : tensor<2xf32>
    %0 = arith.addf %extracted, %extracted_0 : f32
    %1 = arith.subf %extracted, %extracted_0 : f32
    %2 = arith.mulf %0, %1 : f32
    %3 = arith.divf %0, %1 : f32
    %4 = math.atan2 %2, %3 : f32
    return %4 : f32
  }
}
// CHECK: @fused_computation
// CHECK-NOT: xla_gpu.pure_call @add
// CHECK: gpu.thread_id
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: tensor.extract
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.subf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: arith.divf
// CHECK-NEXT: math.atan2
// CHECK-NEXT: tensor.insert
