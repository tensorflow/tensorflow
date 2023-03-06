// RUN: mlir-hlo-opt %s --gml-st-cpu-tiling-pipeline | FileCheck %s

func.func @sort(%input1: tensor<64x8x4xf32>, %input2: tensor<64x8x4xf32>,
                %init1: tensor<64x8x4xf32>, %init2: tensor<64x8x4xf32>)
                -> (tensor<64x8x4xf32>, tensor<64x8x4xf32>) {
  %res0, %res1 = thlo.sort
    ins(%input1: tensor<64x8x4xf32>, %input2: tensor<64x8x4xf32>)
    outs(%init1: tensor<64x8x4xf32>, %init2: tensor<64x8x4xf32>)
    dimension = 1
    is_stable = true
    (%e11: f32, %e12: f32, %e21: f32, %e22: f32) {
      %gt = arith.cmpf ogt, %e11, %e12: f32
      thlo.yield %gt : i1
    }
  func.return %res0, %res1: tensor<64x8x4xf32>, tensor<64x8x4xf32>
}
// CHECK-LABEL: func.func @sort(

// CHECK:      scf.forall
// CHECK:        thlo.sort
// CHECK-SAME:     ins(%{{.*}} : tensor<1x8x1xf32>, %{{.*}} : tensor<1x8x1xf32>)
// CHECK-SAME:     dimension = 1
// CHECK:        tensor.parallel_insert_slice
