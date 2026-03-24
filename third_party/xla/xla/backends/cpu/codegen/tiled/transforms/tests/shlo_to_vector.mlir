// RUN: fusion_compiler_opt %s --xtile-cpu-shlo-to-vector -split-input-file | FileCheck %s

func.func @transpose(%input : tensor<1024x32xf32>) -> tensor<32x1024xf32> {
  // CHECK-NOT: vector.transpose %{{.*}}, [1, 0] : vector<1024x32xf32> to vector<32x1024xf32>
  %transposed = stablehlo.transpose %input, dims = [1, 0] : (tensor<1024x32xf32>) -> tensor<32x1024xf32>
  return %transposed : tensor<32x1024xf32>
}
// -----

// CHECK-DAG: #[[LHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d0)>
// CHECK-DAG: #[[RHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[OUTPUT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
func.func @dot_general(%lhs : tensor<1024x32xf32>, %rhs : tensor<32x1024xf32>) -> tensor<1024x1024xf32> {
  // CHECK: %[[ACCUMULATOR:.*]] = arith.constant dense<0.000000e+00> : vector<1024x1024xf32>
  // CHECK: vector.contract
  // CHECK-SAME: {indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[OUTPUT_MAP]]],
  // CHECK-SAME: iterator_types = ["reduction", "parallel", "parallel"],
  // CHECK-SAME: kind = #vector.kind<add>}
  // CHECK-SAME: %[[ACCUMULATOR]] : vector<1024x32xf32>, vector<32x1024xf32> into vector<1024x1024xf32>
  %result = stablehlo.dot_general %lhs, %rhs, contracting_dims = [1] x [0] : (tensor<1024x32xf32>, tensor<32x1024xf32>) -> tensor<1024x1024xf32>
  return %result : tensor<1024x1024xf32>
}

// -----

// CHECK-DAG: #[[INPUT_MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[OUTPUT_MAP:.*]] = affine_map<(d0) -> ()>
func.func @dot_scalar_output(%lhs : tensor<1024xf32>, %rhs : tensor<1024xf32>) -> tensor<f32> {
  // CHECK: %[[ACCUMULATOR:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[RESULT:.*]] = vector.contract
  // CHECK-SAME: {indexing_maps = [#[[INPUT_MAP]], #[[INPUT_MAP]], #[[OUTPUT_MAP]]],
  // CHECK-SAME: iterator_types = ["reduction"],
  // CHECK-SAME: kind = #vector.kind<add>}
  // CHECK-SAME: %[[ACCUMULATOR]] : vector<1024xf32>, vector<1024xf32> into f32
  // CHECK: %[[RESULT_TENSOR:.*]] = tensor.from_elements %[[RESULT]] : tensor<f32>
  %result = stablehlo.dot_general %lhs, %rhs, contracting_dims = [0] x [0] : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<f32>
  // CHECK: return %[[RESULT_TENSOR]] : tensor<f32>
  return %result : tensor<f32>
}

// -----


func.func @reduce_outer(%input : tensor<1024x32xf32>, %init : tensor<f32>) -> tensor<32xf32> {
  %result = stablehlo.reduce(%input init: %init) across dimensions = [0] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<32xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = arith.addf %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  return %result : tensor<32xf32>
}

// CHECK: func.func @reduce_outer
// CHECK:   memref.alloca() : memref<32xf32>
// CHECK:   vector.transfer_read %{{.*}} : tensor<1024x32xf32>, vector<32xf32>
// CHECK:   scf.for
// CHECK:     vector.transfer_read %{{.*}} : tensor<1024x32xf32>, vector<32xf32>
// CHECK:     arith.addf {{.*}} : vector<32xf32>
// CHECK:     scf.yield %{{.*}} : vector<32xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<32xf32>, memref<32xf32>

// -----


func.func @reduce_inner(%input : tensor<1024x32xf32>, %init : tensor<f32>) -> tensor<1024xf32> {
  %result = stablehlo.reduce(%input init: %init) across dimensions = [1] : (tensor<1024x32xf32>, tensor<f32>) -> tensor<1024xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = arith.addf %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  return %result : tensor<1024xf32>
}

// CHECK: func.func @reduce_inner
// CHECK:   memref.alloca() : memref<1024xf32>
// CHECK:   scf.for
// CHECK:     vector.transfer_read {{.*}} : tensor<1024x32xf32>, vector<32xf32>
// CHECK:     vector.reduction <add>, {{.*}} : vector<32xf32> into f32
// CHECK:     memref.store {{.*}} : memref<1024xf32>
// CHECK:   }

// -----

func.func @reduce_middle(%input : tensor<1024x32x8xf32>, %init : tensor<f32>) -> tensor<1024x8xf32> {
  %result = stablehlo.reduce(%input init: %init) across dimensions = [1] : (tensor<1024x32x8xf32>, tensor<f32>) -> tensor<1024x8xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = arith.addf %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  return %result : tensor<1024x8xf32>
}

// CHECK: func.func @reduce_middle
// CHECK:   memref.alloca() : memref<1024x8xf32>
// CHECK:   scf.for
// CHECK:     vector.transfer_read {{.*}} : tensor<1024x32x8xf32>, vector<8xf32>
// CHECK:     scf.for
// CHECK:       vector.transfer_read %{{.*}} : tensor<1024x32x8xf32>, vector<8xf32>
// CHECK:       arith.addf {{.*}} : vector<8xf32>
// CHECK:       scf.yield {{.*}} : vector<8xf32>
// CHECK:     }
// CHECK:     vector.transfer_write {{.*}} : vector<8xf32>, memref<1024x8xf32>
// CHECK:   }
// CHECK: }

// -----

func.func @reduce_outer_and_inner(%input : tensor<1024x32x8xf32>, %init : tensor<f32>) -> tensor<32xf32> {
  %result = stablehlo.reduce(%input init: %init) across dimensions = [0, 2] : (tensor<1024x32x8xf32>, tensor<f32>) -> tensor<32xf32>
    reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
      %add = arith.addf %arg0, %arg1 : tensor<f32>
      stablehlo.return %add : tensor<f32>
    }
  return %result : tensor<32xf32>
}

// CHECK: func.func @reduce_outer_and_inner
// CHECK:   %[[BUFFER:.*]] = memref.alloca() : memref<32xf32>
// CHECK:   scf.for
// CHECK:     vector.transfer_read {{.*}} : tensor<1024x32x8xf32>, vector<8xf32>
// CHECK:     scf.for
// CHECK:       vector.transfer_read %{{.*}} : tensor<1024x32x8xf32>, vector<8xf32>
// CHECK:       arith.addf %{{.*}} : vector<8xf32>
// CHECK:       scf.yield {{.*}} : vector<8xf32>
// CHECK:     }
// CHECK:     vector.reduction <add>, {{.*}} : vector<8xf32> into f32
// CHECK:     memref.store {{.*}}, %[[BUFFER]]{{.*}} : memref<32xf32>
// CHECK:   }
// CHECK: }

// -----

func.func @broadcast_0D_tensor(%input : tensor<f32>) -> tensor<32xf32> {
  %result = stablehlo.broadcast_in_dim %input, dims = [] : (tensor<f32>) -> tensor<32xf32>
  return %result : tensor<32xf32>
}

// CHECK-LABEL: @broadcast_0D_tensor
// CHECK-NOT: vector.shape_cast
// CHECK: vector.broadcast {{.*}} : vector<f32> to vector<32xf32>

// -----

func.func @broadcast_2D_tensor_inner(%input : tensor<4xf32>) -> tensor<32x4xf32> {
  %result = stablehlo.broadcast_in_dim %input, dims = [1] : (tensor<4xf32>) -> tensor<32x4xf32>
  return %result : tensor<32x4xf32>
}

// CHECK-LABEL: @broadcast_2D_tensor_inner
// CHECK-NOT: vector.shape_cast
// CHECK: vector.broadcast {{.*}} : vector<4xf32> to vector<32x4xf32>

// -----

func.func @broadcast_2D_tensor_outer(%input : tensor<4xf32>) -> tensor<4x32xf32> {
  %result = stablehlo.broadcast_in_dim %input, dims = [0] : (tensor<4xf32>) -> tensor<4x32xf32>
  return %result : tensor<4x32xf32>
}

// CHECK-LABEL: @broadcast_2D_tensor_outer
// CHECK: vector.shape_cast {{.*}} : vector<4xf32> to vector<4x1xf32>
// CHECK: vector.broadcast {{.*}} : vector<4x1xf32> to vector<4x32xf32>

// -----

func.func @iota() -> tensor<4xi32> {
  %result = stablehlo.iota dim = 0 : tensor<4xi32>
  return %result : tensor<4xi32>
}

// CHECK-LABEL: @iota
// CHECK: arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>

