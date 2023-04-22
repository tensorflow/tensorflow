// RUN: mlir-hlo-opt --hlo-canonicalize-reduction %s | FileCheck %s

// rank2 column reduction should not be converted
// CHECK-LABEL: @test_rank2_column_reduction
// CHECK-NOT: reshape
func @test_rank2_column_reduction(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// -----

// rank2 row reduction should not be converted
// CHECK-LABEL: @test_rank2_row_reduction
// CHECK-NOT: reshape
func @test_rank2_row_reduction(%arg0: tensor<?x?xf32>) -> tensor<?xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// -----

// rank3 column reduction
// CHECK-LABEL: @test_rank3_column_reduction
// CHECK: [[R1:%[a-zA-Z0-9]+]] = "mhlo.dynamic_reshape"
// CHECK-SAME: (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK-NEXT: [[R2:%[a-zA-Z0-9]+]] = "mhlo.reduce"
// CHECK-SAME: [[R1]]
// CHECK: dimensions = dense<0> : tensor<1xi64>
// CHECK: "mhlo.dynamic_reshape"
// CHECK-SAME:  (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
func @test_rank3_column_reduction(%arg0: tensor<?x?x?xf32>) -> tensor<?xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// // -----

// rank3 row reduction
// CHECK-LABEL: @test_rank3_row_reduction
// CHECK: [[R1:%[a-zA-Z0-9]+]] = "mhlo.dynamic_reshape"
// CHECK-SAME: (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK-NEXT: [[R2:%[a-zA-Z0-9]+]] = "mhlo.reduce"
// CHECK-SAME: [[R1]]
// CHECK: dimensions = dense<1> : tensor<1xi64>
// CHECK: "mhlo.dynamic_reshape"
// CHECK-SAME:  (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
func @test_rank3_row_reduction(%arg0: tensor<?x?x?xf32>) -> tensor<?xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?xf32>
  return %2 : tensor<?xf32>
}

// // -----

// reduce to scalar
// CHECK-LABEL: @test_reduce_to_scalar
// CHECK: [[R1:%[a-zA-Z0-9]+]] = "mhlo.dynamic_reshape"
// CHECK-SAME: (tensor<?x?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK-NEXT: [[R2:%[a-zA-Z0-9]+]] = "mhlo.reduce"
// CHECK-SAME: [[R1]]
// CHECK: dimensions = dense<0> : tensor<1xi64>
// CHECK: "mhlo.reshape"
// CHECK-SAME: (tensor<?xf32>) -> tensor<f32>
func @test_reduce_to_scalar(%arg0: tensor<?x?x?xf32>) -> tensor<f32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<f32>
  return %2 : tensor<f32>
}

// -----

// reduce the dimension in the middle, should not be converted.
// CHECK-LABEL: @test_mid_reduction
// CHECK-NOT: reshape
func @test_mid_reduction(%arg0: tensor<?x?x?xf32>) -> tensor<?x?xf32> {
  %0 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  %2 = "mhlo.reduce"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %4 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%4) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
