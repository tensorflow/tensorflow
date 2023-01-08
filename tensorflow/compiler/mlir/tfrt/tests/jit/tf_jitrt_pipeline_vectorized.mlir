// RUN: tf-tfrt-opt -tf-jitrt-pipeline="vectorize" \
// RUN: %s -split-input-file | FileCheck %s

// CHECK-LABEL: @reduce_row_sum_2d_dynamic
func.func @reduce_row_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK:   scf.for
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NOT: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>

// -----

// CHECK-LABEL: @reduce_column_sum_2d_dynamic
func.func @reduce_column_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK:   scf.for
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NOT: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>

// -----

// CHECK-LABEL: @reduce_row_mean_2d_dynamic
func.func @reduce_row_mean_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Mean"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK:   scf.for
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK:     arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK-NOT: arith.addf %{{.*}}, %{{.*}} : vector<4xf32>
// CHECK: scf.parallel
// CHECK:      vector.broadcast %{{.*}} : f32 to vector<8xf32>
// CHECK-NEXT: arith.divf %{{.*}}, %{{.*}} : vector<8xf32>

// -----

// CHECK-LABEL: @reduce_1d_dynamic
func.func @reduce_1d_dynamic(%input: tensor<?xf32>) -> tensor<f32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK: scf.for
// CHECK:   arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
// CHECK: vector.reduction

// -----

// CHECK-LABEL: @reduction_of_cast
func.func @reduction_of_cast(%arg0: tensor<?xi64>) -> tensor<i32> {
  %cst = "tf.Const"()
    {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Cast"(%arg0) {Truncate = false}
    : (tensor<?xi64>) -> tensor<?xi32>
  %1 = "tf.Prod"(%0, %cst) {keep_dims = false}
    : (tensor<?xi32>, tensor<1xi32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}
// CHECK: scf.for
// CHECK:   arith.trunci %{{.*}} : vector<4x8xi64> to vector<4x8xi32>
// CHECK:   arith.muli %{{.*}}, %{{.*}} : vector<8xi32>
// CHECK: vector.reduction
// CHECK: scf.for
// CHECK:   linalg.generic
// CHECK:     arith.trunci
// CHECK:     arith.muli
