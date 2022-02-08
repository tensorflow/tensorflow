// RUN: tf-tfrt-opt -tf-jitrt-pipeline="vectorize reduction-2d-tile-sizes=4,4" \
// RUN: %s -split-input-file | FileCheck %s

// CHECK-LABEL: @reduce_row_sum_2d_dynamic
func @reduce_row_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
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
func @reduce_column_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
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
func @reduce_row_mean_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Mean"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
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
func @reduce_1d_dynamic(%input: tensor<?xf32>) -> tensor<f32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?xf32>, tensor<1xi32>) -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK: scf.for
// CHECK:   arith.addf %{{.*}}, %{{.*}} : vector<8xf32>
// CHECK: vector.reduction
