// RUN: tf-tfrt-opt -split-input-file -tf-cpurt-pipeline=vectorize %s |\
// RUN: FileCheck %s

// CHECK-LABEL: @reduce_column_sum_2d_dynamic
func @reduce_column_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[1]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK:   scf.for
// CHECK:     vector.multi_reduction

// -----

// CHECK-LABEL: @reduce_row_sum_2d_dynamic
func @reduce_row_sum_2d_dynamic(%input: tensor<?x?xf32>) -> tensor<?xf32> {
  %dim_to_reduce =  "tf.Const"() {value = dense<[0]> : tensor<1xi32>}
     : () -> tensor<1xi32>
  %0 = "tf.Sum"(%input, %dim_to_reduce) {keep_dims = false}
      : (tensor<?x?xf32>, tensor<1xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}
// CHECK: linalg.fill
// CHECK: scf.parallel
// CHECK:   scf.for
// CHECK:     vector.multi_reduction
