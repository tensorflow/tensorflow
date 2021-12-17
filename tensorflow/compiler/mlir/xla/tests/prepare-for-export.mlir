// RUN: xla-opt -xla-prepare-for-export %s | FileCheck %s

// CHECK-LABEL: func @splat_constants
func @splat_constants() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
  return %cst : tensor<1x64x224x224xf32>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[CST]])
  // CHECK-SAME: (tensor<f32>) -> tensor<1x64x224x224xf32>
}

// -----

// CHECK-LABEL: @while_with_implicit_arg_capture
func @while_with_implicit_arg_capture(%arg0: tensor<i64>) -> tensor<i64> {
  // CHECK: mhlo.while
  // CHECK-SAME: (%arg0, %arg0)
  // CHECK-NEXT: %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: tensor
  %0 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tensor<i64>):
    // CHECK: mhlo.compare
    // CHECK-SAME: %[[ARG2]], %[[ARG1]]
    %1 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "LT"} : (tensor<i64>, tensor<i64>) -> tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  },  {
  // CHECK: ^bb0(%[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: tensor
  ^bb0(%arg1: tensor<i64>):
    // CHECK: %[[ADD:.*]] = mhlo.add %[[ARG1]], %[[ARG1]]
    %2 = mhlo.add %arg1, %arg1 : tensor<i64>
    // CHECK: mhlo.return
    // CHECK-SAME: %[[ADD]], %[[ARG2]]
    "mhlo.return"(%2) : (tensor<i64>) -> ()
  }) : (tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: @while_with_implicit_capture
func @while_with_implicit_capture(%arg0 :  tuple<tensor<i1>, tensor<5xi32>>) -> tuple<tensor<i1>, tensor<5xi32>> {
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = mhlo.constant dense<false> : tensor<i1>
  // Check that the iota implicit capture is made explicit
  // CHECK: %[[IOTA:.*]] = "mhlo.iota
  %2 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<5xi32>
  // CHECK: mhlo.while{{.*}} %[[IOTA]])
  %5 = "mhlo.while"(%arg0) ( {
  ^bb0(%arg1: tuple<tensor<i1>, tensor<5xi32>>):  // no predecessors
    %6 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i1>, tensor<5xi32>>) -> tensor<i1>
    "mhlo.return"(%6) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tuple<tensor<i1>, tensor<5xi32>>):  // no predecessors
    %6 = "mhlo.get_tuple_element"(%arg1) {index = 1 : i32} : (tuple<tensor<i1>, tensor<5xi32>>) -> tensor<5xi32>
    %i1 = "mhlo.get_tuple_element"(%arg1) {index = 0 : i32} : (tuple<tensor<i1>, tensor<5xi32>>) -> tensor<i1>
    %8 = "mhlo.tuple"(%i1, %2) : (tensor<i1>, tensor<5xi32>) -> tuple<tensor<i1>, tensor<5xi32>>
    "mhlo.return"(%8) : (tuple<tensor<i1>, tensor<5xi32>>) -> ()
  }) : (tuple<tensor<i1>, tensor<5xi32>>) -> tuple<tensor<i1>, tensor<5xi32>>
  return %5 : tuple<tensor<i1>, tensor<5xi32>>
  }
