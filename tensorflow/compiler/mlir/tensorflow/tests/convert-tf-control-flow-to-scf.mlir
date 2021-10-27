// RUN: tf-opt -convert-tf-control-flow-to-scf %s | FileCheck %s

// `tf.IfRegion` which returns values gets converted to `scf.if`.
func private @test_if_then1(tensor<4xf32>) -> tensor<4xf32>
func private @test_if_else1(tensor<4xf32>) -> tensor<4xf32>
// CHECK-LABEL: func @test_supported_lowering_of_tf_if_region1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<4xf32>)
func @test_supported_lowering_of_tf_if_region1(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> (tensor<*xf32>, tensor<4xf32>) {
  %res:2 = "tf.IfRegion"(%arg0) ( {
    %call = call @test_if_then1(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
    %add = "tf.AddV2"(%call, %call) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "tf.Yield"(%call, %add) : (tensor<4xf32>, tensor<4xf32>) -> ()
  },  {
    %call_0 = call @test_if_else1(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
    "tf.Yield"(%call_0, %call_0) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> (tensor<*xf32>, tensor<4xf32>)
  return %res#0, %res#1 : tensor<*xf32>, tensor<4xf32>

  // CHECK-NEXT: %[[COND:.*]] = tensor.extract %[[ARG0]][] : tensor<i1>
  // CHECK-NEXT: %[[RES:.*]]:2 = scf.if %[[COND]] -> (tensor<*xf32>, tensor<4xf32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = call @test_if_then1(%[[ARG1]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = "tf.AddV2"(%[[CALL]], %[[CALL]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[CAST:.*]] = "tf.Cast"(%[[CALL]]) {Truncate = false} : (tensor<4xf32>) -> tensor<*xf32>
  // CHECK-NEXT:   scf.yield %[[CAST]], %[[ADD]] : tensor<*xf32>, tensor<4xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[CALL_0:.*]] = call @test_if_else1(%[[ARG1]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[CAST_0:.*]] = "tf.Cast"(%[[CALL_0]]) {Truncate = false} : (tensor<4xf32>) -> tensor<*xf32>
  // CHECK-NEXT:   scf.yield %[[CAST_0]], %[[CALL_0]] : tensor<*xf32>, tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1 : tensor<*xf32>, tensor<4xf32>
}

// `tf.IfRegion` which doesn't return values gets converted to `scf.if`.
func private @test_if_then2(tensor<4xf32>) -> ()
func private @test_if_else2(tensor<4xf32>) -> ()
// CHECK-LABEL: func @test_supported_lowering_of_tf_if_region2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<4xf32>)
func @test_supported_lowering_of_tf_if_region2(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> () {
  "tf.IfRegion"(%arg0) ( {
    call @test_if_then2(%arg1) : (tensor<4xf32>) -> ()
    "tf.Yield"() : () -> ()
  },  {
    call @test_if_else2(%arg1) : (tensor<4xf32>) -> ()
    "tf.Yield"() : () -> ()
  }) {is_stateless = false} : (tensor<i1>) -> ()
  return

  // CHECK-NEXT: %[[COND:.*]] = tensor.extract %[[ARG0]][] : tensor<i1>
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   call @test_if_then2(%[[ARG1]]) : (tensor<4xf32>) -> ()
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   call @test_if_else2(%[[ARG1]]) : (tensor<4xf32>) -> ()
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}
