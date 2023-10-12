// RUN: tf-opt -convert-tf-control-flow-to-scf %s | FileCheck %s

// `tf.IfRegion` which returns values gets converted to `scf.if`.
func.func private @test_if_then1(tensor<4xf32>) -> tensor<4xf32>
func.func private @test_if_else1(tensor<4xf32>) -> tensor<4xf32>
// CHECK-LABEL: func @test_supported_lowering_of_tf_if_region1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<4xf32>)
func.func @test_supported_lowering_of_tf_if_region1(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> (tensor<*xf32>, tensor<4xf32>) {
  %res:2 = "tf.IfRegion"(%arg0) ({
    %call = func.call @test_if_then1(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
    %add = "tf.AddV2"(%call, %call) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "tf.Yield"(%call, %add) : (tensor<4xf32>, tensor<4xf32>) -> ()
  },  {
    %call_0 = func.call @test_if_else1(%arg1) : (tensor<4xf32>) -> tensor<4xf32>
    "tf.Yield"(%call_0, %call_0) : (tensor<4xf32>, tensor<4xf32>) -> ()
  }) {is_stateless = false} : (tensor<i1>) -> (tensor<*xf32>, tensor<4xf32>)
  func.return %res#0, %res#1 : tensor<*xf32>, tensor<4xf32>

  // CHECK-NEXT: %[[COND:.*]] = tensor.extract %[[ARG0]][] : tensor<i1>
  // CHECK-NEXT: %[[RES:.*]]:2 = scf.if %[[COND]] -> (tensor<*xf32>, tensor<4xf32>) {
  // CHECK-NEXT:   %[[CALL:.*]] = func.call @test_if_then1(%[[ARG1]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[ADD:.*]] = "tf.AddV2"(%[[CALL]], %[[CALL]]) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[CAST:.*]] = "tf.Cast"(%[[CALL]]) {Truncate = false} : (tensor<4xf32>) -> tensor<*xf32>
  // CHECK-NEXT:   scf.yield %[[CAST]], %[[ADD]] : tensor<*xf32>, tensor<4xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   %[[CALL_0:.*]] = func.call @test_if_else1(%[[ARG1]]) : (tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT:   %[[CAST_0:.*]] = "tf.Cast"(%[[CALL_0]]) {Truncate = false} : (tensor<4xf32>) -> tensor<*xf32>
  // CHECK-NEXT:   scf.yield %[[CAST_0]], %[[CALL_0]] : tensor<*xf32>, tensor<4xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]]#0, %[[RES]]#1 : tensor<*xf32>, tensor<4xf32>
}

// `tf.IfRegion` which doesn't return values gets converted to `scf.if`.
func.func private @test_if_then2(tensor<4xf32>) -> ()
func.func private @test_if_else2(tensor<4xf32>) -> ()
// CHECK-LABEL: func @test_supported_lowering_of_tf_if_region2
// CHECK-SAME: (%[[ARG0:.*]]: tensor<i1>, %[[ARG1:.*]]: tensor<4xf32>)
func.func @test_supported_lowering_of_tf_if_region2(%arg0: tensor<i1>, %arg1: tensor<4xf32>) -> () {
  "tf.IfRegion"(%arg0) ({
    func.call @test_if_then2(%arg1) : (tensor<4xf32>) -> ()
    "tf.Yield"() : () -> ()
  },  {
    func.call @test_if_else2(%arg1) : (tensor<4xf32>) -> ()
    "tf.Yield"() : () -> ()
  }) {is_stateless = false} : (tensor<i1>) -> ()
  func.return

  // CHECK-NEXT: %[[COND:.*]] = tensor.extract %[[ARG0]][] : tensor<i1>
  // CHECK-NEXT: scf.if %[[COND]] {
  // CHECK-NEXT:   func.call @test_if_then2(%[[ARG1]]) : (tensor<4xf32>) -> ()
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   func.call @test_if_else2(%[[ARG1]]) : (tensor<4xf32>) -> ()
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// `tf.WhileRegion` gets converted to `scf.while`.
// CHECK-LABEL: func @test_supported_lowering_of_tf_while_region
// CHECK-SAME: (%[[ARG0:.*]]: tensor<f32>, %[[ARG1:.*]]: tensor<f32>, %[[ARG2:.*]]: tensor<*xf32>)
func.func @test_supported_lowering_of_tf_while_region(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<*xf32>) -> (tensor<f32>){
  %0:2 = "tf.WhileRegion"(%arg0, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<*xf32>):
    %1 = "tf.Identity"(%arg3) : (tensor<f32>) -> tensor<f32>
    %2 = "tf.Add"(%arg1, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "tf.NotEqual"(%1, %2) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "tf.Yield"(%3) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<*xf32>):
    %cst = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
    %1 = "tf.Sub"(%arg3, %cst) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "tf.Yield"(%1, %arg4) : (tensor<f32>, tensor<*xf32>) -> ()
  }) {is_stateless = false} : (tensor<f32>, tensor<*xf32>) -> (tensor<f32>, tensor<*xf32>)
  func.return %0#0 : tensor<f32>

  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-NEXT: %[[RES:.*]]:2 = scf.while (%[[ARG3:.*]] = %[[ARG0]], %[[ARG4:.*]] = %[[ARG2]]) : (tensor<f32>, tensor<*xf32>) -> (tensor<f32>, tensor<*xf32>) {
  // CHECK-NEXT:   %[[IDEN:.*]] = "tf.Identity"(%[[ARG3]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:   %[[ADD:.*]] = "tf.Add"(%[[ARG1]], %[[ARG3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:   %[[NOT_EQUAL:.*]] = "tf.NotEqual"(%[[IDEN]], %[[ADD]]) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:   %[[CONDITION:.*]] = tensor.extract %[[NOT_EQUAL]][] : tensor<i1>
  // CHECK-NEXT:   scf.condition(%[[CONDITION]]) %[[ARG3]], %[[ARG4]] : tensor<f32>, tensor<*xf32>
  // CHECK-NEXT: } do {
  // CHECK-NEXT: ^bb0(%[[ARG3]]: tensor<f32>, %[[ARG4]]: tensor<*xf32>):
  // CHECK-NEXT:   %[[SUB:.*]] = "tf.Sub"(%[[ARG3]], %[[CST]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:   scf.yield %[[SUB]], %[[ARG4]] : tensor<f32>, tensor<*xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[RES]]#0 : tensor<f32>
}
