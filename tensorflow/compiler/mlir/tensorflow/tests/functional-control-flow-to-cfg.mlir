// RUN: tf-opt %s -tf-functional-control-flow-to-cfg -split-input-file -verify-diagnostics | FileCheck %s

func @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>)
func @testIf1Result(tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK:   %0 = extract_element %arg0[] : tensor<i1>
// CHECK:   cond_br %0, ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %1 = call @testIf1Then(%arg1, %arg2)
// CHECK:   br ^bb3(%1 : tensor<*xf32>)
// CHECK: ^bb2:
// CHECK:   %2 = call @testIf1Else(%arg1, %arg2)
// CHECK:   br ^bb3(%2 : tensor<*xf32>)
// CHECK: ^bb3(%3: tensor<*xf32>):

  return %1 : tensor<*xf32>
// CHECK:   return %3 : tensor<*xf32>
}

func @testIf3Then(tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
func @testIf3Else(tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)

// CHECK-LABEL: func @testIf3Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>)
func @testIf3Result(tensor<i1>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>) {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>):
  %1:3 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIf3Then, else_branch = @testIf3Else, is_stateless = false
  } : (tensor<i1>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)

// CHECK:   %0 = extract_element %arg0[] : tensor<i1>
// CHECK:   cond_br %0, ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %1:3 = call @testIf3Then(%arg1)
// CHECK:   br ^bb3(%1#0, %1#1, %1#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
// CHECK: ^bb2:
// CHECK:   %2:3 = call @testIf3Else(%arg1)
// CHECK:   br ^bb3(%2#0, %2#1, %2#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>)
// CHECK: ^bb3(%3: tensor<*xf32>, %4: tensor<*xi8>, %5: tensor<*xbf16>):
  return %1#0, %1#1, %1#2 : tensor<*xf32>, tensor<*xi8>, tensor<*xbf16>
// CHECK:  return %3, %4, %5
}

// -----

func @testIfThen(%arg0: tensor<!tf.variant>) -> tensor<!tf.variant> {
  return %arg0 : tensor<!tf.variant>
}
func @testIfElse(%arg0: tensor<!tf.variant>) -> tensor<!tf.variant> {
  return %arg0 : tensor<!tf.variant>
}

// CHECK-LABEL: func @testIfCasts(%arg0: tensor<i1>, %arg1: tensor<!tf.variant<tensor<f32>>>) -> tensor<!tf.variant<tensor<f32>>>
func @testIfCasts(%arg0: tensor<i1>, %arg1: tensor<!tf.variant<tensor<f32>>>) -> tensor<!tf.variant<tensor<f32>>> {
  %0 = "tf.If"(%arg0, %arg1) {
    then_branch = @testIfThen, else_branch = @testIfElse, is_stateless = false
  } : (tensor<i1>, tensor<!tf.variant<tensor<f32>>>) -> tensor<!tf.variant<tensor<f32>>>
  return %0: tensor<!tf.variant<tensor<f32>>>
// CHECK:   %0 = extract_element %arg0[] : tensor<i1>
// CHECK:   cond_br %0, ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   %1 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf.variant<tensor<f32>>>) -> tensor<!tf.variant>
// CHECK:   %2 = call @testIfThen(%1) : (tensor<!tf.variant>) -> tensor<!tf.variant>
// CHECK:   %3 = "tf.Cast"(%2) {Truncate = false} : (tensor<!tf.variant>) -> tensor<!tf.variant<tensor<f32>>>
// CHECK:   br ^bb3(%3 : tensor<!tf.variant<tensor<f32>>>)
// CHECK: ^bb2:
// CHECK:   %4 = "tf.Cast"(%arg1) {Truncate = false} : (tensor<!tf.variant<tensor<f32>>>) -> tensor<!tf.variant>
// CHECK:   %5 = call @testIfElse(%4) : (tensor<!tf.variant>) -> tensor<!tf.variant>
// CHECK:   %6 = "tf.Cast"(%5) {Truncate = false} : (tensor<!tf.variant>) -> tensor<!tf.variant<tensor<f32>>>
// CHECK:   br ^bb3(%6 : tensor<!tf.variant<tensor<f32>>>)
// CHECK: ^bb3(%7: tensor<!tf.variant<tensor<f32>>>):
// CHECK:   return %7 : tensor<!tf.variant<tensor<f32>>>
}

// -----

// If with a 4xi1 condition.

func @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

func @testIf1x4(tensor<4xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<4xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):

  // expected-error @+1 {{only supports zero-D bool tensors now}}
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else, is_stateless = false
  } : (tensor<4xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  return %1 : tensor<*xf32>
}

// -----


func @testWhile2Cond(tensor<*xf32>, tensor<*xf32>) -> (tensor<i1>)
func @testWhile2Body(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

// CHECK-LABEL: func @testWhile2Result(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
func @testWhile2Result(tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
^bb0(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>):
  %1:2 = "tf.While"(%arg0, %arg1) {
    cond = @testWhile2Cond, body = @testWhile2Body, is_stateless = false
  } : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

// CHECK:   br ^bb1(%arg0, %arg1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb1(%0: tensor<*xf32>, %1: tensor<*xf32>):
// CHECK:   %2 = call @testWhile2Cond(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<i1>
// CHECK:   %3 = extract_element %2[] : tensor<i1>
// CHECK:   cond_br %3, ^bb2(%0, %1 : tensor<*xf32>, tensor<*xf32>), ^bb3(%0, %1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb2(%4: tensor<*xf32>, %5: tensor<*xf32>):
// CHECK:   %6:2 = call @testWhile2Body(%4, %5) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
// CHECK:   br ^bb1(%6#0, %6#1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb3(%7: tensor<*xf32>, %8: tensor<*xf32>):

  return %1#0, %1#1 : tensor<*xf32>, tensor<*xf32>
// CHECK:   return %7, %8 : tensor<*xf32>, tensor<*xf32>
}


func @testWhile0Cond() -> (tensor<i1>)
func @testWhile0Body() -> ()

// CHECK-LABEL: func @testWhile0Result() {
func @testWhile0Result() {

^bb0:
  "tf.While"() { cond = @testWhile0Cond, body = @testWhile0Body, is_stateless = false } : () -> ()
// CHECK:   br ^bb1
// CHECK: ^bb1:
// CHECK:   %0 = call @testWhile0Cond() : () -> tensor<i1>
// CHECK:   %1 = extract_element %0[] : tensor<i1>
// CHECK:   cond_br %1, ^bb2, ^bb3
// CHECK: ^bb2:
// CHECK:   call @testWhile0Body() : () -> ()
// CHECK:   br ^bb1
// CHECK: ^bb3:

  return
// CHECK:   return
}


// CHECK-LABEL: func @testComplexWhile1Result(%arg0: tensor<*xf32>) -> tensor<*xf32> {
func @testComplexWhile1Result(tensor<*xf32>) -> (tensor<*xf32>) {

^bb0(%arg0: tensor<*xf32>):
  br ^bb1(%arg0, %arg0 : tensor<*xf32>, tensor<*xf32>)
^bb1(%0: tensor<*xf32>, %1: tensor<*xf32>):
  %2 = addf %0, %1 : tensor<*xf32>
  %3:2 = "tf.While"(%0, %2) {
    cond = @testWhile2Cond, body = @testWhile2Body, is_stateless = false
  } : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

// CHECK:   br ^bb2(%0, %2 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb2(%3: tensor<*xf32>, %4: tensor<*xf32>):
// CHECK:   %5 = call @testWhile2Cond(%3, %4) : (tensor<*xf32>, tensor<*xf32>) -> tensor<i1>
// CHECK:   %6 = extract_element %5[] : tensor<i1>
// CHECK:   cond_br %6, ^bb3(%3, %4 : tensor<*xf32>, tensor<*xf32>), ^bb4(%3, %4 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb3(%7: tensor<*xf32>, %8: tensor<*xf32>):
// CHECK:   %9:2 = call @testWhile2Body(%7, %8) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)
// CHECK:   br ^bb2(%9#0, %9#1 : tensor<*xf32>, tensor<*xf32>)
// CHECK: ^bb4(%10: tensor<*xf32>, %11: tensor<*xf32>):

// CHECK:   br ^bb5(%11, %2 : tensor<*xf32>, tensor<*xf32>)
  br ^bb2(%3#1, %2 : tensor<*xf32>, tensor<*xf32>)

// CHECK: ^bb5(%12: tensor<*xf32>, %13: tensor<*xf32>):
^bb2(%4: tensor<*xf32>, %5: tensor<*xf32>):
  %6 = subf %0, %1 : tensor<*xf32>

  return %6 : tensor<*xf32>
// CHECK:   return %14 : tensor<*xf32>
}

// -----

func @testWhileCond(%arg0: tensor<!tf.variant>) -> (tensor<i1>) {
  %true = "tf.Const"() { value = dense<true> : tensor<i1> } : () -> (tensor<i1>)
  return %true : tensor<i1>
}
func @testWhileBody(%arg0: tensor<!tf.variant<tensor<1x?xf32>>>) -> (tensor<!tf.variant<tensor<?x?xf32>>>) {
  %0 = "tf.Cast"(%arg0) : (tensor<!tf.variant<tensor<1x?xf32>>>) -> tensor<!tf.variant<tensor<?x?xf32>>>
  return %0 : tensor<!tf.variant<tensor<?x?xf32>>>
}

// CHECK-LABEL: func @testWhileCasts(%arg0: tensor<!tf.variant<tensor<1x3xf32>>>) -> tensor<!tf.variant<tensor<*xf32>>>
func @testWhileCasts(%arg0: tensor<!tf.variant<tensor<1x3xf32>>>) -> (tensor<!tf.variant<tensor<*xf32>>>) {
  %0 = "tf.While"(%arg0) {
    cond = @testWhileCond, body = @testWhileBody, is_stateless = false
  } : (tensor<!tf.variant<tensor<1x3xf32>>>) -> (tensor<!tf.variant<tensor<*xf32>>>)
  return %0 : tensor<!tf.variant<tensor<*xf32>>>
// CHECK:   %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<!tf.variant<tensor<1x3xf32>>>) -> tensor<!tf.variant>
// CHECK:   br ^bb1(%0 : tensor<!tf.variant>)
// CHECK: ^bb1(%1: tensor<!tf.variant>):        // 2 preds: ^bb0, ^bb2
// CHECK:   %2 = call @testWhileCond(%1) : (tensor<!tf.variant>) -> tensor<i1>
// CHECK:   %3 = extract_element %2[] : tensor<i1>
// CHECK:   %4 = "tf.Cast"(%1) {Truncate = false} : (tensor<!tf.variant>) -> tensor<!tf.variant<tensor<1x?xf32>>>
// CHECK:   cond_br %3, ^bb2(%4 : tensor<!tf.variant<tensor<1x?xf32>>>), ^bb3(%4 : tensor<!tf.variant<tensor<1x?xf32>>>)
// CHECK: ^bb2(%5: tensor<!tf.variant<tensor<1x?xf32>>>):       // pred: ^bb1
// CHECK:   %6 = call @testWhileBody(%5) : (tensor<!tf.variant<tensor<1x?xf32>>>) -> tensor<!tf.variant<tensor<?x?xf32>>>
// CHECK:   %7 = "tf.Cast"(%6) {Truncate = false} : (tensor<!tf.variant<tensor<?x?xf32>>>) -> tensor<!tf.variant>
// CHECK:   br ^bb1(%7 : tensor<!tf.variant>)
// CHECK: ^bb3(%8: tensor<!tf.variant<tensor<1x?xf32>>>):       // pred: ^bb1
// CHECK:   %9 = "tf.Cast"(%8) {Truncate = false} : (tensor<!tf.variant<tensor<1x?xf32>>>) -> tensor<!tf.variant<tensor<*xf32>>>
// CHECK:   return %9 : tensor<!tf.variant<tensor<*xf32>>>

}

// -----
