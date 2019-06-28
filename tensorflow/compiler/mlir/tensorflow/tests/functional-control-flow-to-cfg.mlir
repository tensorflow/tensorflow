// RUN: tf-opt %s -tf-functional-control-flow-to-cfg -split-input-file -verify-diagnostics | FileCheck %s

func @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Result(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>)
func @testIf1Result(tensor<i1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else
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
    then_branch = @testIf3Then, else_branch = @testIf3Else
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

func @testIf1Then(tensor<2x?xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
func @testIf1Else(tensor<*xf32>, tensor<2x?xf32>) -> tensor<*xf32>

// CHECK-LABEL: func @testIf1Casts(%arg0: tensor<i1>, %arg1: tensor<2x2xf32>, %arg2: tensor<*xf32>)
func @testIf1Casts(tensor<i1>, tensor<2x2xf32>, tensor<*xf32>) -> tensor<2x?xf32> {
^bb0(%arg0: tensor<i1>, %arg1: tensor<2x2xf32>, %arg2: tensor<*xf32>):

  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else
  } : (tensor<i1>, tensor<2x2xf32>, tensor<*xf32>) -> tensor<2x?xf32>

// CHECK:  %0 = extract_element %arg0[] : tensor<i1>
// CHECK:  cond_br %0, ^bb1, ^bb2
// CHECK:^bb1:  // pred: ^bb0
// CHECK:  %1 = tensor_cast %arg1 : tensor<2x2xf32> to tensor<2x?xf32>
// CHECK:  %2 = tensor_cast %arg2 : tensor<*xf32> to tensor<2x2xf32>
// CHECK:  %3 = call @testIf1Then(%1, %2) : (tensor<2x?xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
// CHECK:  %4 = tensor_cast %3 : tensor<2x2xf32> to tensor<2x?xf32>
// CHECK:  br ^bb3(%4 : tensor<2x?xf32>)

// CHECK:^bb2:  // pred: ^bb0
// CHECK:  %5 = tensor_cast %arg1 : tensor<2x2xf32> to tensor<*xf32>
// CHECK:  %6 = tensor_cast %arg2 : tensor<*xf32> to tensor<2x?xf32>
// CHECK:  %7 = call @testIf1Else(%5, %6) : (tensor<*xf32>, tensor<2x?xf32>) -> tensor<*xf32>
// CHECK:  %8 = tensor_cast %7 : tensor<*xf32> to tensor<2x?xf32>
// CHECK:  br ^bb3(%8 : tensor<2x?xf32>)

// CHECK:^bb3(%9: tensor<2x?xf32>):	// 2 preds: ^bb1, ^bb2

  %2 = "tf.Add"(%1, %1) : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>
// CHECK:  %10 = "tf.Add"(%9, %9) : (tensor<2x?xf32>, tensor<2x?xf32>) -> tensor<2x?xf32>

  return %2 : tensor<2x?xf32>
// CHECK:  return %10 : tensor<2x?xf32>
}

// -----

// If with a 4xi1 condition.

func @testIf1Then(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
func @testIf1Else(tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

func @testIf1x4(tensor<4xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32> {
^bb0(%arg0: tensor<4xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>):

  // expected-error @+1 {{only supports zero-D bool tensors now}}
  %1 = "tf.If"(%arg0, %arg1, %arg2) {
    then_branch = @testIf1Then, else_branch = @testIf1Else
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
    cond = @testWhile2Cond, body = @testWhile2Body
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
  "tf.While"() { cond = @testWhile0Cond, body = @testWhile0Body } : () -> ()
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
    cond = @testWhile2Cond, body = @testWhile2Body
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

func @testWhileCond(tensor<?x3xf32>) -> (tensor<i1>)
func @testWhileBody(tensor<*xf32>) -> (tensor<?x?xf32>)

// CHECK-LABEL: func @testWhileCasts(%arg0: tensor<1x3xf32>)
func @testWhileCasts(%arg0: tensor<1x3xf32>) -> (tensor<?x?xf32>) {
  %0 = "tf.While"(%arg0) {
    cond = @testWhileCond, body = @testWhileBody
  } : (tensor<1x3xf32>) -> (tensor<?x?xf32>)

// CHECK:   %0 = tensor_cast %arg0 : tensor<1x3xf32> to tensor<?x3xf32>
// CHECK:   br ^bb1(%0 : tensor<?x3xf32>)
// CHECK: ^bb1(%1: tensor<?x3xf32>):
// CHECK:   %2 = call @testWhileCond(%1) : (tensor<?x3xf32>) -> tensor<i1>
// CHECK:   %3 = extract_element %2[] : tensor<i1>
// CHECK:   %4 = tensor_cast %1 : tensor<?x3xf32> to tensor<*xf32>
// CHECK:   cond_br %3, ^bb2(%4 : tensor<*xf32>), ^bb3(%4 : tensor<*xf32>)
// CHECK: ^bb2(%5: tensor<*xf32>):
// CHECK:   %6 = call @testWhileBody(%5) : (tensor<*xf32>) -> tensor<?x?xf32>
// CHECK:   %7 = tensor_cast %6 : tensor<?x?xf32> to tensor<?x3xf32>
// CHECK:   br ^bb1(%7 : tensor<?x3xf32>)
// CHECK: ^bb3(%8: tensor<*xf32>):
// CHECK:   %9 = tensor_cast %8 : tensor<*xf32> to tensor<?x?xf32>

  return %0 : tensor<?x?xf32>
// CHECK:   return %9 : tensor<?x?xf32>
}

// -----
