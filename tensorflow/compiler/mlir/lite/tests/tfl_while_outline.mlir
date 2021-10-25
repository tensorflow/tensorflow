// Test to verify loop outlining.

// RUN: tf-opt --split-input-file --tfl-while-loop-outline %s | FileCheck %s
// Check that while loop outlining is nop if re-ran.
// RUN: tf-opt --tfl-while-loop-outline %s -o %t1
// RUN: tf-opt --tfl-while-loop-outline %t1 -o %t2
// RUN: diff %t1 %t2

// CHECK-LABEL: func @while
func @while() -> tensor<1xf32>
    attributes {tf.entry_function = {outputs = "result"}} {
  %cst = arith.constant dense<1> : tensor<i32> loc("dec")
  %cst0 = arith.constant dense<5> : tensor<i32> loc("N")
  %cst1 = arith.constant dense<3.0> : tensor<1xf32> loc("val")
  %0:2 = "tfl.while"(%cst0, %cst1) ( {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_cond
      // CHECK-SAME: (tensor<*xi32>, tensor<*xf32>)
      %cst_0 = arith.constant dense<0> : tensor<i32>
      %1 = "tfl.greater"(%arg2, %cst_0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      // CHECK: call @WhileOp_body
      // CHECK-SAME: (tensor<*xi32>, tensor<*xf32>)
      %1 = "tfl.sub"(%arg2, %cst) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %arg3, %arg3 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  return %0#1 : tensor<1xf32>
}
// CHECK-LABEL: func private @WhileOp_cond(
// CHECK: tfl.greater
// CHECK-LABEL: func private @WhileOp_body(
// CHECK: tfl.sub
// CHECK: tfl.add

// -----

// CHECK-LABEL: func @while2
// Verify that while body//cond with implicitly captured values result in changing while operands/results.
func @while2(%cst : tensor<i32>) -> tensor<1xf32> attributes {tf.entry_function = {outputs = "result"}} {
  %cst_0 = arith.constant dense<5> : tensor<i32>
  %cst_1 = arith.constant dense<3.000000e+00> : tensor<1xf32>
  // Verifies 3 operands post outlining.
  // CHECK: "tfl.while"({{.*}}, {{.*}}, {{.*}}) (
  %0:2 = "tfl.while"(%cst_0, %cst_1) ( {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>):   // no predecessors
    // CHECK: call @WhileOp_cond
    // CHECK-SAME: (tensor<*xi32>, tensor<*xf32>, tensor<i32>)
    %1 = call @WhileOp_cond(%arg0, %arg1, %cst) : (tensor<*xi32>, tensor<*xf32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>):   // no predecessors
    // CHECK: call @WhileOp_body
    // CHECK-SAME: (tensor<*xi32>, tensor<*xf32>, tensor<i32>)
    %1:3 = call @WhileOp_body(%arg0, %arg1, %cst) : (tensor<*xi32>, tensor<*xf32>, tensor<i32>) -> (tensor<*xi32>, tensor<*xf32>, tensor<i32>)
    "tfl.yield"(%1#0, %1#1) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  // CHECK: (tensor<i32>, tensor<1xf32>, tensor<i32>) ->
  // CHECK-SAME: (tensor<i32>, tensor<1xf32>, tensor<i32>)
  return %0#1 : tensor<1xf32>
}

func private @WhileOp_cond(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>, %arg2: tensor<i32>) -> tensor<i1> {
  %cst = arith.constant dense<0> : tensor<i32>
  %0 = "tfl.greater"(%arg0, %cst) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>
}

func private @WhileOp_body(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>, %arg2: tensor<i32>) -> (tensor<*xi32>, tensor<*xf32>, tensor<i32>) {
  %0 = "tfl.sub"(%arg0, %arg2) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %1 = tfl.add %arg1, %arg1 {fused_activation_function = "NONE"} : tensor<*xf32>
  return %0, %1, %arg2 : tensor<*xi32>, tensor<*xf32>, tensor<i32>
}

// CHECK-LABEL: func private @WhileOp_cond(
// CHECK: tfl.greater
// CHECK-LABEL: func private @WhileOp_body(
// CHECK: tfl.sub
// CHECK: tfl.add

// -----

func @rnn(%arg0: tensor<4x4x3xf32> {tf.device = "/device:CPU:0"}) -> tensor<4x?x2xf32> attributes {tf.entry_function = {inputs = "Placeholder", outputs = "rnn/transpose_1"}} {
  %cst = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<8xf32>
  %cst_1 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<4x4x2xf32>
  %cst_3 = arith.constant dense<4> : tensor<i32>
  %cst_4 = arith.constant dense<1.000000e+00> : tensor<f32>
  %cst_5 = arith.constant dense<1> : tensor<i32>
  %cst_6 = arith.constant dense<0> : tensor<1xi32>
  %cst_7 = arith.constant dense<0> : tensor<i32>
  %cst_8 = arith.constant dense<-1> : tensor<1xi32>
  %cst_9 = arith.constant dense<-1> : tensor<i32>
  %cst_10 = arith.constant dense<2.1> : tensor<8x5xf32>
  %cst_11 = arith.constant dense<2> : tensor<1xi32>
  %cst_12 = arith.constant dense<1> : tensor<1xi32>
  %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<4x4x3xf32>, tensor<3xi32>) -> tensor<4x4x3xf32>
  %1:6 = "tfl.while"(%cst_7, %cst_7, %cst_2, %cst, %cst, %0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<*xf32>):  // no predecessors
    %5 = "tfl.less"(%arg2, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = "tfl.less"(%arg1, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = tfl.logical_and %6, %5 : tensor<i1>
    "tfl.yield"(%7) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf32>, %arg6: tensor<*xf32>):  // no predecessors
    %5 = tfl.add %arg2, %cst_5 {fused_activation_function = "NONE"} : tensor<i32>
    %6 = tfl.add %arg1, %cst_5 {fused_activation_function = "NONE"} : tensor<i32>
    %7 = "tfl.gather"(%0, %arg2) {axis = 0 : i32} : (tensor<4x4x3xf32>, tensor<i32>) -> tensor<4x3xf32>
    %8 = "tfl.concatenation"(%7, %arg5) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<4x3xf32>, tensor<4x2xf32>) -> tensor<4x5xf32>
    %9 = "tfl.fully_connected"(%8, %cst_10, %cst_0) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x5xf32>, tensor<8x5xf32>, tensor<8xf32>) -> tensor<4x8xf32>
    %10:4 = "tfl.split"(%cst_5, %9) {num_splits = 4 : i32} : (tensor<i32>, tensor<4x8xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>)
    %11 = "tfl.add"(%10#2, %cst_4) {fused_activation_function = "NONE"} : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
    %12 = "tfl.logistic"(%11) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %13 = tfl.mul %arg4, %12 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %14 = "tfl.relu"(%10#1) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %15 = "tfl.logistic"(%10#0) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %16 = tfl.mul %15, %14 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %17 = tfl.add %13, %16 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %18 = "tfl.relu"(%17) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %19 = "tfl.logistic"(%10#3) : (tensor<4x2xf32>) -> tensor<4x2xf32>
    %20 = tfl.mul %18, %19 {fused_activation_function = "NONE"} : tensor<4x2xf32>
    %21 = "tfl.fill"(%cst_11, %cst_7) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
    %22 = "tfl.concatenation"(%cst_6, %21) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %23 = "tfl.reshape"(%arg2, %cst_12) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %24 = "tfl.fill"(%cst_11, %cst_9) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
    %25 = "tfl.concatenation"(%23, %24) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %26 = "tfl.slice"(%arg3, %22, %25) : (tensor<*xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
    %27 = "tfl.reshape"(%5, %cst_12) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
    %28 = "tfl.concatenation"(%27, %21) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %29 = "tfl.concatenation"(%cst_8, %24) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
    %30 = "tfl.slice"(%arg3, %28, %29) : (tensor<*xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>
    %31 = "tfl.expand_dims"(%20, %cst_7) : (tensor<4x2xf32>, tensor<i32>) -> tensor<*xf32>
    %32 = "tfl.concatenation"(%26, %31, %30) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "tfl.yield"(%6, %5, %32, %17, %20, %0) : (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x4x3xf32>) -> ()
  }) {is_stateless = true} : (tensor<i32>, tensor<i32>, tensor<4x4x2xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<4x4x3xf32>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>)
  %2 = "tfl.shape"(%1#2) : (tensor<*xf32>) -> tensor<?xi32>
  %3 = "tfl.reshape"(%1#2, %2) : (tensor<*xf32>, tensor<?xi32>) -> tensor<?x4x2xf32>
  %4 = "tfl.transpose"(%3, %cst_1) : (tensor<?x4x2xf32>, tensor<3xi32>) -> tensor<4x?x2xf32>
  return %4 : tensor<4x?x2xf32>
}

// CHECK-LABEL:   func @rnn(
// CHECK:           tfl.while
// CHECK:             tfl.yield
// CHECK-SAME:  (tensor<i1>) -> ()
// CHECK:             [[VAL_30:%.*]]:7 =
// CHECK: call @tfl.while_body
// CHECK:             tfl.yield
// CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<4x4x3xf32>) -> ()

// CHECK-LABEL:   func private @tfl.while_cond(
// CHECK-SAME:                         [[VAL_35:%.*]]: tensor<i32>, [[VAL_36:%.*]]: tensor<i32>, [[VAL_37:%.*]]: tensor<*xf32>, [[VAL_38:%.*]]: tensor<4x2xf32>, [[VAL_39:%.*]]: tensor<4x2xf32>, [[VAL_40:%.*]]: tensor<*xf32>, [[VAL_41:%.*]]: tensor<4x4x3xf32>) -> tensor<i1> {
// CHECK:           return
// CHECK-SAME:        tensor<i1>
// CHECK:         }

// CHECK-LABEL:   func private @tfl.while_body(
// CHECK-SAME:                         [[VAL_46:%.*]]: tensor<i32>, [[VAL_47:%.*]]: tensor<i32>, [[VAL_48:%.*]]: tensor<*xf32>, [[VAL_49:%.*]]: tensor<4x2xf32>, [[VAL_50:%.*]]: tensor<4x2xf32>, [[VAL_51:%.*]]: tensor<*xf32>, [[VAL_52:%.*]]: tensor<4x4x3xf32>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<4x4x3xf32>) {
// CHECK:           [[VAL_91:%.*]] = "tfl.cast"
// CHECK:           return
// CHECK-SAME:       [[VAL_91]], [[VAL_52]] : tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf32>, tensor<*xf32>, tensor<4x4x3xf32>
// CHECK:         }
// CHECK:       }

// -----

// CHECK-LABEL: func @whileDifferentResultShapes
func @whileDifferentResultShapes(%arg0: tensor<i32>) -> tensor<?xf32>
    attributes {tf.entry_function = {outputs = "result"}} {
  %cst0 = arith.constant dense<5> : tensor<i32> loc("N")
  %cst1 = arith.constant dense<3.0> : tensor<1xf32> loc("val")

  %0:2 = "tfl.while"(%cst0, %cst1) ( {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      %cst_0 = arith.constant dense<0> : tensor<i32>
      %1 = "tfl.greater"(%arg2, %cst_0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
  },  {
    ^bb0(%arg2: tensor<*xi32>, %arg3: tensor<*xf32>):
      %1 = "tfl.sub"(%arg2, %arg0) {fused_activation_function = "NONE"} :
        (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %arg3, %arg3 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
  }) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<?xf32>) loc("WhileOp")

  // CHECK: (tensor<i32>, tensor<1xf32>, tensor<i32>) -> (tensor<i32>, tensor<?xf32>, tensor<i32>)
  return %0#1 : tensor<?xf32>
}

// -----

func @unsupportedCast(%arg0: tensor<4x4x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<0.000000e+00> : tensor<4x2xf32>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<4x4x3xf64>
  %cst_1 = arith.constant dense<[1, 0, 2]> : tensor<3xi32>
  %cst_2 = arith.constant dense<0.000000e+00> : tensor<4x4x2xf32>
  %cst_3 = arith.constant dense<4> : tensor<i32>
  %cst_4 = arith.constant dense<0> : tensor<i32>
  %cst_5 = arith.constant dense<0.000000e+00> : tensor<4x2xf64>
  %0 = "tfl.transpose"(%arg0, %cst_1) : (tensor<4x4x3xf32>, tensor<3xi32>) -> tensor<4x4x3xf32>
  %1:6 = "tfl.while"(%cst_4, %cst_4, %cst_2, %cst, %cst_5, %cst_0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf64>, %arg6: tensor<*xf64>):  // no predecessors
    %5 = "tfl.less"(%arg2, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = "tfl.less"(%arg1, %cst_3) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %7 = tfl.logical_and %6, %5 : tensor<i1>
    "tfl.yield"(%7) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<*xf32>, %arg4: tensor<4x2xf32>, %arg5: tensor<4x2xf64>, %arg6: tensor<*xf64>):  // no predecessors
    "tfl.yield"(%arg1, %arg2, %arg3, %arg4, %arg5, %cst_0) : (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf64>, tensor<4x4x3xf64>) -> ()
  }) {is_stateless = true} : (tensor<i32>, tensor<i32>, tensor<4x4x2xf32>, tensor<4x2xf32>, tensor<4x2xf64>, tensor<4x4x3xf64>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf64>, tensor<*xf32>)
  return %1#2 : tensor<*xf32>
}

// CHECK-LABEL:  func @unsupportedCast(

// CHECK-LABEL:  func private @tfl.while_body(
// CHECK-SAME:     %arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<*xf32>, %arg3: tensor<4x2xf32>, %arg4: tensor<4x2xf64>, %arg5: tensor<*xf64>) -> (tensor<i32>, tensor<i32>, tensor<*xf32>, tensor<4x2xf32>, tensor<4x2xf64>, tensor<*xf64>)
// CHECK:           [[VAL:%.*]] = "tf.Cast"
