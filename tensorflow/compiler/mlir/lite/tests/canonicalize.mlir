// RUN: tf-opt -pass-pipeline='func(canonicalize)' -split-input-file -verify-diagnostics %s | FileCheck %s

// Checks that tfl.reshape shape operand is converted to a vector if it is possible
func @reshape_vector_shape(tensor<4x4x4xf32>) -> tensor<16x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[[16, 4]]> : tensor<1x2xi32>
  // expected-error @+1 {{'tfl.reshape' op requires 'shape' to be rank 1, but got 2}}
  %1 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<1x2xi32>) -> tensor<16x4xf32>
  return %1 : tensor<16x4xf32>
}

// -----

// Checks that tfl.reshape should be removed if its output's only user is
// another tfl.reshape
func @reshape_removeAdjacent(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacent
// CHECK:  %[[CST:.*]] = constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return %[[RESHAPE]]
}

// Checks that tfl.reshape should be removed if its output has more than one
// user but all users are tfl.reshape
func @reshape_removeAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %2 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %3 = addf %1, %2 : tensor<64xf32>
  return %3 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacentWithMultipleUse
// CHECK:  %[[CST:.*]] = constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE_1:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %[[RESHAPE_2:.*]]  = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %[[RESULT:.*]] = addf %[[RESHAPE_1]], %[[RESHAPE_2]]
// CHECK:  return %[[RESULT]]
}

// Checks that tfl.reshape should be kept if its output has more than one
// user and not all users are tfl.reshape
func @reshape_keepAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> (tensor<16x4xf32>, tensor<64xf32>) {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  return %0, %1 : tensor<16x4xf32>, tensor<64xf32>

// CHECK-LABEL: func @reshape_keepAdjacentWithMultipleUse
// CHECK:  %[[CST:.*]]  = constant dense<[16, 4]> : tensor<2xi32>
// CHECK:  %[[CST_0:.*]]  = constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE_1:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
// CHECK:  %[[RESHAPE_2:.*]] = "tfl.reshape"(%arg0, %[[CST_0]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return  %[[RESHAPE_1]],  %[[RESHAPE_2]]
}

// Checks that tfl.reshape should be removed if its output type is the same
// as its input type and both are static.
func @reshape_removeIdentity(tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %cst = constant dense<[4, 4, 4]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<3xi32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>

// CHECK-LABEL: func @reshape_removeIdentity
// CHECK:  return %arg0 : tensor<4x4x4xf32>
}

// Checks that tfl.reshape shouldn't be removed if either output type or input
// type are dynamic.
func @reshape_not_removeIdentity(%arg0: tensor<?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>

// CHECK-LABEL: func @reshape_not_removeIdentity
// CHECK-NEXT: "tfl.reshape"
}

// -----

// CHECK-LABEL: @RemoveRedundantUnpackPack
func @RemoveRedundantUnpackPack(%arg0: tensor<2x5xf32>) -> tensor<2x5xf32> {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  return %1: tensor<2x5xf32>
  // CHECK-NOT: pack
  // CHECK: return %arg0 : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: @RemoveRedundantPack
func @RemoveRedundantPack(%arg0: tensor<2x5xf32>) -> (tensor<2x5xf32>, tensor<5xf32>) {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  return %1, %0#0: tensor<2x5xf32>, tensor<5xf32>
  // CHECK: %[[UNPACK:.*]]:2 = "tfl.unpack"
  // CHECK-NOT: pack
  // CHECK: return %arg0, %[[UNPACK]]#0 : tensor<2x5xf32>, tensor<5xf32>
}

// -----

func @Int64SliceBeginSize(%arg0: tensor<4x128x32xf32>) -> tensor<1x128x32xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 128, 32]> : tensor<3xi64>} : () -> tensor<3xi64>
  %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<4x128x32xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x128x32xf32>
  return %2 : tensor<1x128x32xf32>

// CHECK:  [[VAL_1:%.*]] = constant dense<0> : tensor<3xi32>
// CHECK:  [[VAL_2:%.*]] = constant dense<[1, 128, 32]> : tensor<3xi32>
// CHECK:  [[VAL_3:%.*]] = "tfl.slice"(%arg0, [[VAL_1]], [[VAL_2]]) : (tensor<4x128x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x128x32xf32>
}

// -----

// CHECK-LABEL: @WhileCanonicalizeBug
// Make sure that second output of the tf.while is not incorrectly inferred as
// pass through just because the corresponding input is not used in either
// condition or body. The tensor<f32> result of the loop can be either %arg1
// (if the body never executes, or 22.0 if the body executes at least once).
func @WhileCanonicalizeBug(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ( {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>):
    %limit = constant dense<100> : tensor<i32>
    %test = "tfl.less"(%arg0, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%test) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>):
    %cst = constant dense<22.0> : tensor<f32>
    %stride = constant dense<1> : tensor<i32>
    %inc = tfl.add %arg2, %stride {fused_activation_function = "NONE"} : tensor<i32>
    "tfl.yield"(%inc, %cst) : (tensor<i32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  // CHECK: return %0#1 : tensor<f32>
  return %0#1 : tensor<f32>
}

// -----

// Test case to test bug due to checking
// `while_op.getResult(arg_index).use_empty()` instead of
// `while_op.getResult(while_index).use_empty()` in the tfl.while
// canonicalization.
// arg0 is a pass through. After first iteration, arg_index = 0
// and while_index = 1. Make arg1 use empty in block and condition, but not in
// result. Canonicalize will think it can remove both slot#0 and slot#1 and do
// so without replacing all operands, and in assert builds it will fail an
// assert failure ( op->use_empty() && "expected 'op' to have no uses")
// CHECK-LABEL: WhileCanonicalizeBug1
func @WhileCanonicalizeBug1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ( {
  ^bb0(%carg0: tensor<f32>, %carg1: tensor<f32>):
    %limit = constant dense<100> : tensor<i32>
    %test = "tfl.less"(%limit, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%test) : (tensor<i1>) -> ()
  },  {
  ^bb0(%barg0: tensor<f32>, %barg1: tensor<f32>):
    %cst = constant dense<22.0> : tensor<f32>
    "tfl.yield"(%barg0, %cst) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  return %0#1 : tensor<f32>
}

// -----

// Test case to test While op with resources that are not read-only variables.
// Do not remove resource arugments if they are not read-only variables to keep
// the graph's control dependency.
// CHECK-LABEL: WhileWithNonReadOnlyVariableResources
func @WhileWithNonReadOnlyVariableResources(%arg0: tensor<i32>) -> tensor<!tf.resource> {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %4 = "tf.StackV2"(%3) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf.resource>
  %5:5 = "tfl.while"(%2, %3, %2, %4, %0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<!tf.resource>, %arg5: tensor<f32>):  // no predecessors
    %9 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Less"(%arg3, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%10) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<!tf.resource>, %arg5: tensor<f32>):  // no predecessors
    %9 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Cast"(%arg3) {Truncate = false, device = ""} : (tensor<i32>) -> tensor<f32>
    %11 = "tf.AddV2"(%arg3, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = "tf.StackPushV2"(%arg4, %10) {device = "", swap_memory = false} : (tensor<!tf.resource>, tensor<f32>) -> tensor<f32>
    %13 = "tf.AddV2"(%arg1, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tfl.yield"(%13, %arg2, %11, %arg4, %12) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<f32>) -> ()
  }) {is_stateless = false} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<f32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf.resource>, tensor<f32>)
  return %5#3 : tensor<!tf.resource>

// CHECK: "tfl.while"
// CHECK: (tensor<i32>, tensor<i32>, tensor<!tf.resource>) -> (tensor<i32>, tensor<i32>, tensor<!tf.resource>)
}

// CHECK-LABEL: @RemoveFcZeroBias
func @RemoveFcZeroBias(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0.0> : tensor<40xf32>} : () -> tensor<40xf32>
  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<1x40xf32>
// CHECK: "tfl.fully_connected"
// CHECK-SAME: (tensor<1x37xf32>, tensor<40x37xf32>, none) -> tensor<1x40xf32>
  return %1 : tensor<1x40xf32>
}

// CHECK-LABEL: RemoveLstmQuantZeroBias
func @RemoveLstmQuantZeroBias(
  %arg0: tensor<1x528xf32>,
  %arg1: tensor<2048x528xf32>,
  %arg2: tensor<2048x528xf32>,
  %arg3: tensor<2048x528xf32>,
  %arg4: tensor<2048x528xf32>,
  %arg5: tensor<2048x640xf32>,
  %arg6: tensor<2048x640xf32>,
  %arg7: tensor<2048x640xf32>,
  %arg8: tensor<2048x640xf32>,
  %arg9: tensor<2048xf32>,
  %arg10: tensor<2048xf32>,
  %arg11: tensor<2048xf32>,
  %arg12: tensor<2048xf32>,
  %arg13: tensor<640x2048xf32>,
  %arg14: tensor<640xf32>,
  %arg15: tensor<2048xf32>,
  %arg16: tensor<2048xf32>,
  %arg17: tensor<2048xf32>,
  %arg18: tensor<2048xf32>,
  %arg19: tensor<1x640xf32>,
  %arg20: tensor<1x2048xf32>
) -> tensor<1x640xf32> {
  %cst = constant unit
  %zero = "tfl.pseudo_const"() {value = dense<0.0> : tensor<640xf32>} : () -> tensor<640xf32>
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %cst, %cst, %cst, %arg9, %arg10, %arg11, %arg12, %arg13, %zero, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({}) {
     cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", kernel_type = "FULL", proj_clip = 0.01 : f32
  } : (tensor<1x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, none, none, none, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<640x2048xf32>, tensor<640xf32>, tensor<1x640xf32>, tensor<1x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x640xf32>
    return %0 : tensor<1x640xf32>
// CHECK: %[[NONE:.+]] = constant unit
// CHECK: "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[NONE]], %[[NONE]], %[[NONE]], %arg9, %arg10, %arg11, %arg12, %arg13, %[[NONE]], %arg19, %arg20, %arg15, %arg16, %arg17, %arg18)
}

