// RUN: tf-opt -canonicalize=test-convergence -tfl-runtime-verify -split-input-file -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @squeeze_folder
func.func @squeeze_folder(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "tfl.squeeze"(%arg0) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK: return %arg0
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @squeeze_folder
func.func @squeeze_folder(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  %0 = "tfl.squeeze"(%arg0) : (tensor<?x?xf32>) -> tensor<*xf32>
  // CHECK: "tfl.squeeze"
  func.return %0 : tensor<*xf32>
}

// -----

// Checks that tfl.reshape shape operand is converted to a vector if it is possible
func.func @reshape_vector_shape(tensor<4x4x4xf32>) -> tensor<16x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = arith.constant dense<[[16, 4]]> : tensor<1x2xi32>
  // expected-error @+1 {{'tfl.reshape' op requires 'shape' to be rank 1, but got 2}}
  %1 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<1x2xi32>) -> tensor<16x4xf32>
  func.return %1 : tensor<16x4xf32>
}

// -----

// Checks that tfl.reshape should be removed if its output's only user is
// another tfl.reshape
func.func @reshape_removeAdjacent(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = arith.constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = arith.constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  func.return %1 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacent
// CHECK:  %[[CST:.*]] = arith.constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return %[[RESHAPE]]
}

// Checks that tfl.reshape should be removed if its output has more than one
// user but all users are tfl.reshape
func.func @reshape_removeAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = arith.constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = arith.constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %2 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  %3 = arith.addf %1, %2 : tensor<64xf32>
  func.return %3 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacentWithMultipleUse
// CHECK:  %[[CST:.*]] = arith.constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE_1:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %[[RESHAPE_2:.*]]  = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  %[[RESULT:.*]] = arith.addf %[[RESHAPE_1]], %[[RESHAPE_2]]
// CHECK:  return %[[RESULT]]
}

// Checks that tfl.reshape should be kept if its output has more than one
// user and not all users are tfl.reshape
func.func @reshape_keepAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> (tensor<16x4xf32>, tensor<64xf32>) {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %shape0 = arith.constant dense<[16, 4]> : tensor<2xi32>
  %shape1 = arith.constant dense<[64]> : tensor<1xi32>
  %0 = "tfl.reshape"(%arg0, %shape0) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0, %shape1) : (tensor<16x4xf32>, tensor<1xi32>) -> tensor<64xf32>
  func.return %0, %1 : tensor<16x4xf32>, tensor<64xf32>

// CHECK-LABEL: func @reshape_keepAdjacentWithMultipleUse
// CHECK-DAG:  %[[CST:.*]]  = arith.constant dense<[16, 4]> : tensor<2xi32>
// CHECK-DAG:  %[[CST_0:.*]]  = arith.constant dense<64> : tensor<1xi32>
// CHECK:  %[[RESHAPE_1:.*]] = "tfl.reshape"(%arg0, %[[CST]]) : (tensor<4x4x4xf32>, tensor<2xi32>) -> tensor<16x4xf32>
// CHECK:  %[[RESHAPE_2:.*]] = "tfl.reshape"(%arg0, %[[CST_0]]) : (tensor<4x4x4xf32>, tensor<1xi32>) -> tensor<64xf32>
// CHECK:  return  %[[RESHAPE_1]],  %[[RESHAPE_2]]
}

// Checks that tfl.reshape should be removed if its output type is the same
// as its input type and both are static.
func.func @reshape_removeIdentity(tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %cst = arith.constant dense<[4, 4, 4]> : tensor<3xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<3xi32>) -> tensor<4x4x4xf32>
  func.return %0 : tensor<4x4x4xf32>

// CHECK-LABEL: func @reshape_removeIdentity
// CHECK:  return %arg0 : tensor<4x4x4xf32>
}

// Checks that tfl.reshape shouldn't be removed if either output type or input
// type are dynamic.
func.func @reshape_not_removeIdentity(%arg0: tensor<?xf32>, %arg1: tensor<3xi32>) -> tensor<?x?x?xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>

// CHECK-LABEL: func @reshape_not_removeIdentity
// CHECK-NEXT: "tfl.reshape"
}

// -----

// CHECK-LABEL: @RemoveRedundantUnpackPack
func.func @RemoveRedundantUnpackPack(%arg0: tensor<2x5xf32>) -> tensor<2x5xf32> {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  func.return %1: tensor<2x5xf32>
  // CHECK-NOT: pack
  // CHECK: return %arg0 : tensor<2x5xf32>
}

// -----

// CHECK-LABEL: @RemoveRedundantPack
func.func @RemoveRedundantPack(%arg0: tensor<2x5xf32>) -> (tensor<2x5xf32>, tensor<5xf32>) {
  %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x5xf32>) -> (tensor<5xf32>, tensor<5xf32>)
  %1 = "tfl.pack"(%0#0, %0#1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<5xf32>, tensor<5xf32>) -> (tensor<2x5xf32>)
  func.return %1, %0#0: tensor<2x5xf32>, tensor<5xf32>
  // CHECK: %[[UNPACK:.*]]:2 = "tfl.unpack"
  // CHECK-NOT: pack
  // CHECK: return %arg0, %[[UNPACK]]#0 : tensor<2x5xf32>, tensor<5xf32>
}

// -----

// CHECK-LABEL: @ReplacePackWithReshape
func.func @ReplacePackWithReshape(%arg0: tensor<5xf32>) -> tensor<1x5xf32> {
  %1 = "tfl.pack"(%arg0) {axis = 0 : i32, values_count = 1 : i32} : (tensor<5xf32>) -> (tensor<1x5xf32>)
  // CHECK: reshape
  // CHECK-NOT: pack
  func.return %1: tensor<1x5xf32>
}

// -----

func.func @Int64SliceBeginSize(%arg0: tensor<4x128x32xf32>) -> tensor<1x128x32xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0> : tensor<3xi64>} : () -> tensor<3xi64>
  %1 = "tfl.pseudo_const"() {value = dense<[1, 128, 32]> : tensor<3xi64>} : () -> tensor<3xi64>
  %2 = "tfl.slice"(%arg0, %0, %1) : (tensor<4x128x32xf32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x128x32xf32>
  func.return %2 : tensor<1x128x32xf32>

// CHECK-DAG:  [[VAL_1:%.*]] = arith.constant dense<0> : tensor<3xi32>
// CHECK-DAG:  [[VAL_2:%.*]] = arith.constant dense<[1, 128, 32]> : tensor<3xi32>
// CHECK:  [[VAL_3:%.*]] = "tfl.slice"(%arg0, [[VAL_1]], [[VAL_2]]) : (tensor<4x128x32xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<1x128x32xf32>
}

// -----

// CHECK-LABEL: @WhileCanonicalizeBug
// Make sure that second output of the tf.while is not incorrectly inferred as
// pass through just because the corresponding input is not used in either
// condition or body. The tensor<f32> result of the loop can be either %arg1
// (if the body never executes, or 22.0 if the body executes at least once).
func.func @WhileCanonicalizeBug(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>):
    %limit = arith.constant dense<100> : tensor<i32>
    %test = "tfl.less"(%arg0, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%test) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<f32>):
    %cst = arith.constant dense<22.0> : tensor<f32>
    %stride = arith.constant dense<1> : tensor<i32>
    %inc = tfl.add %arg2, %stride {fused_activation_function = "NONE"} : tensor<i32>
    "tfl.yield"(%inc, %cst) : (tensor<i32>, tensor<f32>) -> ()
  }) : (tensor<i32>, tensor<f32>) -> (tensor<i32>, tensor<f32>)
  // CHECK: return %0#1 : tensor<f32>
  func.return %0#1 : tensor<f32>
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
// CHECK-LABEL: @WhileCanonicalizeBug1
func.func @WhileCanonicalizeBug1(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0:2 = "tfl.while"(%arg0, %arg1) ({
  ^bb0(%carg0: tensor<f32>, %carg1: tensor<f32>):
    %limit = arith.constant dense<100> : tensor<i32>
    %test = "tfl.less"(%limit, %limit) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%test) : (tensor<i1>) -> ()
  },  {
  ^bb0(%barg0: tensor<f32>, %barg1: tensor<f32>):
    %cst = arith.constant dense<22.0> : tensor<f32>
    "tfl.yield"(%barg0, %cst) : (tensor<f32>, tensor<f32>) -> ()
  }) : (tensor<f32>, tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#1 : tensor<f32>
}

// -----

// Test case to test While op with resources that are not read-only variables.
// Do not remove resource arugments if they are not read-only variables to keep
// the graph's control dependency.
// CHECK-LABEL: WhileWithNonReadOnlyVariableResources
func.func @WhileWithNonReadOnlyVariableResources(%arg0: tensor<i32>) -> tensor<!tf_type.resource> {
  %0 = "tf.Const"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %1 = "tf.Const"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %4 = "tf.StackV2"(%3) {elem_type = f32, stack_name = "s"} : (tensor<i32>) -> tensor<!tf_type.resource>
  %5:5 = "tfl.while"(%2, %3, %2, %4, %0) ({
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<!tf_type.resource>, %arg5: tensor<f32>):
    %9 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Less"(%arg3, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "tfl.yield"(%10) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>, %arg4: tensor<!tf_type.resource>, %arg5: tensor<f32>):
    %9 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %10 = "tf.Cast"(%arg3) {Truncate = false, device = ""} : (tensor<i32>) -> tensor<f32>
    %11 = "tf.AddV2"(%arg3, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %12 = "tf.StackPushV2"(%arg4, %10) {device = "", swap_memory = false} : (tensor<!tf_type.resource>, tensor<f32>) -> tensor<f32>
    %13 = "tf.AddV2"(%arg1, %9) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "tfl.yield"(%13, %arg2, %11, %arg4, %12) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.resource>, tensor<f32>) -> ()
  }) {is_stateless = false} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.resource>, tensor<f32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.resource>, tensor<f32>)
  func.return %5#3 : tensor<!tf_type.resource>

// CHECK: "tfl.while"
// CHECK: (tensor<i32>, tensor<i32>, tensor<!tf_type.resource>) -> (tensor<i32>, tensor<i32>, tensor<!tf_type.resource>)
}

// CHECK-LABEL: @RemoveFcZeroBias
func.func @RemoveFcZeroBias(%arg0: tensor<1x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<1x40xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<0.0> : tensor<40xf32>} : () -> tensor<40xf32>
  %1 = "tfl.fully_connected"(%arg0, %arg1, %0) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<1x37xf32>, tensor<40x37xf32>, tensor<40xf32>) -> tensor<1x40xf32>
// CHECK: "tfl.fully_connected"
// CHECK-SAME: (tensor<1x37xf32>, tensor<40x37xf32>, none) -> tensor<1x40xf32>
  func.return %1 : tensor<1x40xf32>
}

// CHECK-LABEL: forceAsymmetricQuantizeInput
func.func @forceAsymmetricQuantizeInput(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %cst0 = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>
  %cst1 = arith.constant dense<2.0> : tensor<2xf32>

  %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %0 : tensor<4x2xf32>
  // CHECK %0 = "tfl.fully_connected"(%arg0, %cst0, %cst1) {asymmetric_quantize_inputs = true, fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  // CHECK return %0
}

// CHECK-LABEL: RemoveLstmQuantZeroBias
func.func @RemoveLstmQuantZeroBias(
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
  %cst = "tfl.no_value"() {value = unit} : () -> none
  %zero = "tfl.pseudo_const"() {value = dense<0.0> : tensor<640xf32>} : () -> tensor<640xf32>
  %0 = "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %cst, %cst, %cst, %arg9, %arg10, %arg11, %arg12, %arg13, %zero, %arg19, %arg20, %arg15, %arg16, %arg17, %arg18) ({}) {
     cell_clip = 1.000000e+01 : f32, fused_activation_function = "TANH", kernel_type = #tfl<lstm_kernel_type_attr FULL>, proj_clip = 0.01 : f32
  } : (tensor<1x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x528xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, tensor<2048x640xf32>, none, none, none, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<640x2048xf32>, tensor<640xf32>, tensor<1x640xf32>, tensor<1x2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>) -> tensor<1x640xf32>
    func.return %0 : tensor<1x640xf32>
// CHECK: %[[NONE:.+]] = "tfl.no_value"() <{value}> : () -> none
// CHECK: "tfl.lstm"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %[[NONE]], %[[NONE]], %[[NONE]], %arg9, %arg10, %arg11, %arg12, %arg13, %[[NONE]], %arg19, %arg20, %arg15, %arg16, %arg17, %arg18)
}

func.func @keepCustomFlexOps(%arg0: tensor<1x10xf32>) -> tensor<1x10xf32> {
  %0 = "tfl.custom"() {custom_code = "FlexVarHandleOp", custom_option = #tfl<const_bytes : "0x0B56617248616E646C654F700074120B56617248616E646C654F702A190A0B7368617265645F6E616D65120A12085661726961626C652A0F0A09636F6E7461696E6572120212002A0B0A056474797065120230012A150A0F616C6C6F7765645F6465766963657312020A002A130A057368617065120A3A08120208011202080A3200000283771414042801">} : () -> tensor<!tf_type.resource<tensor<1x10xf32>>>
  %1 = "tfl.custom"(%0) {custom_code = "FlexReadVariableOp", custom_option = #tfl<const_bytes : "0x0E526561645661726961626C654F700021120E526561645661726961626C654F701A002A0B0A056474797065120230013200000233241414042801">} : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
  %2 = "tfl.custom"(%1, %arg0) {custom_code = "FlexAddV2", custom_option = #tfl<const_bytes : "0x0541646456320016120541646456321A001A002A070A015412023001320000021F191414042801">} : (tensor<1x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
  "tfl.custom"(%0, %2) {custom_code = "FlexAssignVariableOp", custom_option = #tfl<const_bytes : "0x1041737369676E5661726961626C654F70003B121041737369676E5661726961626C654F701A001A002A0B0A056474797065120230012A140A0E76616C69646174655F736861706512022800320000024F3E1414042801">} : (tensor<!tf_type.resource<tensor<1x10xf32>>>, tensor<1x10xf32>) -> ()
  %3 = "tfl.custom"(%0) {custom_code = "FlexReadVariableOp", custom_option = #tfl<const_bytes : "0x0E526561645661726961626C654F700021120E526561645661726961626C654F701A002A0B0A056474797065120230013200000233241414042801">} : (tensor<!tf_type.resource<tensor<1x10xf32>>>) -> tensor<1x10xf32>
  // CHECK:      %0 = "tfl.custom"() <{custom_code = "FlexVarHandleOp"
  // CHECK-NEXT: %1 = "tfl.custom"(%0) <{custom_code = "FlexReadVariableOp"
  // CHECK-NEXT: %2 = "tfl.custom"(%1, %arg0) <{custom_code = "FlexAddV2"
  // CHECK-NEXT: "tfl.custom"(%0, %2) <{custom_code = "FlexAssignVariableOp"
  // CHECK-NEXT: %3 = "tfl.custom"(%0) <{custom_code = "FlexReadVariableOp"
  func.return %3 : tensor<1x10xf32>
}

// -----

// Converts tfl.broadcast_to to tfl.reshape if input and output have the same
// number of elements.
// CHECK-LABEL: broadcast_to_to_reshape
func.func @broadcast_to_to_reshape(%arg0: tensor<4x4x4xf32>, %arg1 : tensor<4xi32>) -> tensor<1x4x4x4xf32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<4x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x4xf32>
  // CHECK: "tfl.reshape"
  // CHECK-SAME: (tensor<4x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x4xf32>
  func.return %0 : tensor<1x4x4x4xf32>
}

// Converts tfl.broadcast_to to tfl.reshape if input and output have the same
// number of elements.
// CHECK-LABEL: broadcast_to_to_reshape_i64
func.func @broadcast_to_to_reshape_i64(%arg0: tensor<4x4x4xf32>, %arg1 : tensor<4xi64>) -> tensor<1x4x4x4xf32> {
  %0 = "tfl.broadcast_to"(%arg0, %arg1) : (tensor<4x4x4xf32>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  // CHECK: "tfl.cast"
  // CHECK-SAME: (tensor<4xi64>) -> tensor<4xi32>
  // CHECK-NEXT: "tfl.reshape"
  // CHECK-SAME: (tensor<4x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x4xf32>
  func.return %0 : tensor<1x4x4x4xf32>
}


// Converts tfl.broadcast_to to tfl.reshape if input and output have the same
// number of elements.
// CHECK-LABEL: broadcast_to_to_reshape_i64_const
func.func @broadcast_to_to_reshape_i64_const(%arg0: tensor<4x4x4xf32>) -> tensor<1x4x4x4xf32> {
  %cst = arith.constant dense<[1, 4, 4, 4]> : tensor<4xi64>
  %0 = "tfl.broadcast_to"(%arg0, %cst) : (tensor<4x4x4xf32>, tensor<4xi64>) -> tensor<1x4x4x4xf32>
  // CHECK: arith.constant dense<[1, 4, 4, 4]> : tensor<4xi32>
  // CHECK-NEXT: "tfl.reshape"
  // CHECK-SAME: (tensor<4x4x4xf32>, tensor<4xi32>) -> tensor<1x4x4x4xf32>
  func.return %0 : tensor<1x4x4x4xf32>
}

// -----

func.func @trivial_dynamic_update_slice(%arg0: tensor<2x7x14xf32>, %arg1: tensor<2x7x14xf32>) -> tensor<2x7x14xf32> {
  %0 = arith.constant dense<0> : tensor<3xi32>
  %1 = "tfl.dynamic_update_slice"(%arg0, %arg1, %0) : (tensor<2x7x14xf32>, tensor<2x7x14xf32>, tensor<3xi32>) -> tensor<2x7x14xf32>
  // CHECK: return %arg1
  func.return %1 : tensor<2x7x14xf32>
}

// -----

func.func @trivial_dynamic_update_slice_wrong_update_shape(%arg0: tensor<2x7x14xf32>, %arg1: tensor<2x7x7xf32>) -> tensor<2x7x14xf32> {
  %0 = arith.constant dense<0> : tensor<3xi32>
  %1 = "tfl.dynamic_update_slice"(%arg0, %arg1, %0) : (tensor<2x7x14xf32>, tensor<2x7x7xf32>, tensor<3xi32>) -> tensor<2x7x14xf32>
  // CHECK: "tfl.dynamic_update_slice"
  func.return %1 : tensor<2x7x14xf32>
}

// CHECK-LABEL: OptimizeTranposeWithRank7orMoreEffectiveRank6
func.func @OptimizeTranposeWithRank7orMoreEffectiveRank6(%arg0: tensor<7x6x5x4x3x2x1xf32> ) -> (tensor<1x2x3x4x5x6x7xf32>)  {
  %cst = arith.constant dense<[6, 5, 4, 3, 2, 1, 0]> : tensor<7xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<7x6x5x4x3x2x1xf32>, tensor<7xi32>) -> tensor<1x2x3x4x5x6x7xf32>
  return %0 : tensor<1x2x3x4x5x6x7xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[7, 6, 5, 4, 3, 2]> : tensor<6xi32>
  // CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<[5, 4, 3, 2, 1, 0]> : tensor<6xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[1, 2, 3, 4, 5, 6, 7]> : tensor<7xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %[[cst]]) : (tensor<7x6x5x4x3x2x1xf32>, tensor<6xi32>) -> tensor<7x6x5x4x3x2xf32>
  // CHECK: %1 = "tfl.transpose"(%0, %[[cst_0]]) : (tensor<7x6x5x4x3x2xf32>, tensor<6xi32>) -> tensor<2x3x4x5x6x7xf32>
  // CHECK: %2 = "tfl.reshape"(%1, %[[cst_1]]) : (tensor<2x3x4x5x6x7xf32>, tensor<7xi32>) -> tensor<1x2x3x4x5x6x7xf32>
  // CHECK: return %2
}

// CHECK-LABEL: OptimizeTranposeWithRank7orMoreEffectiveRank4
func.func @OptimizeTranposeWithRank7orMoreEffectiveRank4(%arg0: tensor<56x8x56x1x1x1x7xf32> ) -> (tensor<1x1x8x56x56x7x1xf32>)  {
  %cst = arith.constant dense<[4, 5, 1, 2, 0, 6, 3]> : tensor<7xi32>
  %0 = "tfl.transpose"(%arg0, %cst) : (tensor<56x8x56x1x1x1x7xf32>, tensor<7xi32>) -> tensor<1x1x8x56x56x7x1xf32>
  return %0 : tensor<1x1x8x56x56x7x1xf32>
  // CHECK-DAG: %[[cst:.*]] = arith.constant dense<[56, 8, 56, 7]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_0:.*]] = arith.constant dense<[1, 2, 0, 3]> : tensor<4xi32>
  // CHECK-DAG: %[[cst_1:.*]] = arith.constant dense<[1, 1, 8, 56, 56, 7, 1]> : tensor<7xi32>
  // CHECK: %0 = "tfl.reshape"(%arg0, %[[cst]]) : (tensor<56x8x56x1x1x1x7xf32>, tensor<4xi32>) -> tensor<56x8x56x7xf32>
  // CHECK: %1 = "tfl.transpose"(%0, %[[cst_0]]) : (tensor<56x8x56x7xf32>, tensor<4xi32>) -> tensor<8x56x56x7xf32>
  // CHECK: %2 = "tfl.reshape"(%1, %[[cst_1]]) : (tensor<8x56x56x7xf32>, tensor<7xi32>) -> tensor<1x1x8x56x56x7x1xf32>
  // CHECK: return %2
}

// CHECK-LABEL: @ConstPadToI32
func.func @ConstPadToI32(%arg0: tensor<15600xf32>) -> tensor<15602xf32> {
  %0 = "tfl.pseudo_const"() {value = dense<1> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %1 = "tfl.pad"(%arg0, %0) : (tensor<15600xf32>, tensor<1x2xi64>) -> tensor<15602xf32>
  func.return %1 : tensor<15602xf32>
  // CHECK: "tfl.pad"(%arg0, %cst) : (tensor<15600xf32>, tensor<1x2xi32>) -> tensor<15602xf32>
}


