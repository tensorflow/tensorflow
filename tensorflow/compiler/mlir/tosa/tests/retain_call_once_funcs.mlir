// RUN: tf-tosa-opt --split-input-file --pass-pipeline='builtin.module(tflite-retain-call-once-funcs)' %s | FileCheck %s


// CHECK-LABEL: module {
module {
  // CHECK-LABEL: @main
  func.func @main(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    // CHECK: "tfl.call_once"() <{session_init_function = "NoOp"}> {session_init_function_symbol = @NoOp} : () -> ()
    "tfl.call_once"() {session_init_function = "NoOp"} : () -> ()
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func.func private @NoOp() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<0.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}
