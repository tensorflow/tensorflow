// RUN: tf-opt --split-input-file --pass-pipeline='builtin.module(tflite-lower-global-tensors)' %s | FileCheck %s

module {
  // CHECK: ml_program.global private mutable @Variable(dense<1.000000e+00> : tensor<16x16xf32>)
  // CHECK-LABEL: func.func @state
  func.func @state(%arg0: tensor<16x16xf32>) -> () {
    "tfl.call_once"() {session_init_function = "StateInit"} : () -> ()
    return
  }

  func.func private @StateInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

module {
  // CHECK: ml_program.global private mutable @Variable(dense<1.000000e+00> : tensor<16x16xf32>)

  // CHECK-LABEL: func.func @assign
  func.func @assign(%arg0: tensor<16x16xf32>) -> () {
    "tfl.call_once"() {session_init_function = "AssignInit"} : () -> ()
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: ml_program.global_store @Variable = %arg0
    "tfl.assign_variable"(%0, %arg0) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }

  func.func private @AssignInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

module {
  // CHECK: ml_program.global private mutable @Variable(dense<1.000000e+00> : tensor<16x16xf32>)

  // CHECK-LABEL: func.func @read
  func.func @read(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "ReadInit"} : () -> ()

    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: %[[LOAD:.+]] = ml_program.global_load @Variable : tensor<16x16xf32>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
    return %1 : tensor<16x16xf32>
  }

  func.func private @ReadInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<1.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

module {
  // CHECK: ml_program.global private mutable @Variable(dense<2.000000e+00> : tensor<16x16xf32>)

  // CHECK-LABEL: func.func @readAssign
  func.func @readAssign(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "ReadAssignInit"} : () -> ()
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: %[[LOAD:.+]] = ml_program.global_load @Variable : tensor<16x16xf32>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>

    // CHECK: %[[ADD:.+]] = tfl.add %[[LOAD]], %arg0
    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>

    // CHECK: ml_program.global_store  @Variable = %[[ADD]]
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func.func private @ReadAssignInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {value = dense<2.000000e+00> : tensor<16x16xf32>} : () -> tensor<16x16xf32>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return
  }
}

// -----

module {
  // CHECK: ml_program.global private mutable @Variable(dense<42> : tensor<2x3xi8>)
  // CHECK-LABEL: func.func @readAssignQuant
  func.func @readAssignQuant(%arg0: tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) -> (tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) {
    "tfl.call_once"() {session_init_function = "ReadAssignInit"} : () -> ()
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: %[[ADDR:.+]] = ml_program.global_load @Variable : tensor<2x3xi8>
    // CHECK: %[[CAST:.+]] = builtin.unrealized_conversion_cast %[[ADDR]] : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>

    // CHECK: %[[ADD:.+]] = tfl.add %[[CAST]], %arg0 {fused_activation_function = "NONE"}
    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>

    // CHECK: %[[CAST2:.+]] = builtin.unrealized_conversion_cast %[[ADD]] : tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>> to tensor<2x3xi8>
    // CHECK: ml_program.global_store @Variable = %[[CAST2]]
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) -> ()
    return %2 : tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>
  }
  func.func private @ReadAssignInit() {
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
    %1 = "tfl.pseudo_const"() {qtype = tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>, value = dense<42> : tensor<2x3xi8>} : () -> tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>
    "tfl.assign_variable"(%0, %1) : (tensor<*x!tf_type.resource>, tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) -> ()
    return
  }
}

// -----

module {
  // CHECK-label: @nostate
  func.func @nostate(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
    "tfl.call_once"() {session_init_function = "NoStateInit"} : () -> ()
    // CHECK: tfl.var_handle
    %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>

    // CHECK: tfl.read_variable
    %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>

    %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>

    // CHECK: tfl.assign_variable
    "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    return %2 : tensor<16x16xf32>
  }
  func.func private @NoStateInit() {
    return
  }
}

