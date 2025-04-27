// RUN: tf-tosa-opt --split-input-file --tfl-to-tosa-pipeline --verify-each %s | FileCheck %s

// RUN: tf-tosa-opt --split-input-file --tf-tfl-to-tosa-pipeline --verify-each %s | FileCheck %s


// Operations for testing tfl-to-tosa-pipeline

// -----

// CHECK-LABEL: tosa.variable @var_x = dense<7.000000e+00> : tensor<1xf32>
// CHECK-LABEL: test_stateful_ops(
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1xf32>
// CHECK: tosa.variable.write @var_x, %[[VAL_0]] : tensor<1xf32>
// CHECK: %[[VAL_1:.*]] = tosa.variable.read @var_x : tensor<1xf32>
// CHECK: return %[[VAL_1]] : tensor<1xf32>
module attributes {tf_saved_model.semantics, tfl.description = "Test.", tfl.schema_version = 3 : i32} {
    func.func @test_stateful_ops(%arg0: tensor<1xf32> {tf_saved_model.index_path = ["placeholder_0"]})
      -> (tensor<1xf32> {tf_saved_model.index_path = ["output_0"]})
      attributes {tf_saved_model.exported_names = ["serving_default"]} {
        "tfl.call_once"() {session_init_function = "InitializeX"} : () -> ()
        %0 = "tfl.var_handle"() {container = "", shared_name = "var_x"} : () -> tensor<!tf_type.resource>
        "tfl.assign_variable"(%0, %arg0) : (tensor<!tf_type.resource>, tensor<1xf32>) -> ()
        %1 = "tfl.read_variable"(%0) : (tensor<!tf_type.resource>) -> tensor<1xf32>
        return %1 : tensor<1xf32>
    }

    // initialize variable var_x to 7.0
    func.func private @InitializeX() {
        %0 = "tfl.var_handle"() {container = "", shared_name = "var_x"} : () -> tensor<!tf_type.resource>
        %1 = "tfl.pseudo_const"() {value = dense<7.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
        "tfl.assign_variable"(%0, %1) : (tensor<!tf_type.resource>, tensor<1xf32>) -> ()
        return
    }
}

// -----

// CHECK-LABEL: tosa.variable @Variable = dense<42> : tensor<2x3xi8>
// CHECK-LABEL: readAssignQuant
// CHECK-SAME: %[[VAL_0:.*]]: tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>
// CHECK: %[[VAL_1:.*]] = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_2:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_3:.*]] = "tosa.const"() <{values = dense<2> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_4:.*]] = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK: %[[VAL_5:.*]] = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
// CHECK: %[[VAL_6:.*]] = tosa.variable.read @Variable : tensor<2x3xi8>
// CHECK: %[[VAL_7:.*]] = builtin.unrealized_conversion_cast %[[VAL_6]] : tensor<2x3xi8> to tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>
// CHECK: %[[VAL_8:.*]] = tosa.rescale %[[VAL_7]], %[[VAL_5]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<2x3xi32>
// CHECK: %[[VAL_9:.*]] = tosa.rescale %[[VAL_0]], %[[VAL_5]], %[[VAL_4]], %[[VAL_3]], %[[VAL_2]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<2x3xi32>
// CHECK: %[[VAL_10:.*]] = tosa.add %[[VAL_8]], %[[VAL_9]] : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
// CHECK: %[[VAL_11:.*]] = tosa.rescale %[[VAL_10]], %[[VAL_5]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = "DOUBLE_ROUND", scale32 = true} : (tensor<2x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>
// CHECK: %[[VAL_12:.*]] = builtin.unrealized_conversion_cast %[[VAL_11]] : tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>> to tensor<2x3xi8>
// CHECK: tosa.variable.write @Variable, %[[VAL_12]] : tensor<2x3xi8>
// CHECK: return %[[VAL_11]] : tensor<2x3x!quant.uniform<i8:f32, 1.000000e-01:2>>
module {
    func.func @readAssignQuant(%arg0: tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) -> (tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>) {
        "tfl.call_once"() {session_init_function = "ReadAssignInit"} : () -> ()
        %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
        %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>
        %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<2x3x!quant.uniform<i8:f32, 0.1:2>>
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
    // CHECK-LABEL: @nostate
    // CHECK: %[[VAL_0:.*]]: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // CHECK: %[[VAL_1:.*]] = "tfl.var_handle"() <{container = "", shared_name = "Variable"}> : () -> tensor<*x!tf_type.resource>
    // CHECK: %[[VAL_2:.*]] = "tfl.read_variable"(%[[VAL_1]]) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
    // CHECK: %[[VAL_3:.*]] = tosa.add %[[VAL_2]], %[[VAL_0]] : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    // CHECK: "tfl.assign_variable"(%[[VAL_1]], %[[VAL_3]]) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
    // CHECK: return %[[VAL_3]] : tensor<16x16xf32>
    func.func @nostate(%arg0: tensor<16x16xf32>) -> (tensor<16x16xf32>) {
        "tfl.call_once"() {session_init_function = "NoStateInit"} : () -> ()
        %0 = "tfl.var_handle"() {container = "", shared_name = "Variable"} : () -> tensor<*x!tf_type.resource>
        %1 = "tfl.read_variable"(%0) : (tensor<*x!tf_type.resource>) -> tensor<16x16xf32>
        %2 = tfl.add %1, %arg0 {fused_activation_function = "NONE"} : tensor<16x16xf32>
        "tfl.assign_variable"(%0, %2) : (tensor<*x!tf_type.resource>, tensor<16x16xf32>) -> ()
        return %2 : tensor<16x16xf32>
    }
    func.func private @NoStateInit() {
        return
    }
}
