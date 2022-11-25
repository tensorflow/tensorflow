// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that duplicate init operands do not result in duplicate names.

// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  %Init, %ctlInit = Init name("init") : () -> (tensor<*xi32>)
  %While:2, %ctl_1 = WhileRegion(%Init, %Init) {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    %True, %ctlTrue = True : () -> (tensor<*xi1>)
    condition %True : tensor<*xi1> (%arg0, %arg1) : tensor<*xi32>, tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>, %arg2: !tf_type.control, %arg3: !tf_type.control):
    yield(%arg0, %arg1) : tensor<*xi32>, tensor<*xi32>
  } {parallel_iterations = 10 : i64} : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>, tensor<*xi32>)
}

// CHECK-LABEL: tfg.func @while_cond_function
// CHECK-SAME: tfg.name = "init_tfg_result_0"
// CHECK-NEXT: tfg.name = "init_tfg_result_0_0"
// CHECK-NEXT: ->

// CHECK-LABEL: tfg.func @while_body_function
// CHECK-SAME: tfg.name = "init_tfg_result_0"
// CHECK-NEXT: tfg.name = "init_tfg_result_0_0"
// CHECK-NEXT: ->
// CHECK-SAME: tfg.name = "init_tfg_result_0_1"
// CHECK-NEXT: tfg.name = "init_tfg_result_0_2"
