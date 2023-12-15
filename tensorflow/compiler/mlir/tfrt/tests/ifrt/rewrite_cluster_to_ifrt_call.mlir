// RUN: tf-tfrt-opt -split-input-file -rewrite-cluster-to-ifrt-call %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @serving_default(%arg0: tensor<3x1xf32>, %arg1: tensor<1x3xf32>) -> tensor<1x1xf32> {
// CHECK-NEXT:  %0 = "tf.IfrtCall"(%arg1, %arg0) 
// CHECK-SAME:       {program_id = [[PROGRAM_ID:.*]] : i64, variable_names = []} 
// CHECK-SAME:       (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    %1 = "tf.Identity"(%arg1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:    %2 = "tf.IfrtCall"(%1, %arg0) 
// CHECK-SAME:       {program_id = [[PROGRAM_ID]] : i64, variable_names = []} 
// CHECK-SAME:       (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
// CHECK-NEXT:    %3 = "tf.add"(%0, %2) : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
// CHECK:    return
//
// CHECK:  func.func private @_ifrt_program__func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> tensor<1x1xf32> 
// CHECK-SAME:      attributes {tfrt_ifrt_serving.program_id = [[PROGRAM_ID]] : i64
// CHECK-NEXT:     %0 = "tf.MatMul"(%arg0, %arg1)
// CHECK:          return

func.func @serving_default(%arg0: tensor<3x1xf32>,  %arg1: tensor<1x3xf32>) -> (tensor<1x1xf32>) {
  %outputs  =  "tf.TPUCompilationResult"() {_tpu_compilation_status = "cluster", device = ""} : () -> tensor<!tf_type.string>
  %outputs_0 = "tf_device.cluster_func"(%arg1, %arg0) {_producer_name = "UNKNOWN", func = @_func } : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %duplicate_arg =  "tf.Identity"(%arg1) {device = ""} : (tensor<1x3xf32>) -> tensor<1x3xf32>
  %outputs_1 = "tf_device.cluster_func"(%duplicate_arg, %arg0) {_producer_name = "UNKNOWN", func = @_func } : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  %outputs_2 = "tf.add"(%outputs_0, %outputs_1): (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
  return %outputs_2 : tensor<1x1xf32>
}

// CHECK-LABEL: @_func
func.func private @_func(%arg0: tensor<1x3xf32>, %arg1: tensor<3x1xf32>) -> (tensor<1x1xf32>) {
  %outputs_0 =  "tf.MatMul"(%arg0, %arg1) {transpose_a = false, transpose_b = false} : (tensor<1x3xf32>, tensor<3x1xf32>) -> tensor<1x1xf32>
  return %outputs_0 : tensor<1x1xf32>
}