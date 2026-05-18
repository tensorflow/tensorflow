// RUN: tf-tfrt-opt -ifrt-pack-inputs-orchestrator %s | FileCheck %s

// CHECK-LABEL:   func.func @serving_default(%arg0: tensor<1x4xf32>, %arg1: tensor<1000xf32>, %arg2: tensor<1x100xi32>)
// CHECK-NEXT:    %0 = "tf.IfrtCall"(%arg0, %arg1, %arg2)
// CHECK-SAME:      __ifrt_pack_group_ids = [0, -1, 0]
// CHECK-SAME:      __ifrt_pack_offsets = [0, 0, 16]
// CHECK-SAME:      program_id = 42 : i64
// CHECK-NEXT:    return %0 : tensor<1x1xf32>

// CHECK-LABEL:   func.func @tpu_program_42(
// CHECK-SAME:      %arg0: tensor<1000xf32> {tf.device = "tpu"},
// CHECK-SAME:      %arg1: tensor<416xi8>)
// CHECK-DAG:       stablehlo.slice %arg1 [0:16]
// CHECK-DAG:       stablehlo.reshape
// CHECK-DAG:       stablehlo.bitcast_convert
// CHECK-DAG:       stablehlo.slice %arg1 [16:416]
// CHECK-DAG:       stablehlo.reshape
// CHECK-DAG:       stablehlo.bitcast_convert
// CHECK-DAG:       stablehlo.add

module {
  func.func @serving_default(%arg0: tensor<1x4xf32>, %arg1: tensor<1000xf32>, %arg2: tensor<1x100xi32>) -> tensor<1x1xf32> {
    %result = "tf.IfrtCall"(%arg0, %arg1, %arg2) <{operandSegmentSizes = array<i32: 3, 0>, program_id = 42 : i64, variable_arg_indices = [], __ifrt_pack_group_ids = [0, -1, 0]}> : (tensor<1x4xf32>, tensor<1000xf32>, tensor<1x100xi32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }

  func.func @tpu_program_42(%arg0: tensor<1x4xf32> {tf.device = "tpu"}, %arg1: tensor<1000xf32> {tf.device = "tpu"}, %arg2: tensor<1x100xi32> {tf.device = "tpu"}) -> tensor<1x1xf32> attributes {tfrt_ifrt_serving.program_id = 42 : i64} {
    %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %1 = "stablehlo.add"(%arg2, %arg2) : (tensor<1x100xi32>, tensor<1x100xi32>) -> tensor<1x100xi32>
    %result = "stablehlo.constant"() {value = dense<0.0> : tensor<1x1xf32>} : () -> tensor<1x1xf32>
    return %result : tensor<1x1xf32>
  }
}
