// RUN: tf-tfrt-opt -ifrt-pack-inputs-planner %s | FileCheck %s

// CHECK-LABEL: func.func @serving_default(
// CHECK-SAME:   %arg0: tensor<1x4xf32>,
// CHECK-SAME:   %arg1: tensor<1024x1024xf32>,
// CHECK-SAME:   %arg2: tensor<!tf_type.string>,
// CHECK-SAME:   %arg3: tensor<1x100xi32>
// CHECK-NEXT:   %0 = "tf.IfrtCall"(%arg0, %arg1, %arg2, %arg3)
// CHECK-SAME:     __ifrt_pack_group_ids = [0, -1, -1, 0]
module {
  func.func @serving_default(%arg0: tensor<1x4xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<!tf_type.string>, %arg3: tensor<1x100xi32>) -> tensor<1x1xf32> {
    %result = "tf.IfrtCall"(%arg0, %arg1, %arg2, %arg3) <{operandSegmentSizes = array<i32: 4, 0>, program_id = 1001 : i64, variable_arg_indices = [2 : i32]}> : (tensor<1x4xf32>, tensor<1024x1024xf32>, tensor<!tf_type.string>, tensor<1x100xi32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}

// -----
// CHECK-LABEL: func.func @serving_default_async(
// CHECK-SAME:   %arg0: tensor<1x4xf32>
// CHECK-NEXT:   %0 = "tf.AsyncIfrtCall"(%arg0)
// CHECK-SAME:     __ifrt_pack_group_ids = [0]
module {
  func.func @serving_default_async(%arg0: tensor<1x4xf32>) -> tensor<1x1xf32> {
    %result = "tf.AsyncIfrtCall"(%arg0) <{operandSegmentSizes = array<i32: 1, 0>, program_id = 1002 : i64, variable_arg_indices = []}> : (tensor<1x4xf32>) -> (tensor<1x1xf32>)
    return %result : tensor<1x1xf32>
  }
}
