// RUN: tf-tfrt-opt -tfrt-deduplicate-if-result %s | FileCheck %s -dump-input=fail

func.func private @then(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %x, %x : tensor<i32>, tensor<i32>
}

func.func private @else(%x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  func.return %y, %y : tensor<i32>, tensor<i32>
}

// CHECK-LABEL: then/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<i32>

// CHECK-LABEL: else/tfrt_dedup_results
// CHECK: return {{%.*}} : tensor<i32>

// CHECK-LABEL: @main
func.func @main(%cond: tensor<i1>, %x: tensor<i32>, %y: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
  // CHECK: [[r:%.*]] = "tf.If"
  // CHECK: return [[r]], [[r]] : tensor<i32>, tensor<i32>
  %0, %1 = "tf.If"(%cond, %x, %y) {else_branch = @else, then_branch = @then, is_stateless = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
  return %0, %1 : tensor<i32>, tensor<i32>
}
