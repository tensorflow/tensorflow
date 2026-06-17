// RUN: tf-tfrt-opt -split-input-file -tf-mlrt-assign-op-key %s | FileCheck %s

// CHECK-LABEL: func @main
// CHECK: tf.AddV2
// CHECK-SAME: {__op_key = 0 : i32}

// CHECK: tf.AddV2
// CHECK-SAME: {__op_key = 1 : i32}

// CHECK: tf.AddV2
// CHECK-SAME: {__op_key = 2 : i32}

// CHECK: tf.AddV2
// CHECK-SAME: {__op_key = 3 : i32}

// CHECK: tf.Sub
// CHECK-SAME: {__op_key = 4 : i32}

// CHECK: tf.Sub
// CHECK-SAME: {__op_key = 5 : i32}

// CHECK: tf.Sub
// CHECK-SAME: {__op_key = 6 : i32}

// CHECK: tf.Sub
// CHECK-SAME: {__op_key = 7 : i32}


// CHECK: [[x:%.*]] = "tf.AddV2"
// CHECK-SAME: {__op_key = 8 : i32}

// CHECK: return [[x]]

func.func @main(%a: tensor<i32>, %b: tensor<i32>) -> tensor<i32> {

  %a0 = "tf.AddV2"(%a, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a1 = "tf.AddV2"(%a0, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a2 = "tf.AddV2"(%a1, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %a3 = "tf.AddV2"(%a2, %a) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %b0 = "tf.Sub"(%b, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b1 = "tf.Sub"(%b0, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b2 = "tf.Sub"(%b1, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %b3 = "tf.Sub"(%b2, %b) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  %c = "tf.AddV2"(%a3, %b3) : (tensor<i32>, tensor<i32>) -> tensor<i32>

  func.return %c : tensor<i32>
}
