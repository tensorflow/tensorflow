// RUN: tf-opt -corert-optimize %s | FileCheck %s

// CHECK-LABEL: func @fold_test
func @fold_test(%arg: !corert.tensorhandle) -> !corert.tensorhandle {
    %cpu = corert.get_device "cpu"
    // CHECK-NOT: tf.Const
    %0 = corert.executeop(%cpu) "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : 1
    // CHECK: "_tf.Transpose"({{%.*}})
    // CHECK-SAME: perm = dense<[0, 3, 1, 2]> : tensor<4xi32>
    %1 = corert.executeop(%cpu) "tf.Transpose"(%arg, %0) {T = f32, Tperm = i32} : 1
    hex.return %1 : !corert.tensorhandle
}
