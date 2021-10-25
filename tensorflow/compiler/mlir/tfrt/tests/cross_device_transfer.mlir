// RUN: tf-tfrt-opt -tfrt-cross-device-transfer %s | FileCheck %s

// CHECK-LABEL: test_transfer_op_result
func @test_transfer_op_result(%arg0: !tfrt.chain) -> () {
  // CHECK-NEXT: %[[RESULT_0:.*]] = corert.get_op_handler %[[ARG_0:.*]] "cpu"
  %0 = corert.get_op_handler %arg0 "cpu"
  // CHECK-NEXT: %[[RESULT_1:.*]] = corert.get_op_handler %[[ARG_0]] "gpu"
  %1 = corert.get_op_handler %arg0 "gpu"
  // CHECK-NEXT: %[[RESULT_2:.*]] = corert.create_dense_tensor.i32 {shape = [0], value = []}
  %2 = corert.create_dense_tensor.i32 {shape = [0], value = []}
  // CHECK-NEXT: %[[RESULT_3:.*]] = corert.executeop(%[[RESULT_0]]) "tf.AddV2"(%[[RESULT_2]], %[[RESULT_2]])
  %3 = corert.executeop(%0) "tf.AddV2"(%2, %2) {T = f32, device = "/device:CPU:0"} : 1
  // CHECK-NEXT: %[[RESULT_4:.*]] = tfrt.get_device %[[ARG_0]] {device_name = "/device:GPU:0"}
  // CHECK-NEXT: %[[RESULT_5:.*]] = corert.get_dst_tensor_type %[[RESULT_3]], %[[RESULT_4]]
  // CHECK-NEXT: %[[RESULT_6:.*]] = corert.transfer %[[RESULT_3]], %[[RESULT_4]], %[[RESULT_5]]
  // CHECK-NEXT: %[[RESULT_7:.*]] = corert.executeop(%[[RESULT_1]]) "tf.AddV2"(%[[RESULT_6]], %[[RESULT_6]])
  %4 = corert.executeop(%1) "tf.AddV2"(%3, %3) {T = f32, device = "/device:GPU:0"} : 1
  tfrt.return
}

// CHECK: func @test_transfer_func_arg(%[[ARG_0:.*]]: !tfrt.chain, %[[ARG_1:.*]]: !corert.tensorhandle
func @test_transfer_func_arg(%arg0: !tfrt.chain, %arg1: !corert.tensorhandle {tfrt.device = "/device:CPU:0"}) -> () {
  // CHECK-NEXT: %[[RESULT_0:.*]] = corert.get_op_handler %[[ARG_0]] "gpu"
  %0 = corert.get_op_handler %arg0 "gpu"
  // CHECK-NEXT: %[[RESULT_1:.*]] = tfrt.get_device %[[ARG_0]] {device_name = "/device:GPU:0"}
  // CHECK-NEXT: %[[RESULT_2:.*]] = corert.get_dst_tensor_type %[[ARG_1]], %[[RESULT_1]]
  // CHECK-NEXT: %[[RESULT_3:.*]] = corert.transfer %[[ARG_1]], %[[RESULT_1]], %[[RESULT_2]]
  // CHECK-NEXT: %[[RESULT_4:.*]] = corert.executeop(%[[RESULT_0]]) "tf.AddV2"(%[[RESULT_3]], %[[RESULT_3]])
  %1 = corert.executeop(%0) "tf.AddV2"(%arg1, %arg1) {T = f32, device = "/device:GPU:0"} : 1
  tfrt.return
}
