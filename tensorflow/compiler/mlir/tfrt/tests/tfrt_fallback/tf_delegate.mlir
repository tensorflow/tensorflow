// RUN: tfrt_translate -mlir-to-bef %s | tf_bef_executor | FileCheck %s

// CHECK: --- Running 'matmul_delegate_test'
func.func @matmul_delegate_test() {
  %c0 = tfrt.new.chain

  // Create 2x2 dht<i32, 2> with value 1
  %dht_a = tfrt_dht.create_uninitialized_tensor.i32.2 [2 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %dht_a, %c0 1 : i32

  // Create 2x2 dht<i32, 2> with value 2
  %dht_b = tfrt_dht.create_uninitialized_tensor.i32.2 [2 : i64, 2 : i64]
  %c2 = tfrt_dht.fill_tensor_with_constant.i32 %dht_b, %c0 2 : i32

  // Convert dht to tf tensor
  %tft_a, %c3 = "tfd.move_dht_to_tft"(%dht_a, %c1)
      : (!tfrt_tensor.tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)
  %tft_b, %c4 = "tfd.move_dht_to_tft"(%dht_b, %c2)
      : (!tfrt_tensor.tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)

  // Print legacy TF tensors
  // CHECK: shape = [2, 2], values = [1, 1, 1, 1]
  %cc0 = "tfd.print_tft"(%tft_a, %c0) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: shape = [2, 2], values = [2, 2, 2, 2]
  %cc1 = "tfd.print_tft"(%tft_b, %cc0) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // Create TF eager context
  %c6 = "tfd.init_eager_context"(%c0): (!tfrt.chain) -> !tfrt.chain

  // Delegate to tf.matmul
  %c7, %tft_x = "tfd.delegate_kernel"(%c6, %tft_a, %tft_b) {op_name = "MatMul"}
      : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // CHECK: shape = [2, 2], values = [4, 4, 4, 4]
  %cc2 = "tfd.print_tft"(%tft_x, %cc1) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // Delegate to tf.matmul by using %tft_x from the previous delegation as input
  %c8, %tft_y = "tfd.delegate_kernel"(%c6, %tft_x, %tft_b) {op_name = "MatMul"}
      : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // Print legacy TF tensors
  // CHECK: shape = [2, 2], values = [16, 16, 16, 16]
  %cc3 = "tfd.print_tft"(%tft_y, %cc2) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // Convert tf tensor back to dht
  %dht_c, %c9 = "tfd.convert_tft_to_dht"(%tft_x, %cc2)
      : (!tfd.tf_tensor, !tfrt.chain) -> (!tfrt_tensor.tensor, !tfrt.chain)
  %dht_d, %c10 = "tfd.convert_tft_to_dht"(%tft_y, %cc3)
      : (!tfd.tf_tensor, !tfrt.chain) -> (!tfrt_tensor.tensor, !tfrt.chain)

  // Print the result dht
  // CHECK: shape = [2, 2], values = [4, 4, 4, 4]
  %cc4 = tfrt_dht.print_tensor %dht_c, %cc3
  // CHECK: shape = [2, 2], values = [16, 16, 16, 16]
  %cc5 = tfrt_dht.print_tensor %dht_d, %cc4

  tfrt.return
}

// CHECK: --- Running 'bad_op_name_test'
func.func @bad_op_name_test() {
  %c0 = tfrt.new.chain

  // Create 2x2 dht<i32, 2> with value 2
  %dht_a = tfrt_dht.create_uninitialized_tensor.i32.2 [2 : i64, 2 : i64]
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %dht_a, %c0 2 : i32

  // Convert dht to tf tensor
  %tft_a, %c2 = "tfd.move_dht_to_tft"(%dht_a, %c1)
      : (!tfrt_tensor.tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)

  // Create TF eager context
  %c3 = "tfd.init_eager_context"(%c0): (!tfrt.chain) -> !tfrt.chain
  // expected-error @+1 {{runtime error: 'BadOp'}}
  %c4, %tft_x = "tfd.delegate_kernel"(%c3, %tft_a) {op_name = "BadOp"}
       : (!tfrt.chain, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  tfrt.return
}

// CHECK: --- Running 'addn_delegate_test'
func.func @addn_delegate_test() {
  %c0 = tfrt.new.chain

  // Create scalar dht<i32, 0> with value 1
  %dht_a = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %c1 = tfrt_dht.fill_tensor_with_constant.i32 %dht_a, %c0 1 : i32

  // Create scalar dht<i32, 0> with value 2
  %dht_b = tfrt_dht.create_uninitialized_tensor.i32.0 []
  %c2 = tfrt_dht.fill_tensor_with_constant.i32 %dht_b, %c0 2 : i32

  // Convert dht to tf tensor
  %tft_a, %c3 = "tfd.move_dht_to_tft"(%dht_a, %c1)
      : (!tfrt_tensor.tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)
  %tft_b, %c4 = "tfd.move_dht_to_tft"(%dht_b, %c2)
      : (!tfrt_tensor.tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)

  // Print legacy TF tensors
  // CHECK: shape = [], values = [1]
  %cc0 = "tfd.print_tft"(%tft_a, %c0) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: shape = [], values = [2]
  %cc1 = "tfd.print_tft"(%tft_b, %cc0) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // Create TF eager context
  %c6 = "tfd.init_eager_context"(%c0): (!tfrt.chain) -> !tfrt.chain

  // Delegate to AddN
  %c7, %tft_x = "tfd.delegate_kernel"(%c6, %tft_a, %tft_b) {_op_name = "AddN", attr0_name = "N", attr0_value = "int$2", attr1_name = "T", attr1_value = "tfdtype$DT_INT32"}
      : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // CHECK: shape = [], values = [3]
  %cc2 = "tfd.print_tft"(%tft_x, %cc1) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  // Delegate to AddN by using %tft_x from the previous delegation as input
  %c8, %tft_y = "tfd.delegate_kernel"(%c6, %tft_x, %tft_b) {_op_name = "AddN", attr0_name = "N", attr0_value = "int$2", attr1_name = "T", attr1_value = "tfdtype$DT_INT32"}
      : (!tfrt.chain, !tfd.tf_tensor, !tfd.tf_tensor) -> (!tfrt.chain, !tfd.tf_tensor)

  // Print legacy TF tensors
  // CHECK: shape = [], values = [5]
  %cc3 = "tfd.print_tft"(%tft_y, %cc2) : (!tfd.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return
}

