// RUN: tf-opt %s -pass-pipeline='func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @simplify_double_conversion_test(
func @simplify_double_conversion_test() {
  // CHECK: %[[CREATE:.*]] = dht.create
  // CHECK: %[[FILL:.*]] = dht.fill
  // CHECK: dht.print_tensor %[[CREATE]], %[[FILL]]
  %c0 = hex.new.chain

  // Create 2x2 dht<i32, 2> with value 1
  %dht0 = dht.create_uninitialized_tensor.i32.2 [2 : i32, 2 : i32]
  %c1 = dht.fill_tensor_with_constant.i32 %dht0, %c0 1 : i32

  // Convert dht to tf tensor
  %tft0, %c2 = "tfd.move_dht_to_tft"(%dht0, %c1)
      : (!t.tensor, !hex.chain) -> (!tfd.tf_tensor, !hex.chain)

  // Convert tf tensor back to dht
  %dht1, %c3 = "tfd.convert_tft_to_dht"(%tft0, %c2)
      : (!tfd.tf_tensor, !hex.chain) -> (!t.tensor, !hex.chain)

  // Print the result dht
  %c4 = dht.print_tensor %dht1, %c3

  hex.return
}
