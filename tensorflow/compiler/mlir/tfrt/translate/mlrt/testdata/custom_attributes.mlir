func.func @add_const_custom_dense_i32_10(%c0: i32) -> i32 {
  %c1 = "test_custom.add.const.i32"(%c0) {value = dense<[0]> : tensor<1xi32>} : (i32) -> i32
  %c2 = "test_custom.add.const.i32"(%c1) {value = dense<[1]> : tensor<1xi32>} : (i32) -> i32
  %c3 = "test_custom.add.const.i32"(%c2) {value = dense<[2]> : tensor<1xi32>} : (i32) -> i32
  %c4 = "test_custom.add.const.i32"(%c3) {value = dense<[3]> : tensor<1xi32>} : (i32) -> i32
  %c5 = "test_custom.add.const.i32"(%c4) {value = dense<[4]> : tensor<1xi32>} : (i32) -> i32
  %c6 = "test_custom.add.const.i32"(%c5) {value = dense<[5]> : tensor<1xi32>} : (i32) -> i32
  %c7 = "test_custom.add.const.i32"(%c6) {value = dense<[6]> : tensor<1xi32>} : (i32) -> i32
  %c8 = "test_custom.add.const.i32"(%c7) {value = dense<[7]> : tensor<1xi32>} : (i32) -> i32
  %c9 = "test_custom.add.const.i32"(%c8) {value = dense<[8]> : tensor<1xi32>} : (i32) -> i32
  %c10 = "test_custom.add.const.i32"(%c9) {value = dense<[9]> : tensor<1xi32>} : (i32) -> i32
  func.return %c10 : i32
}
