// RUN: litert-opt %s -tf-shape-inference -verify-diagnostics | FileCheck %s
!tf_variant = tensor<!tf_type.variant<tensor<*x!tf_type.variant>>>

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {
  func.func private @quant_fn(%arg0: tensor<*x!quant.uniform<u8:f32, 0.007:128>>) -> () {
    func.return
  }

  // CHECK-LABEL: unppack_const_quant() -> tensor<!quant.uniform<u8:f32, 7.000000e-03:128>>
  func.func @unppack_const_quant() -> (tensor<*x!quant.uniform<u8:f32, 0.007:128>>) {
    %cst = arith.constant dense<5> : tensor<2xi8>
    %0 = "quant.scast"(%cst) : (tensor<2xi8>) -> tensor<2x!quant.uniform<u8:f32, 0.007:128>>
    // CHECK: (tensor<*x!quant.uniform<u8:f32, 7.000000e-03:128>>, tensor<!quant.uniform<u8:f32, 7.000000e-03:128>>)
    %1:2 = "tfl.unpack"(%0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x!quant.uniform<u8:f32, 0.007:128>>) -> (tensor<*x!quant.uniform<u8:f32, 0.007:128>>, tensor<*x!quant.uniform<u8:f32, 0.007:128>>)
    func.call @quant_fn(%1#0) : (tensor<*x!quant.uniform<u8:f32, 0.007:128>>) -> ()

    func.return %1#1 : tensor<*x!quant.uniform<u8:f32, 0.007:128>>
  }
}