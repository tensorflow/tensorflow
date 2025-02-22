module {
  func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    %cpu_out = tfl.add %x, %y {fused_activation_function = "NONE"} : tensor<2xf32>
    %npu_out = "tfl.custom"(%x, %y) {custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes: "simple">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %cpu_out, %npu_out : tensor<2xf32>, tensor<2xf32>
  }
}
