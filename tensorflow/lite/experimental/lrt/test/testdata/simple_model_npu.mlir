module {
  func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %out = "tfl.custom"(%x, %y) {custom_code = "dispatch_node", custom_option = #tfl<const_bytes: "npu_bytecode">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %out : tensor<2xf32>
  }
}
