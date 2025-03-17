module {
  func.func @main(%x: tensor<2xf32>, %y: tensor<2xf32>) -> tensor<2xf32> {
    %out = "tfl.custom"(%x, %y) {custom_code = "DISPATCH_OP", custom_option = #tfl<const_bytes: "">} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %out : tensor<2xf32>
  }
}
