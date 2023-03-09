// RUN: mlir-hlo-opt %s --split-input-file --xla-cpu-transform-conv \
// RUN: | FileCheck %s

func.func @conv_is_matmul(%input: tensor<1x41x140x1xf32>,
    %kernel: tensor<1x140x1x128xf32>) -> tensor<1x41x1x128xf32> {
  %empty = tensor.empty() : tensor<1x41x1x128xf32>

  %c0 = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%c0 : f32)
    outs(%empty: tensor<1x41x1x128xf32>) -> tensor<1x41x1x128xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %kernel : tensor<1x41x140x1xf32>, tensor<1x140x1x128xf32>)
    outs(%fill : tensor<1x41x1x128xf32>) -> tensor<1x41x1x128xf32>

  func.return %conv : tensor<1x41x1x128xf32>
}
// CHECK-LABEL: @conv_is_matmul
// CHECK:       scf.for
// CHECK:         linalg.matmul
// CHECK-SAME:      tensor<41x140xf32>, tensor<140x128xf32>
// CHECK-SAME:      tensor<41x128xf32>) -> tensor<41x128xf32>

// -----

func.func @conv_is_matmul_after_tiling(%input: tensor<1x45x140x1xf32>,
    %kernel: tensor<5x140x1x128xf32>) -> tensor<1x41x1x128xf32> {
  %empty = tensor.empty() : tensor<1x41x1x128xf32>

  %c0 = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%c0 : f32)
    outs(%empty: tensor<1x41x1x128xf32>) -> tensor<1x41x1x128xf32>

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%input, %kernel : tensor<1x45x140x1xf32>, tensor<5x140x1x128xf32>)
    outs(%fill : tensor<1x41x1x128xf32>) -> tensor<1x41x1x128xf32>

  func.return %conv : tensor<1x41x1x128xf32>
}
// CHECK-LABEL: @conv_is_matmul_after_tiling
// CHECK:       scf.for
// CHECK:         linalg.matmul
// CHECK-SAME:      tensor<41x140xf32>, tensor<140x128xf32>
// CHECK-SAME:      tensor<41x128xf32>) -> tensor<41x128xf32>
