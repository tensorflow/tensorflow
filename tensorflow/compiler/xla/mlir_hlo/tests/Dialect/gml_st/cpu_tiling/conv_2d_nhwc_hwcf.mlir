// RUN: mlir-hlo-opt %s --split-input-file \
// RUN: --xla-cpu-transform-conv | FileCheck %s

func.func @conv(%input: tensor<1x41x140x1xf32>,
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
// CHECK-LABEL: @conv
// CHECK-NOT:   scf.for
