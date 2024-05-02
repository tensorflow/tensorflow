// RUN: odml-converter --shlo-simplify %s -split-input-file | FileCheck %s

func.func @foldDiv() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf32>
  %2 = stablehlo.divide %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: foldDiv
// CHECK: stablehlo.constant dense<5.000000e-01> : tensor<2xf32>

// -----

func.func @foldDivLHSSplat() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<2.0> : tensor<2xf32>
  %1 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf32>
  %2 = stablehlo.divide %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: foldDivLHSSplat
// CHECK: stablehlo.constant dense<[5.000000e-01, 0.333333343]> : tensor<2xf32>

// -----

func.func @foldDivF64() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf64>
  %1 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf64>
  %2 = stablehlo.divide %0, %1 : tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: foldDivF64
// CHECK: stablehlo.constant dense<5.000000e-01> : tensor<2xf64>


