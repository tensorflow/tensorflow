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

func.func @foldDivRHSSplat() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf32>
  %1 = stablehlo.constant dense<2.0> : tensor<2xf32>
  %2 = stablehlo.divide %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: foldDivRHSSplat
// CHECK: stablehlo.constant dense<[2.000000e+00, 3.000000e+00]> : tensor<2xf32>

// -----

func.func @foldDivBothSplat() -> tensor<2xf32> {
  %0 = stablehlo.constant dense<4.0> : tensor<2xf32>
  %1 = stablehlo.constant dense<2.0> : tensor<2xf32>
  %2 = stablehlo.divide %0, %1 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: foldDivBothSplat
// CHECK: stablehlo.constant dense<2.000000e+00> : tensor<2xf32>

// -----

func.func @foldDivF64() -> tensor<2xf64> {
  %0 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf64>
  %1 = stablehlo.constant dense<[4.0, 6.0]> : tensor<2xf64>
  %2 = stablehlo.divide %0, %1 : tensor<2xf64>
  return %2 : tensor<2xf64>
}

// CHECK-LABEL: foldDivF64
// CHECK: stablehlo.constant dense<5.000000e-01> : tensor<2xf64>

// -----

func.func @foldDivI32() -> tensor<2xi32> {
  %0 = stablehlo.constant dense<[9, 3]> : tensor<2xi32>
  %1 = stablehlo.constant dense<[4, 6]> : tensor<2xi32>
  %2 = stablehlo.divide %0, %1 : tensor<2xi32>
  return %2 : tensor<2xi32>
}

// CHECK-LABEL: foldDivI32
// CHECK: stablehlo.constant dense<[2, 0]> : tensor<2xi32>

// -----

func.func @divideToMulReciprocalSplat(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<2.0> : tensor<2xf32>
  %2 = stablehlo.divide %arg0, %0 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: divideToMulReciprocalSplat
// CHECK: stablehlo.constant dense<5.000000e-01> : tensor<2xf32>
// CHECK: stablehlo.multiply

// -----

func.func @divideToMulReciprocal(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = stablehlo.constant dense<[2.0, 3.0]> : tensor<2xf32>
  %2 = stablehlo.divide %arg0, %0 : tensor<2xf32>
  return %2 : tensor<2xf32>
}

// CHECK-LABEL: divideToMulReciprocal
// CHECK: stablehlo.constant dense<[5.000000e-01, 0.333333343]> : tensor<2xf32>
// CHECK: stablehlo.multiply

