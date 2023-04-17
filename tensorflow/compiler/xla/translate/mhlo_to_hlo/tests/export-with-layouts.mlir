// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -with-layouts -print-layouts %s | FileCheck %s
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text  -with-layouts -print-layouts --via-builder=true %s | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"],
  dimOrdering = affine_map<(i, j) -> (i, j)>,
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32, #CSR>) -> tensor<3x4xf32, #CSR> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4]{1,0:D(D,C)} parameter(0)
  return %arg : tensor<3x4xf32, #CSR>
}

// -----

#COO = #sparse_tensor.encoding<{
  dimLevelType = ["compressed-nu", "singleton"],
  dimOrdering = affine_map<(i, j) -> (i, j)>,
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32, #COO>) -> tensor<3x4xf32, #COO> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4]{1,0:D(C+,S)} parameter(0)
  return %arg : tensor<3x4xf32, #COO>
}

// -----

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"],
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32, #CSR>) -> tensor<3x4xf32, #CSR> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4]{1,0:D(D,C)} parameter(0)
  return %arg : tensor<3x4xf32, #CSR>
}

// -----

#UnorderedCOOTensor = #sparse_tensor.encoding<{
  dimLevelType = ["compressed-nu", "singleton-nu-no", "singleton-no"],
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4x5xf32, #UnorderedCOOTensor>) -> tensor<3x4x5xf32, #UnorderedCOOTensor> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4,5]{2,1,0:D(C+,S+~,S~)} parameter(0)
  return %arg : tensor<3x4x5xf32, #UnorderedCOOTensor>
}

