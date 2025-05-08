// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -with-layouts -print-layouts %s | FileCheck %s
// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text -with-layouts -print-layouts --via-builder=true %s | FileCheck %s

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
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
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4xf32, #COO>) -> tensor<3x4xf32, #COO> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4]{1,0:D(C,S)} parameter(0)
  return %arg : tensor<3x4xf32, #COO>
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
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
  map = (d0, d1, d2) -> (d0 : compressed(nonunique), d1 : singleton(nonunique, nonordered), d2 : singleton(nonordered)),
  posWidth = 32,
  crdWidth = 32
}>

// CHECK:  HloModule
func.func @main(%arg: tensor<3x4x5xf32, #UnorderedCOOTensor>) -> tensor<3x4x5xf32, #UnorderedCOOTensor> {
  // CHECK: ROOT %[[ARG0:.*]] = f32[3,4,5]{2,1,0:D(C,S,S)} parameter(0)
  return %arg : tensor<3x4x5xf32, #UnorderedCOOTensor>
}

