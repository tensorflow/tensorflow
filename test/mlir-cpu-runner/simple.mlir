// RUN: mlir-cpu-runner %s | FileCheck %s
// RUN: mlir-cpu-runner -e foo -init-value 1000 %s | FileCheck -check-prefix=NOMAIN %s

func @fabsf(f32) -> f32

func @main(%a : memref<2xf32>, %b : memref<1xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %0 = constant -420.0 : f32
  %1 = load %a[%c0] : memref<2xf32>
  %2 = load %a[%c1] : memref<2xf32>
  %3 = addf %0, %1 : f32
  %4 = addf %3, %2 : f32
  %5 = call @fabsf(%4) : (f32) -> f32
  store %5, %b[%c0] : memref<1xf32>
  return
}
// CHECK: 0.000000e+00 0.000000e+00
// CHECK-NEXT: 4.200000e+02

func @foo(%a : memref<1x1xf32>) -> memref<1x1xf32> {
  %c0 = constant 0 : index
  %0 = constant 1234.0 : f32
  %1 = load %a[%c0, %c0] : memref<1x1xf32>
  %2 = addf %1, %0 : f32
  store %2, %a[%c0, %c0] : memref<1x1xf32>
  return %a : memref<1x1xf32>
}
// NOMAIN: 2.234000e+03
// NOMAIN-NEXT: 2.234000e+03