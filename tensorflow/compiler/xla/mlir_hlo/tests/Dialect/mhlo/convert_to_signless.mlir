// RUN: mlir-hlo-opt %s --convert-to-signless --canonicalize | FileCheck %s

func.func @Uint16ToInt16(%arg0: memref<*xui16>) -> memref<ui16> {
  // CHECK-NOT: unrealized_conversion_cast
  // CHECK: %[[CAST:.*]] = memref.cast %arg0 : memref<*xi16> to memref<i16>
  // CHECK: return %[[CAST]] : memref<i16>
  %1 = builtin.unrealized_conversion_cast %arg0 : memref<*xui16> to memref<*xi16>
  %2 = memref.cast %1 : memref<*xi16> to memref<i16>
  %3 = builtin.unrealized_conversion_cast %2 : memref<i16> to memref<ui16>
  %4 = bufferization.to_tensor %3 : memref<ui16>
  %5 = builtin.unrealized_conversion_cast %4 : tensor<ui16> to tensor<i16>
  %6 = bufferization.to_memref %5 : memref<i16>
  %7 = builtin.unrealized_conversion_cast %6 : memref<i16> to memref<ui16>
  func.return %7 : memref<ui16>
}
