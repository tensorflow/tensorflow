// RUN: mlir-bisect %s --debug-strategy=EraseOpWithoutResults | FileCheck %s

func.func @main() -> memref<i32> {
  %a = arith.constant 1 : i32
  %b = memref.alloc() : memref<i32>
  memref.store %a, %b[] : memref<i32>
  func.return %b : memref<i32>
}

//      CHECK: func.func @main()
//      CHECK:   %[[ALLOC:.*]] = memref.alloc
// CHECK-NEXT:   return %[[ALLOC]]
