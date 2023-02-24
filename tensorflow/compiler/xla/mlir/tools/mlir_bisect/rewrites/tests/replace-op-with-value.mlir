// RUN: mlir-bisect %s --debug-strategy=ReplaceOpWithValue | FileCheck %s

func.func @main() -> (memref<i32>, memref<i32>) {
  %a = memref.alloc() : memref<i32>
  %b = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : i32
  memref.store %c0, %b[] : memref<i32>
  return %a, %b : memref<i32>, memref<i32>
}

//      CHECK: func @main()
//      CHECK:   %[[ALLOC:.*]] = memref.alloc()
// CHECK-NEXT:   memref.alloc
// CHECK-NEXT:   constant
// CHECK-NEXT:   memref.store {{.*}}, %[[ALLOC]]
// CHECK-NEXT:   return %[[ALLOC]], %[[ALLOC]]
