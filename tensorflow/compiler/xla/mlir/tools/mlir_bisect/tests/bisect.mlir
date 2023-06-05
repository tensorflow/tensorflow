// RUN: mlir-bisect %s \
// RUN: --pass-pipeline="builtin.module(test-break-linalg-transpose)" \
// RUN: --max-steps-per-run=200 \
// RUN: | FileCheck %s

func.func @main() -> (memref<2x2xindex>, memref<2x2xindex>) {
  %a = memref.alloc() : memref<2x2xindex>
  %b = memref.alloc() : memref<2x2xindex>
  %c = memref.alloc() : memref<2x2xindex>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c2 step %c1 {
      memref.store %i, %a[%i, %j] : memref<2x2xindex>
      memref.store %j, %b[%i, %j] : memref<2x2xindex>
    }
  }

  %i = scf.while: () -> (index) {
    %value = memref.load %a[%c0, %c0] : memref<2x2xindex>
    %cond = arith.cmpi slt, %value, %c3 : index
    scf.condition(%cond) %value : index
  } do {
  ^bb0(%_: index):
    %value = memref.load %a[%c0, %c0] : memref<2x2xindex>
    %add = arith.addi %value, %c1 : index
    memref.store %add, %a[%c0, %c0] : memref<2x2xindex>
    linalg.transpose ins(%b : memref<2x2xindex>) outs(%c : memref<2x2xindex>)
      permutation = [1, 0]
    memref.copy %c, %b : memref<2x2xindex> to memref<2x2xindex>
    scf.yield
  }

  return %a, %b : memref<2x2xindex>, memref<2x2xindex>
}

//     CHECK: Final module
//     CHECK: func @main() -> memref<2x2xindex> {
// CHECK-NOT: scf.while
// CHECK-NOT: scf.for
//     CHECK: linalg.transpose {{.*}} permutation = [1, 0]

//     CHECK: Final module after running pipeline
//     CHECK: linalg.transpose {{.*}} permutation = [0, 1]
