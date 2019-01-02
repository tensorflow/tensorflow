// RUN: mlir-opt %s -vectorizer-test -forward-slicing=true | FileCheck %s --check-prefix=FWD
// RUN: mlir-opt %s -vectorizer-test -backward-slicing=true | FileCheck %s --check-prefix=BWD
// RUN: mlir-opt %s -vectorizer-test -slicing=true | FileCheck %s --check-prefix=FWDBWD

///   1       2      3      4
///   |_______|      |______|
///   |   |             |
///   |   5             6
///   |___|_____________|
///     |               |
///     7               8
///     |_______________|
///             |
///             9
func @slicing_test() {
  // Fake 0 to align on 1 and match ASCII art.
  %0 = alloc() : memref<1xi32>

  // FWD: matched: %1 {{.*}} forward static slice:
  // FWD-NEXT: %5 {{.*}} (i32, i32) -> i32
  // FWD-DAG: %8 {{.*}} (i32, i32) -> i32
  // FWD-DAG: %7 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %1 {{.*}} backward static slice:
  //
  // FWDBWD: matched: %1 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %1 = "slicing-test-op" () : () -> i32

  // FWD-NEXT: matched: %2 {{.*}} forward static slice:
  // FWD-NEXT: %5 {{.*}} (i32, i32) -> i32
  // FWD-DAG: %8 {{.*}} (i32, i32) -> i32
  // FWD-DAG: %7 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %2 {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %2 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %2 = "slicing-test-op" () : () -> i32

  // FWD-NEXT: matched: %3 {{.*}} forward static slice:
  // FWD-NEXT: %6 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %8 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %3 {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %3 {{.*}} static slice:
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-NEXT: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-NEXT: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %3 = "slicing-test-op" () : () -> i32

  // FWD-NEXT: matched: %4 {{.*}} forward static slice:
  // FWD-NEXT: %6 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %8 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %4 {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %4 {{.*}} static slice:
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-NEXT: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-NEXT: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %4 = "slicing-test-op" () : () -> i32

  // FWD-NEXT: matched: %5 {{.*}} forward static slice:
  // FWD-DAG: %7 {{.*}} (i32, i32) -> i32
  // FWD-DAG: %8 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %5 {{.*}} backward static slice:
  // BWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %2 = "slicing-test-op"() : () -> i32
  //
  // FWDBWD-NEXT: matched: %5 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %5 = "slicing-test-op" (%1, %2) : (i32, i32) -> i32

  // FWD-NEXT: matched: %6 {{.*}} forward static slice:
  // FWD-NEXT: %8 {{.*}} (i32, i32) -> i32
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %6 {{.*}} backward static slice:
  // BWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %4 = "slicing-test-op"() : () -> i32
  //
  // FWDBWD-NEXT: matched: %6 {{.*}} static slice:
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-NEXT: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-NEXT: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %6 = "slicing-test-op" (%3, %4) : (i32, i32) -> i32

  // FWD-NEXT: matched: %7 {{.*}} forward static slice:
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %7 {{.*}} backward static slice:
  // BWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // BWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  //
  // FWDBWD-NEXT: matched: %7 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32
  
  %7 = "slicing-test-op" (%1, %5) : (i32, i32) -> i32

  // FWD-NEXT: matched: %8 {{.*}} forward static slice:
  // FWD-NEXT: %9 {{.*}} (i32, i32) -> i32
  //
  // BWD: matched: %8 {{.*}} backward static slice:
  // BWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // BWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // BWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // BWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  //
  // FWDBWD-NEXT: matched: %8 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %8 = "slicing-test-op" (%5, %6) : (i32, i32) -> i32

  // FWD-NEXT: matched: %9 {{.*}} forward static slice:
  //
  // BWD: matched: %9 {{.*}} backward static slice:
  // BWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // BWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // BWD-NEXT: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // BWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // BWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // BWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // BWD-NEXT: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  //
  // FWDBWD-NEXT: matched: %9 {{.*}} static slice:
  // FWDBWD-DAG: %4 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %3 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %6 = "slicing-test-op"(%3, %4) : (i32, i32) -> i32
  // FWDBWD-DAG: %2 = "slicing-test-op"() : () -> i32
  // FWDBWD-DAG: %1 = "slicing-test-op"() : () -> i32
  // FWDBWD-NEXT: %5 = "slicing-test-op"(%1, %2) : (i32, i32) -> i32
  // FWDBWD-DAG: %8 = "slicing-test-op"(%5, %6) : (i32, i32) -> i32
  // FWDBWD-DAG: %7 = "slicing-test-op"(%1, %5) : (i32, i32) -> i32
  // FWDBWD-NEXT: %9 = "slicing-test-op"(%7, %8) : (i32, i32) -> i32

  %9 = "slicing-test-op" (%7, %8) : (i32, i32) -> i32

  return
}
