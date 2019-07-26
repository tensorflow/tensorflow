// RUN: mlir-opt %s -affine-vectorizer-test -forward-slicing=true 2>&1 | FileCheck %s --check-prefix=FWD
// RUN: mlir-opt %s -affine-vectorizer-test -backward-slicing=true 2>&1 | FileCheck %s --check-prefix=BWD
// RUN: mlir-opt %s -affine-vectorizer-test -slicing=true 2>&1 | FileCheck %s --check-prefix=FWDBWD

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
// FWD-LABEL: slicing_test
// BWD-LABEL: slicing_test
// FWDBWD-LABEL: slicing_test
func @slicing_test() {
  // Fake 0 to align on 1 and match ASCII art.
  %0 = alloc() : memref<1xi32>

  // FWD: matched: %[[v1:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v5:.*]] {{.*}} -> i5
  // FWD-DAG: %[[v8:.*]] {{.*}} -> i8
  // FWD-DAG: %[[v7:.*]] {{.*}} -> i7
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v1:.*]] {{.*}} backward static slice:
  //
  // FWDBWD: matched: %[[v1:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %1 = "slicing-test-op" () : () -> i1

  // FWD-NEXT: matched: %[[v2:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v5:.*]] {{.*}} -> i5
  // FWD-DAG: %[[v8:.*]] {{.*}} -> i8
  // FWD-DAG: %[[v7:.*]] {{.*}} -> i7
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v2:.*]] {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %[[v2:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %2 = "slicing-test-op" () : () -> i2

  // FWD-NEXT: matched: %[[v3:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v6:.*]] {{.*}} -> i6
  // FWD-NEXT: %[[v8:.*]] {{.*}} -> i8
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v3:.*]] {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %[[v3:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-NEXT: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-NEXT: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %3 = "slicing-test-op" () : () -> i3

  // FWD-NEXT: matched: %[[v4:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v6:.*]] {{.*}} -> i6
  // FWD-NEXT: %[[v8:.*]] {{.*}} -> i8
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v4:.*]] {{.*}} backward static slice:
  //
  // FWDBWD-NEXT: matched: %[[v4:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-NEXT: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-NEXT: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %4 = "slicing-test-op" () : () -> i4

  // FWD-NEXT: matched: %[[v5:.*]] {{.*}} forward static slice:
  // FWD-DAG: %[[v7:.*]] {{.*}} -> i7
  // FWD-DAG: %[[v8:.*]] {{.*}} -> i8
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v5:.*]] {{.*}} backward static slice:
  // BWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // BWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  //
  // FWDBWD-NEXT: matched: %[[v5:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %5 = "slicing-test-op" (%1, %2) : (i1, i2) -> i5

  // FWD-NEXT: matched: %[[v6:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v8:.*]] {{.*}} -> i8
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v6:.*]] {{.*}} backward static slice:
  // BWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // BWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  //
  // FWDBWD-NEXT: matched: %[[v6:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-NEXT: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-NEXT: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %6 = "slicing-test-op" (%3, %4) : (i3, i4) -> i6

  // FWD-NEXT: matched: %[[v7:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v7:.*]] {{.*}} backward static slice:
  // BWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // BWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // BWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  //
  // FWDBWD-NEXT: matched: %[[v7:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %7 = "slicing-test-op" (%1, %5) : (i1, i5) -> i7

  // FWD-NEXT: matched: %[[v8:.*]] {{.*}} forward static slice:
  // FWD-NEXT: %[[v9:.*]] {{.*}} -> i9
  //
  // BWD: matched: %[[v8:.*]] {{.*}} backward static slice:
  // BWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // BWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // BWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // BWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // BWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // BWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  //
  // FWDBWD-NEXT: matched: %[[v8:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %8 = "slicing-test-op" (%5, %6) : (i5, i6) -> i8

  // FWD-NEXT: matched: %[[v9:.*]] {{.*}} forward static slice:
  //
  // BWD: matched: %[[v9:.*]] {{.*}} backward static slice:
  // BWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // BWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // BWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // BWD-NEXT: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // BWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // BWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // BWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // BWD-NEXT: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  //
  // FWDBWD-NEXT: matched: %[[v9:.*]] {{.*}} static slice:
  // FWDBWD-DAG: %[[v4:.*]] = "slicing-test-op"() : () -> i4
  // FWDBWD-DAG: %[[v3:.*]] = "slicing-test-op"() : () -> i3
  // FWDBWD-NEXT: %[[v6:.*]] = "slicing-test-op"(%[[v3]], %[[v4]]) : (i3, i4) -> i6
  // FWDBWD-DAG: %[[v2:.*]] = "slicing-test-op"() : () -> i2
  // FWDBWD-DAG: %[[v1:.*]] = "slicing-test-op"() : () -> i1
  // FWDBWD-NEXT: %[[v5:.*]] = "slicing-test-op"(%[[v1]], %[[v2]]) : (i1, i2) -> i5
  // FWDBWD-DAG: %[[v8:.*]] = "slicing-test-op"(%[[v5]], %[[v6]]) : (i5, i6) -> i8
  // FWDBWD-DAG: %[[v7:.*]] = "slicing-test-op"(%[[v1]], %[[v5]]) : (i1, i5) -> i7
  // FWDBWD-NEXT: %[[v9:.*]] = "slicing-test-op"(%[[v7]], %[[v8]]) : (i7, i8) -> i9

  %9 = "slicing-test-op" (%7, %8) : (i7, i8) -> i9

  return
}

// FWD-LABEL: slicing_test_2
// BWD-LABEL: slicing_test_2
// FWDBWD-LABEL: slicing_test_2
func @slicing_test_2() {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c16 = constant 16 : index
  loop.for %i0 = %c0 to %c16 step %c1 {
    affine.for %i1 = (i)[] -> (i)(%i0) to 10 {
      // BWD: matched: %[[b:.*]] {{.*}} backward static slice:
      // BWD: loop.for {{.*}}

      // affine.for appears in the body of loop.for
      // BWD: affine.for {{.*}}

      // affine.for appears as a proper op in the backward slice
      // BWD: affine.for {{.*}}
      %b = "slicing-test-op"(%i1): (index) -> index

      // BWD: matched: %[[c:.*]] {{.*}} backward static slice:
      // BWD: loop.for {{.*}}

      // affine.for appears in the body of loop.for
      // BWD-NEXT: affine.for {{.*}}

      // affine.for only appears in the body of loop.for
      // BWD-NOT: affine.for {{.*}}
      %c = "slicing-test-op"(%i0): (index) -> index
    }
  }
  return
}

// FWD-LABEL: slicing_test_3
// BWD-LABEL: slicing_test_3
// FWDBWD-LABEL: slicing_test_3
func @slicing_test_3() {
  %f = constant 1.0 : f32
  %c = "slicing-test-op"(%f): (f32) -> index
  // FWD: matched: {{.*}} (f32) -> index forward static slice:
  // FWD: loop.for {{.*}}
  // FWD: matched: {{.*}} (index, index) -> index forward static slice:
  loop.for %i2 = %c to %c step %c {
    %d = "slicing-test-op"(%c, %i2): (index, index) -> index
  }
  return
}

// FWD-LABEL: slicing_test_function_argument
// BWD-LABEL: slicing_test_function_argument
// FWDBWD-LABEL: slicing_test_function_argument
func @slicing_test_function_argument(%arg0: index) -> index {
  // BWD: matched: {{.*}} (index, index) -> index backward static slice:
  %0 = "slicing-test-op"(%arg0, %arg0): (index, index) -> index
  return %0 : index
}

// This test dumps 2 sets of outputs: first the test outputs themselves followed
// by the module. These labels isolate the test outputs from the module dump.
// FWD-LABEL: slicing_test
// BWD-LABEL: slicing_test
// FWDBWD-LABEL: slicing_test
// FWD-LABEL: slicing_test_2
// BWD-LABEL: slicing_test_2
// FWDBWD-LABEL: slicing_test_2
// FWD-LABEL: slicing_test_3
// BWD-LABEL: slicing_test_3
// FWDBWD-LABEL: slicing_test_3
// FWD-LABEL: slicing_test_function_argument
// BWD-LABEL: slicing_test_function_argument
// FWDBWD-LABEL: slicing_test_function_argument
