// RUN: mlir-opt %s -affine-loop-unroll-jam -unroll-jam-factor=2 | FileCheck %s
// RUN: mlir-opt %s -affine-loop-unroll-jam -unroll-jam-factor=4 | FileCheck --check-prefix=UJAM-FOUR %s

// CHECK-DAG: [[MAP_PLUS_1:#map[0-9]+]] = (d0) -> (d0 + 1)
// CHECK-DAG: [[MAP_DIV_OFFSET:#map[0-9]+]] = ()[s0] -> (((s0 - 1) floordiv 2) * 2 + 1)
// CHECK-DAG: [[MAP_MULTI_RES:#map[0-9]+]] = ()[s0, s1] -> ((s0 floordiv 2) * 2, (s1 floordiv 2) * 2, 1024)
// CHECK-DAG: [[MAP_SYM_UB:#map[0-9]+]] = ()[s0, s1] -> (s0, s1, 1024)

// UJAM-FOUR-DAG: [[UBMAP:#map[0-9]+]] = ()[s0] -> (s0 + 8)
// UJAM-FOUR-DAG: [[MAP_PLUS_1:#map[0-9]+]] = (d0) -> (d0 + 1)
// UJAM-FOUR-DAG: [[MAP_PLUS_2:#map[0-9]+]] = (d0) -> (d0 + 2)
// UJAM-FOUR-DAG: [[MAP_PLUS_3:#map[0-9]+]] = (d0) -> (d0 + 3)

// CHECK-LABEL: func @unroll_jam_imperfect_nest() {
func @unroll_jam_imperfect_nest() {
  affine.for %i = 0 to 101 {
    %x = "addi32"(%i, %i) : (index, index) -> i32
    affine.for %j = 0 to 17 {
      %y = "addi32"(%i, %i) : (index, index) -> i32
      %z = "addi32"(%y, %y) : (i32, i32) -> i32
    }
    %w = "foo"(%i, %x) : (index, i32) -> i32
  }
  return
}
// CHECK:      affine.for [[IV0:%arg[0-9]+]] = 0 to 100 step 2 {
// CHECK-NEXT:   [[RES1:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:   [[INC:%[0-9]+]] = affine.apply [[MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   [[RES2:%[0-9]+]] = "addi32"([[INC]], [[INC]])
// CHECK-NEXT:   affine.for %{{.*}} = 0 to 17 {
// CHECK-NEXT:     [[RES3:%[0-9]+]] = "addi32"([[IV0]], [[IV0]])
// CHECK-NEXT:     "addi32"([[RES3]], [[RES3]]) : (i32, i32) -> i32
// CHECK-NEXT:     [[INC1:%[0-9]+]] = affine.apply [[MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:     [[RES4:%[0-9]+]] = "addi32"([[INC1]], [[INC1]])
// CHECK-NEXT:     "addi32"([[RES4]], [[RES4]]) : (i32, i32) -> i32
// CHECK-NEXT:   }
// CHECK:        "foo"([[IV0]], [[RES1]])
// CHECK-NEXT:   affine.apply [[MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:   "foo"({{.*}}, [[RES2]])
// CHECK:      }
// Cleanup loop (single iteration).
// CHECK:      "addi32"(%c100, %c100)
// CHECK-NEXT: affine.for [[IV0]] = 0 to 17 {
// CHECK-NEXT:   [[RESC:%[0-9]+]] = "addi32"(%c100, %c100)
// CHECK-NEXT:   "addi32"([[RESC]], [[RESC]]) : (i32, i32) -> i32
// CHECK-NEXT: }
// CHECK-NEXT: "foo"(%c100, %{{.*}})
// CHECK-NEXT: return

// CHECK-LABEL: func @loop_nest_unknown_count_1
// CHECK-SAME: [[N:arg[0-9]+]]: index
func @loop_nest_unknown_count_1(%N : index) {
  // CHECK-NEXT: affine.for %{{.*}} = 1 to [[MAP_DIV_OFFSET]]()[%[[N]]] step 2 {
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 100 {
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // A cleanup loop should be generated here.
  // CHECK-NEXT: affine.for %{{.*}} = [[MAP_DIV_OFFSET]]()[%[[N]]] to %[[N]] {
  // CHECK-NEXT:   affine.for %{{.*}} = 1 to 100 {
  // CHECK-NEXT:     "foo"() : () -> i32
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  affine.for %i = 1 to %N {
    affine.for %j = 1 to 100 {
      %x = "foo"() : () -> i32
    }
  }
  return
}

// UJAM-FOUR-LABEL: func @loop_nest_unknown_count_2
// UJAM-FOUR-SAME: %[[N:arg[0-9]+]]: index
func @loop_nest_unknown_count_2(%N : index) {
  // UJAM-FOUR-NEXT: affine.for [[IV0:%arg[0-9]+]] = %[[N]] to  [[UBMAP]]()[%[[N]]] step 4 {
  // UJAM-FOUR-NEXT:   affine.for [[IV1:%arg[0-9]+]] = 1 to 100 {
  // UJAM-FOUR-NEXT:     "foo"([[IV0]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_1:%[0-9]+]] = affine.apply [[MAP_PLUS_1]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_1]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_2:%[0-9]+]] = affine.apply [[MAP_PLUS_2]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_2]])
  // UJAM-FOUR-NEXT:     [[IV_PLUS_3:%[0-9]+]] = affine.apply [[MAP_PLUS_3]]([[IV0]])
  // UJAM-FOUR-NEXT:     "foo"([[IV_PLUS_3]])
  // UJAM-FOUR-NEXT:   }
  // UJAM-FOUR-NEXT: }
  // The cleanup loop is a single iteration one and is promoted.
  // UJAM-FOUR-NEXT: [[RES:%[0-9]+]] = affine.apply [[UBMAP]]()[%[[N]]]
  // UJAM-FOUR-NEXT: affine.for [[IV0]] = 1 to 100 {
  // UJAM-FOUR-NEXT:   "foo"([[RES]])
  // UJAM-FOUR-NEXT: }
  affine.for %i = %N to ()[s0] -> (s0+9) ()[%N] {
    affine.for %j = 1 to 100 {
      "foo"(%i) : (index) -> ()
    }
  }
  return
}

// CHECK-LABEL: func @loop_nest_symbolic_and_min_upper_bound
// CHECK-SAME: [[M:arg[0-9]+]]: index
// CHECK-SAME: [[N:arg[0-9]+]]: index
// CHECK-SAME: [[K:arg[0-9]+]]: index
func @loop_nest_symbolic_and_min_upper_bound(%M : index, %N : index, %K : index) {
  affine.for %i = 0 to min ()[s0, s1] -> (s0, s1, 1024)()[%M, %N] {
    affine.for %j = 0 to %K {
      "foo"(%i, %j) : (index, index) -> ()
    }
  }
  return
}
// CHECK-NEXT:  affine.for [[IV0:%arg[0-9]+]] = 0 to min [[MAP_MULTI_RES]]()[%[[M]], %[[N]]] step 2 {
// CHECK-NEXT:    affine.for [[IV1:%arg[0-9]+]] = 0 to %[[K]] {
// CHECK-NEXT:      "foo"([[IV0]], [[IV1]])
// CHECK-NEXT:      [[RES:%[0-9]+]] = affine.apply [[MAP_PLUS_1]]([[IV0]])
// CHECK-NEXT:      "foo"([[RES]], [[IV1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  affine.for [[IV0]] = max [[MAP_MULTI_RES]]()[%[[M]], %[[N]]] to min [[MAP_SYM_UB]]()[%[[M]], %[[N]]] {
// CHECK-NEXT:    affine.for [[IV1]] = 0 to %[[K]] {
// CHECK-NEXT:      "foo"([[IV0]], [[IV1]])
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
