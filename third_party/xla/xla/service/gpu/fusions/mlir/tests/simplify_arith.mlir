// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-simplify-arith -cse -canonicalize | FileCheck %s

module {
  func.func @unknown(%arg0: index {xla.range = [0 : index, 42 : index]}) -> i1 {
    %c12 = arith.constant 12 : index
    %eq = arith.cmpi eq, %arg0, %c12 : index
    return %eq : i1
  }
}

// CHECK: @unknown
// CHECK: cmpi

// -----

module {
  func.func @true(%arg0: index {xla.range = [12 : index, 42 : index]}) -> i1 {
    %c5 = arith.constant 5 : index
    %eq = arith.cmpi sge, %arg0, %c5 : index
    return %eq : i1
  }
}

// CHECK: @true
// CHECK-NEXT: constant true
// CHECK-NEXT: return

// -----

module {
  func.func @false(%arg0: index {xla.range = [12 : index, 42 : index]}) -> i1 {
    %c5 = arith.constant 5 : index
    %eq = arith.cmpi slt, %arg0, %c5 : index
    return %eq : i1
  }
}

// CHECK: @false
// CHECK-NEXT: constant false
// CHECK-NEXT: return

// -----

module {
  func.func @rhs_range(%arg0: index {xla.range = [12 : index, 42 : index]}) -> i1 {
    %c42 = arith.constant 64 : index
    %eq = arith.cmpi slt, %c42, %arg0 : index
    return %eq : i1
  }
}

// CHECK: @rhs_range
// CHECK-NEXT: constant false
// CHECK-NEXT: return

// -----

module {
  func.func @both_range(%arg0: index {xla.range = [12 : index, 42 : index]},
                        %arg1: index {xla.range = [63 : index, 100 : index]}) -> i1 {
    %eq = arith.cmpi slt, %arg0, %arg1 : index
    return %eq : i1
  }
}

// CHECK-LABEL: @both_range
// CHECK-NEXT: constant true
// CHECK-NEXT: return

// -----

module {
  func.func @minsi_lhs(%arg0: index {xla.range = [12 : index, 42 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %min = arith.minsi %arg0, %arg1 : index
    return %min : index
  }
}

// CHECK-LABEL: @minsi_lhs
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: return %[[ARG0]]

// -----

module {
  func.func @minsi_rhs(%arg0: index {xla.range = [12 : index, 42 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %min = arith.minsi %arg1, %arg0 : index
    return %min : index
  }
}

// CHECK-LABEL: @minsi_rhs
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: return %[[ARG0]]

// -----

module {
  func.func @maxsi_lhs(%arg0: index {xla.range = [12 : index, 42 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %min = arith.maxsi %arg1, %arg0 : index
    return %min : index
  }
}

// CHECK-LABEL: @maxsi_lhs
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: return %[[ARG1]]

// -----

module {
  func.func @maxsi_rhs(%arg0: index {xla.range = [12 : index, 42 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %min = arith.maxsi %arg0, %arg1 : index
    return %min : index
  }
}

// CHECK-LABEL: @maxsi_rhs
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: return %[[ARG1]]

// -----

module {
  func.func @maxsi_add(%arg0: index {xla.range = [102 : index, 142 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %add = arith.addi %arg0, %arg1 : index
    %min = arith.maxsi %add, %arg1 : index
    return %min : index
  }
}

// CHECK-LABEL: @maxsi_add
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG1]]
// CHECK-NEXT: return %[[ADD]]

// -----

module {
  func.func @minsi_add(%arg0: index {xla.range = [102 : index, 142 : index]},
                       %arg1: index {xla.range = [63 : index, 100 : index]}) -> index {
    %add = arith.addi %arg0, %arg1 : index
    %min = arith.minsi %add, %arg1 : index
    return %min : index
  }
}

// CHECK-LABEL: @minsi_add
// CHECK-SAME: (%[[ARG0:.*]]: index {{.*}}, %[[ARG1:.*]]: index {{.*}})
// CHECK-NEXT: return %[[ARG1]]

// -----

module {
  func.func @pred_reduce(%in: i1) -> i1 {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.extui %in : i1 to i32
    %shuffleResult, %valid = gpu.shuffle  down %0, %c16_i32, %c32_i32 : i32
    %1 = arith.trunci %shuffleResult : i32 to i1
    %2 = arith.ori %in, %1 : i1
    %3 = arith.extui %2 : i1 to i32
    %shuffleResult_0, %valid_1 = gpu.shuffle  down %3, %c8_i32, %c32_i32 : i32
    %4 = arith.trunci %shuffleResult_0 : i32 to i1
    %5 = arith.ori %2, %4 : i1
    %6 = arith.extui %5 : i1 to i32
    %shuffleResult_2, %valid_3 = gpu.shuffle  down %6, %c4_i32, %c32_i32 : i32
    %7 = arith.trunci %shuffleResult_2 : i32 to i1
    %8 = arith.ori %5, %7 : i1
    %9 = arith.extui %8 : i1 to i32
    %shuffleResult_4, %valid_5 = gpu.shuffle  down %9, %c2_i32, %c32_i32 : i32
    %10 = arith.trunci %shuffleResult_4 : i32 to i1
    %11 = arith.ori %8, %10 : i1
    %12 = arith.extui %11 : i1 to i32
    %shuffleResult_6, %valid_7 = gpu.shuffle  down %12, %c1_i32, %c32_i32 : i32
    %13 = arith.trunci %shuffleResult_6 : i32 to i1
    %14 = arith.ori %11, %13 : i1
    return %14 : i1
  }
}

// CHECK-LABEL: @pred_reduce
// CHECK-SAME:     (%[[IN:.*]]: i1)
// CHECK:       %[[IN_EXT:.*]] = arith.extui %[[IN]]
// CHECK-NEXT:  %[[SHUFFLE0:.*]], {{.*}} = gpu.shuffle down %[[IN_EXT]]
// CHECK-NEXT:  %[[OR0:.*]] = arith.ori %[[IN_EXT]], %[[SHUFFLE0]]
// CHECK-NEXT:  %[[SHUFFLE1:.*]], {{.*}} = gpu.shuffle down %[[OR0]]
// CHECK-NEXT:  %[[OR1:.*]] = arith.ori %[[OR0]], %[[SHUFFLE1]]
// CHECK-NEXT:  %[[SHUFFLE2:.*]], {{.*}} = gpu.shuffle down %[[OR1]]
// CHECK-NEXT:  %[[OR2:.*]] = arith.ori %[[OR1]], %[[SHUFFLE2]]
// CHECK-NEXT:  %[[SHUFFLE3:.*]], {{.*}} = gpu.shuffle down %[[OR2]]
// CHECK-NEXT:  %[[OR3:.*]] = arith.ori %[[OR2]], %[[SHUFFLE3]]
// CHECK-NEXT:  %[[SHUFFLE4:.*]], {{.*}} = gpu.shuffle down %[[OR3]]
// CHECK-NEXT:  %[[OR4:.*]] = arith.ori %[[OR3]], %[[SHUFFLE4]]
// CHECK-NEXT:  %[[RET:.*]] = arith.trunci %[[OR4]]
// CHECK-NEXT:  return %[[RET]]

// -----

module {
  func.func @andi_no_trunc_arg(%a: i4, %b: i8) -> i4 {
    %lhs = arith.extui %a : i4 to i8
    %add = arith.andi %lhs, %b : i8
    %ret = arith.trunci %add : i8 to i4
    return %ret : i4
  }
}

// CHECK-LABEL: @andi_no_trunc_arg
// CHECK-NEXT: extui
// CHECK-NEXT: andi
// CHECK-NEXT: trunci
// CHECK-NEXT: return

// -----

module {
  func.func @ori_mismatched_narrowest(%a: i8, %b: i8) -> i8 {
    %0 = arith.trunci %a : i8 to i4
    %1 = arith.extui %0 : i4 to i8
    %ret = arith.ori %b, %1 : i8
    return %ret : i8
  }
}

// CHECK-LABEL: @ori_mismatched_narrowest
// CHECK-NEXT: trunci
// CHECK-NEXT: extui
// CHECK-NEXT: ori
// CHECK-NEXT: return
