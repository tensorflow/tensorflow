// RUN: mlir_fusions_opt %s -split-input-file -xla-gpu-simplify-arith -canonicalize | FileCheck %s

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
