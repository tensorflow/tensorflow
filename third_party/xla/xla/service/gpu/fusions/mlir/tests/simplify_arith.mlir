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
    // This is true, but we don't support it yet.
    %eq = arith.cmpi slt, %arg0, %arg1 : index
    return %eq : i1
  }
}

// CHECK: @both_range
// CHECK-NEXT: cmpi
// CHECK-NEXT: return