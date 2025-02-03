// RUN: emitters_opt %s -split-input-file -xla-simplify-arith -cse \
// RUN:   -canonicalize | FileCheck %s

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

// -----

func.func @refine_constraints(%tensor: tensor<100xf32>) -> tensor<100xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c42_f32 = arith.constant 42.0 : f32
  %loop = scf.for %i = %c0 to %c3 step %c1
      iter_args(%in_ = %tensor) -> (tensor<100xf32>) {
    %0 = xla.apply_indexing #xla.indexing_map<"(d0) -> (d0 mod 4),"
      "domain: d0 in [0, 9]">(%i)
    %updated = tensor.insert %c42_f32 into %in_[%0] : tensor<100xf32>
    scf.yield %updated :tensor<100xf32>
  }
  func.return %loop : tensor<100xf32>
}
// CHECK-LABEL: func.func @refine_constraints
// CHECK: %[[CST:.*]] = arith.constant 4.2
// CHECK: scf.for
// CHECK: tensor.insert %[[CST]]


// -----

#map = #xla.indexing_map<
  "(d0, d1)[s0, s1] -> (((d0 * 4 + d1 * 512 + s1) floordiv 9 + s0 * 32768) mod 2400000),"
  "domain: d0 in [0, 127], d1 in [0, 575], s0 in [0, 73], s1 in [0, 3]">
#map1 = #xla.indexing_map<"(d0, d1)[s0] -> ((d0 * 4 + d1 * 512 + s0) mod 9),"
  "domain: d0 in [0, 127], d1 in [0, 575], s0 in [0, 3]">
func.func @refine_constraints_for_symbol(%arg0: tensor<2400000x9xf32>,
    %arg1: tensor<2400000x9xf32>) -> tensor<2400000x9xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c73 = arith.constant 73 : index
  %c42_f32 = arith.constant 42.0 : f32
  %th_x = gpu.thread_id  x {xla.range = [0 : index, 127 : index]}
  %bl_x = gpu.block_id  x {xla.range = [0 : index, 575 : index]}
  %0 = scf.for %i = %c0 to %c73 step %c1 iter_args(%arg3 = %arg1)
      -> (tensor<2400000x9xf32>) {
    %2 = scf.for %j = %c0 to %c4 step %c1 iter_args(%arg5 = %arg3)
        -> (tensor<2400000x9xf32>) {
      %3 = xla.apply_indexing #map(%th_x, %bl_x)[%i, %j]
      %4 = xla.apply_indexing #map1(%th_x, %bl_x)[%j]
      %inserted = tensor.insert %c42_f32 into %arg5[%3, %4]
        : tensor<2400000x9xf32>
      scf.yield %inserted : tensor<2400000x9xf32>
    }
    scf.yield %2 : tensor<2400000x9xf32>
  }
  return %0 : tensor<2400000x9xf32>
}
// CHECK: #[[$MAP:.*]] = #xla.indexing_map<"(d0, d1, d2, d3) -> (d2 * 32768 + (d0 * 4 + d1 * 512 + d3) floordiv 9),
// CHECK-LABEL: func.func @refine_constraints_for_symbol

// -----

#map = #xla.indexing_map<
  "(d0, d1, d2, d3, d4, d5)[s0] -> ((d0 * 4 + s0) floordiv 6, (d0 * 4 + s0) mod 6),"
  "domain:"
  "d0 in [0, 29],"
  "d1 in [0, 0],"
  "d2 in [0, 0],"
  "d3 in [0, 0],"
  "d4 in [0, 0],"
  "d5 in [0, 0],"
  "s0 in [0, 3],"
  "d0 * 4 + s0 in [0, 29]">
func.func @dus(%arg0: tensor<20x30xf32>, %arg1: tensor<5x6xf32>, %arg2: i32, %arg3: i32, %arg4: tensor<20x30xf32>) -> tensor<20x30xf32> {
  %c24 = arith.constant 24 : index
  %c15 = arith.constant 15 : index
  %c0 = arith.constant 0 : index
  %thread_id_x = gpu.thread_id x
  %thread_id_y = gpu.thread_id y
  %thread_id_z = gpu.thread_id z
  %block_id_x = gpu.block_id x
  %block_id_y = gpu.block_id y
  %block_id_z = gpu.block_id z
  %0 = arith.index_cast %arg2 : i32 to index
  %1 = arith.minsi %0, %c15 : index
  %2 = arith.maxsi %1, %c0 : index
  %3 = arith.index_cast %arg3 : i32 to index
  %4 = arith.minsi %3, %c24 : index
  %5 = arith.maxsi %4, %c0 : index
  %xla_loop = xla.loop (%thread_id_x, %thread_id_y, %thread_id_z, %block_id_x, %block_id_y, %block_id_z)[%i] -> (%ra, %rb) in #map iter_args(%iter = %arg4) -> (tensor<20x30xf32>) {
    %6 = arith.addi %2, %ra : index
    %7 = arith.addi %5, %rb : index
    %extracted = tensor.extract %arg1[%ra, %rb] : tensor<5x6xf32>
    %inserted = tensor.insert %extracted into %iter[%6, %7] : tensor<20x30xf32>
    xla.yield %inserted : tensor<20x30xf32>
  }
  return %xla_loop : tensor<20x30xf32>
}

// CHECK-LABEL: func.func @dus
// CHECK: arith.minsi
// CHECK-SAME: xla.range = [-9223372036854775808 : index, 15 : index]
// CHECK: arith.maxsi
// CHECK-SAME: xla.range = [0 : index, 15 : index]
// CHECK: arith.minsi
// CHECK-SAME: xla.range = [-9223372036854775808 : index, 24 : index]
// CHECK: arith.maxsi
// CHECK-SAME: xla.range = [0 : index, 24 : index]
// CHECK: xla.loop
// CHECK: arith.addi
// CHECK-SAME: xla.range = [0 : index, 19 : index]
// CHECK: arith.addi
// CHECK-SAME: xla.range = [0 : index, 29 : index]

// -----

module {
  func.func @annotate_range_abs_index(%v: i32) -> index {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %v, %c0_i32 : i32
    %1 = arith.subi %c0_i32, %v : i32
    %2 = arith.select %0, %v, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    return %3: index
  }
}

// CHECK-LABEL: @annotate_range_abs
// CHECK: arith.select
// CHECK-SAME: xla.range = [0 : index, 2147483647 : index]
// CHECK-NEXT: arith.index_cast
// CHECK-SAME: xla.range = [0 : index, 2147483647 : index]

// -----

module {
  func.func @annotate_range_abs_index(%v: i32 {xla.range = [-31 : i32, 17 : i32]}) -> index {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %v, %c0_i32 : i32
    %1 = arith.subi %c0_i32, %v : i32
    %2 = arith.select %0, %v, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    return %3: index
  }
}

// CHECK-LABEL: @annotate_range_abs
// CHECK: arith.select
// CHECK-SAME: xla.range = [0 : index, 31 : index]
// CHECK-NEXT: arith.index_cast
// CHECK-SAME: xla.range = [0 : index, 31 : index]

// -----

module {
  func.func @annotate_range_abs_index(%v: i32 {xla.range = [-5 : i32, 3 : i32]}) -> index {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi sge, %v, %c0_i32 : i32
    %1 = arith.subi %c0_i32, %v : i32
    %2 = arith.select %0, %v, %1 : i32
    %3 = arith.index_cast %2 : i32 to index
    return %3: index
  }
}

// CHECK-LABEL: @annotate_range_abs
// CHECK: arith.select
// CHECK-SAME: xla.range = [0 : index, 5 : index]
// CHECK-NEXT: arith.index_cast
// CHECK-SAME: xla.range = [0 : index, 5 : index]
