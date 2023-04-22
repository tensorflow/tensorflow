// RUN: mlir-hlo-opt %s -lhlo-legalize-to-linalg -split-input-file | FILECHECK_OPTS="" FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @element_wise
func @element_wise(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                   %result: memref<2x2xf32>) {
  "lmhlo.power"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = math.powf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @element_wise
func @element_wise(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                   %result: memref<2x2xf32>) {
  "lmhlo.add"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = addf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @element_wise_with_dynamic_shape
func @element_wise_with_dynamic_shape(%lhs: memref<?x?xf32>,
                                       %rhs: memref<?x?xf32>,
                                       %result: memref<?x?xf32>) {
  "lmhlo.add"(%lhs, %rhs, %result)
      : (memref<?x?xf32>, memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = addf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @element_wise_scalar
func @element_wise_scalar(%lhs: memref<f32>, %rhs: memref<f32>,
                          %result: memref<f32>) {
  "lmhlo.add"(%lhs, %rhs, %result)
      : (memref<f32>, memref<f32>, memref<f32>) -> ()
  return
}
// CHECK: %[[LHS:.*]] = memref.load
// CHECK: %[[RHS:.*]] = memref.load
// CHECK: %[[RES:.*]] = addf %[[LHS]], %[[RHS]]
// CHECK: memref.store %[[RES]]
// CHECK-NEXT: return

// -----

// CHECK-LABEL: func @minf
func @minf(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
           %result: memref<2x2xf32>) {
  "lmhlo.minimum"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf olt, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[MIN:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[ISNAN:.*]] = cmpf uno, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[NAN:.*]] = constant 0x7FC00000 : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[ISNAN]], %[[NAN]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxi
func @maxi(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
           %result: memref<2x2xi32>) {
  "lmhlo.maximum"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi sgt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @maxu
func @maxu(%lhs: memref<2x2xui32>, %rhs: memref<2x2xui32>,
           %result: memref<2x2xui32>) {
  "lmhlo.maximum"(%lhs, %rhs, %result)
      : (memref<2x2xui32>, memref<2x2xui32>, memref<2x2xui32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi ugt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @and
func @and(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "lmhlo.and"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = and %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @exp
func @exp(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.exponential"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.exp %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_exp
func @complex_exp(%input: memref<2x2xcomplex<f32>>,
                  %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.exponential"(%input, %result)
      : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.exp %[[OPERAND_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @log
func @log(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.log"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.log %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_log
func @complex_log(%input: memref<2x2xcomplex<f32>>,
                  %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.log"(%input, %result) : (memref<2x2xcomplex<f32>>,
                                  memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.log %[[OPERAND_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @log1p
func @log1p(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.log_plus_one"(%input, %result) : (memref<2x2xf32>,
                                           memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.log1p %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_log1p
func @complex_log1p(%input: memref<2x2xcomplex<f32>>,
                    %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.log_plus_one"(%input, %result) : (memref<2x2xcomplex<f32>>,
                                           memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.log1p %[[OPERAND_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @copy
func @copy(%in: memref<2x4x8xf32>, %out: memref<2x4x8xf32>) {
  "lmhlo.copy"(%in, %out) : (memref<2x4x8xf32>, memref<2x4x8xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   linalg.yield %[[OPERAND_IN]] : f32

// -----

// CHECK-LABEL: func @is_finite
func @is_finite(%input: memref<2x2xf32>, %result: memref<2x2xi1>) {
  "lmhlo.is_finite"(%input, %result) : (memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[POS_INF:.+]] = constant 0x7F800000 : f32
// CHECK-NEXT:   %[[ABS_X:.+]] = absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cmp
func @float_cmp(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                %result: memref<2x2xi1>) {
  "lmhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "EQ"}
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func @int_cmp(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
              %result: memref<2x2xi1>) {
  "lmhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "LT"}
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_eq
func @complex_cmp_eq(%lhs: memref<2xcomplex<f32>>, %rhs: memref<2xcomplex<f32>>,
                     %result: memref<2xi1>) {
  "lmhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "EQ"}
      : (memref<2xcomplex<f32>>, memref<2xcomplex<f32>>, memref<2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_neq
func @complex_cmp_neq(%lhs: memref<2xcomplex<f64>>, %rhs: memref<2xcomplex<f64>>,
                      %result: memref<2xi1>) {
  "lmhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "NE"}
      : (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>, memref<2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_divide
func @complex_divide(%lhs: memref<2xcomplex<f64>>, %rhs: memref<2xcomplex<f64>>,
                     %result: memref<2xcomplex<f64>>) {
  "lmhlo.divide"(%lhs, %rhs, %result)
      : (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>, memref<2xcomplex<f64>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: complex<f64>):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.div %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f64>

// -----

// CHECK-LABEL: func @complex_multiply
func @complex_multiply(%lhs: memref<2xcomplex<f64>>, %rhs: memref<2xcomplex<f64>>,
                       %result: memref<2xcomplex<f64>>) {
  "lmhlo.multiply"(%lhs, %rhs, %result)
      : (memref<2xcomplex<f64>>, memref<2xcomplex<f64>>, memref<2xcomplex<f64>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: complex<f64>):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.mul %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f64>

// -----

// CHECK-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>,
             %rhs: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.select"(%pred, %lhs, %rhs, %result)
    : (memref<2x2xi1>, memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota
func @iota(%out: memref<7x10xf32>) {
  "lmhlo.iota"(%out) {iota_dimension = 1 : i64} : (memref<7x10xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[RESULT:.*]]: f32):
// CHECK-NEXT:   %[[D1:.+]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = index_cast %[[D1]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func @broadcast_scalar(%operand: memref<f32>, %result: memref<4x2x1xf32>) {
  "lmhlo.broadcast"(%operand, %result) {
    broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>
  } : (memref<f32>, memref<4x2x1xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<4x?x16xf32>,
                %result: memref<4x2x1x4x?x16xf32>) {
  "lmhlo.broadcast"(%operand, %result) {
    broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>
  } : (memref<4x?x16xf32>, memref<4x2x1x4x?x16xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @dynamic_broadcast_in_dim
func @dynamic_broadcast_in_dim(%operand: memref<?x?x?xf32>,
                               %result: memref<?x?x?x?x?xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>
  } : (memref<?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @static_broadcast_in_dim_no_expansion
func @static_broadcast_in_dim_no_expansion(%operand: memref<5xf32>,
                                           %result: memref<5x10xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[0]> : tensor<1xi64>
  } : (memref<5xf32>, memref<5x10xf32>) -> ()
  return
}
// CHECK-NOT: linalg.{{.*}}shape
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @static_broadcast_in_dim_expansion
func @static_broadcast_in_dim_expansion(%operand: memref<1x5xf32>,
                                        %result: memref<5x10x100xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[2, 0]> : tensor<2xi64>
  } : (memref<1x5xf32>, memref<5x10x100xf32>) -> ()
  return
}
// CHECK: %[[RESHAPED_ARG:.*]] = memref.collapse_shape %{{.*}} {{\[}}[0, 1]]
// CHECK-SAME:                   memref<1x5xf32> into memref<5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps =
// CHECK-SAME:       [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME:   ins(%[[RESHAPED_ARG]] :
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[RESULT_MAP_0:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @static_broadcast_in_dim_scalar
func @static_broadcast_in_dim_scalar(%operand: memref<f32>,
                                     %result: memref<5x10xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[]> : tensor<0xi64>
  } : (memref<f32>, memref<5x10xf32>) -> ()
  return
}
// CHECK-NOT: linalg.{{.*}}shape
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[RESULT_MAP_0]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[CONST:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[CONST]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @static_broadcast_in_dim_with_one_to_one
func @static_broadcast_in_dim_with_one_to_one(%operand: memref<1xf32>,
                                              %result: memref<1x5xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[0]> : tensor<1xi64>
  } : (memref<1xf32>, memref<1x5xf32>) -> ()
  return
}
// CHECK-NOT: linalg.{{.*}}shape
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.+]]: f32, %{{.+}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @static_broadcast_in_dim_with_one_to_many
func @static_broadcast_in_dim_with_one_to_many(%operand: memref<1xf32>,
                                               %result: memref<5x5xf32>) {
  "lmhlo.broadcast_in_dim"(%operand, %result) {
    broadcast_dimensions = dense<[1]> : tensor<1xi64>
  } : (memref<1xf32>, memref<5x5xf32>) -> ()
  return
}
// CHECK-NOT: linalg.{{.*}}shape
// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[VALUE:.*]] = memref.load %{{.*}}[[C0]]
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.+}}: f32):
// CHECK-NEXT:   linalg.yield %[[VALUE]] : f32

// -----

// CHECK-LABEL: func @constant
func @constant(%value: memref<i32>) {
  "lmhlo.constant"(%value) {
    value = dense<10> : tensor<i32>
  } : (memref<i32>) -> ()
  return
}
// CHECK: %[[CONSTANT:.*]] = constant 10 : i32
// CHECK: affine.store %[[CONSTANT]], %{{.*}}[] : memref<i32>

// -----

// CHECK-LABEL: func @absf
func @absf(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.abs"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_abs
func @complex_abs(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xf32>) {
  "lmhlo.abs"(%input, %result)
      : (memref<2x2xcomplex<f32>>, memref<2x2xf32>) -> ()
  return
}

// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[CPLX_IN:.*]]: complex<f32>, %[[ABS_OUT:.*]]: f32):
// CHECK-NEXT:   %[[ABS:.*]] = complex.abs %[[CPLX_IN:.*]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[ABS]] : f32

// -----

// CHECK-LABEL: func @absi
func @absi(%input: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "lmhlo.abs"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[L0:.*]] = constant 0 : i32
// CHECK-NEXT:   %[[L1:.*]] = cmpi sge, %[[OPERAND_IN]], %[[L0]] : i32
// CHECK-NEXT:   %[[L2:.*]] = subi %[[L0]], %[[OPERAND_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[L1]], %[[OPERAND_IN]], %[[L2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @ceil
func @ceil(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.ceil"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = ceilf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i1_to_f32
func @convert_i1_to_f32(%input: memref<2x2xi1>, %result: memref<2x2xf32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi1>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i1, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = uitofp %[[OPERAND_IN]] : i1 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i1_to_i32
func @convert_i1_to_i32(%input: memref<2x2xi1>, %result: memref<2x2xi32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi1>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i1, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = zexti %[[OPERAND_IN]] : i1 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_f32
func @convert_i32_to_f32(%input: memref<2x2xi32>, %result: memref<2x2xf32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = sitofp %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i16_to_i32
func @convert_i16_to_i32(%input: memref<2x2xi16>,
          %result: memref<2x2xi32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi16>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = sexti %[[OPERAND_IN]] : i16 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_i16
func @convert_i32_to_i16(%input: memref<2x2xi32>, %result: memref<2x2xi16>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi32>, memref<2x2xi16>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i16):
// CHECK-NEXT:   %[[RESULT:.*]] = trunci %[[OPERAND_IN]] : i32 to i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16

// -----

// CHECK-LABEL: func @convert_f32_to_f64
func @convert_f32_to_f64(%input: memref<2x2xf32>, %result: memref<2x2xf64>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xf32>, memref<2x2xf64>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f64):
// CHECK-NEXT:   %[[RESULT:.*]] = fpext %[[OPERAND_IN]] : f32 to f64
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f64

// -----

// CHECK-LABEL: func @convert_f64_to_f32
func @convert_f64_to_f32(%input: memref<2x2xf64>, %result: memref<2x2xf32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xf64>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f64, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = fptrunc %[[OPERAND_IN]] : f64 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i32_to_i32
func @convert_i32_to_i32(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT: linalg.yield %[[OPERAND_IN]] : i32

// -----

// CHECK-LABEL: func @convert_f32_to_f32
func @convert_f32_to_f32(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.convert"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT: linalg.yield %[[OPERAND_IN]] : f32

// -----

// CHECK-LABEL: func @convert_i32_to_i1
func @convert_i32_to_i1(%input: memref<2x2xi32>, %result: memref<2x2xi1>) {
  "lmhlo.convert"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[ZERO:.*]] = constant 0 : i32
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi ne, %[[OPERAND_IN]], %[[ZERO]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i1
func @convert_f32_to_i1(%input: memref<2x2xf32>, %result: memref<2x2xi1>) {
  "lmhlo.convert"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[ZERO:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf une, %[[OPERAND_IN]], %[[ZERO]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i32
func @convert_f32_to_i32(%input: memref<2x2xf32>, %result: memref<2x2xi32>) {
  "lmhlo.convert"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = fptosi %[[OPERAND_IN]] : f32 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @cos
func @cos(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.cosine"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.cos %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @sin
func @sin(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "lmhlo.sine"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.sin %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @floor
func @floor(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.floor"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = floorf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @negf
func @negf(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.negate"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = negf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex_neg
func @complex_neg(%input: memref<2x2xcomplex<f32>>,
                  %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.negate"(%input, %result) : (memref<2x2xcomplex<f32>>,
                                     memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neg %[[OPERAND_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// -----

// CHECK-LABEL: func @negi
func @negi(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  "lmhlo.negate"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[L0:.*]] = constant 0 : i32
// CHECK-NEXT:   %[[RESULT:.*]] = subi %[[L0]], %[[OPERAND_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @not
func @not(%input: memref<2x2xi64>, %result: memref<2x2xi64>) {
  "lmhlo.not"(%input, %result) : (memref<2x2xi64>, memref<2x2xi64>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i64, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[N1:.*]] = constant -1 : i64
// CHECK-NEXT:   %[[RESULT:.*]] = xor %[[N1]], %[[OPERAND_IN]] : i64
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i64

// -----

// CHECK-LABEL: func @rem
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
                %result: memref<2x2xf32>) {
  "lmhlo.remainder"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = remf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @rsqrt
func @rsqrt(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.rsqrt"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.rsqrt %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @sign
func @sign(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.sign"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[CST_0:.*]] = constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[NE_0:.*]] = cmpf one, %[[OPERAND_IN]], %[[CST_0]] : f32
// CHECK-NEXT:   %[[NE_0_FLOAT:.*]] = uitofp %[[NE_0]] : i1 to f32
// CHECK-NEXT:   %[[SIGN:.*]] = copysign %[[NE_0_FLOAT]], %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[CMP:.*]] = cmpf uno, %[[OPERAND_IN]], %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[OPERAND_IN]], %[[SIGN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @sign_bf16
func @sign_bf16(%input: memref<2x2xbf16>, %result: memref<2x2xbf16>) {
  "lmhlo.sign"(%input, %result) : (memref<2x2xbf16>, memref<2x2xbf16>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: bf16, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[CST_0:.*]] = constant 0.000000e+00 : bf16
// CHECK-NEXT:   %[[NE_0:.*]] = cmpf one, %[[OPERAND_IN]], %[[CST_0]] : bf16
// CHECK-NEXT:   %[[NE_0_FLOAT:.*]] = uitofp %[[NE_0]] : i1 to bf16
// CHECK-NEXT:   %[[SIGN:.*]] = copysign %[[NE_0_FLOAT]], %[[OPERAND_IN]] : bf16
// CHECK-NEXT:   %[[CMP:.*]] = cmpf uno, %[[OPERAND_IN]], %[[OPERAND_IN]] : bf16
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[OPERAND_IN]], %[[SIGN]] : bf16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : bf16

// -----

// CHECK-LABEL: func @sign_i16
func @sign_i16(%input: memref<2x2xi16>, %result: memref<2x2xi16>) {
  "lmhlo.sign"(%input, %result) : (memref<2x2xi16>, memref<2x2xi16>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : i16
// CHECK-NEXT:   %[[C15:.*]] = constant 15 : i16
// CHECK-NEXT:   %[[C1:.*]] = constant 1 : i16
// CHECK-NEXT:   %[[CMP:.*]] = cmpi eq, %[[OPERAND_IN]], %[[C0]] : i16
// CHECK-NEXT:   %[[ASHR:.*]] = shift_right_signed %[[OPERAND_IN]], %[[C15]] : i16
// CHECK-NEXT:   %[[OR:.*]] = or %[[ASHR]], %[[C1]] : i16
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[C0]], %[[OR]] : i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16

// -----

// CHECK-LABEL: func @sign_complex
func @sign_complex(%input: memref<2x2xcomplex<f32>>,
                   %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.sign"(%input, %result) : (memref<2x2xcomplex<f32>>,
                                   memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.sign %[[OPERAND_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @sqrt
func @sqrt(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.sqrt"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.sqrt %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @tanh
func @tanh(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.tanh"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = math.tanh %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @complex
func @complex(%real: memref<2x2xf32>,
              %imag: memref<2x2xf32>,
              %cplx: memref<2x2xcomplex<f32>>) {
  "lmhlo.complex"(%real, %imag, %cplx)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xcomplex<f32>>) -> ()
  return
}
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[RE:.*]]: f32, %[[IM:.*]]: f32, %[[CP:.*]]: complex<f32>):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.create %[[RE]], %[[IM]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @real
func @real(%cplx: memref<2x2xcomplex<f32>>,
           %real: memref<2x2xf32>) {
  "lmhlo.real"(%cplx, %real)
      : (memref<2x2xcomplex<f32>>, memref<2x2xf32>) -> ()
  return
}
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[CPLX_IN:.*]]: complex<f32>, %[[REAL_OUT:.*]]: f32):
// CHECK-NEXT:   %[[REAL:.*]] = complex.re %[[CPLX_IN:.*]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[REAL]] : f32

// -----

// CHECK-LABEL: func @imag
func @imag(%cplx: memref<2x2xcomplex<f32>>,
           %imag: memref<2x2xf32>) {
  "lmhlo.imag"(%cplx, %imag)
      : (memref<2x2xcomplex<f32>>, memref<2x2xf32>) -> ()
  return
}
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[CPLX_IN:.*]]: complex<f32>, %[[IMAG_OUT:.*]]: f32):
// CHECK-NEXT:   %[[IMAG:.*]] = complex.im %[[CPLX_IN:.*]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[IMAG]] : f32

// -----

// CHECK: func @slice(%[[IN:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?x?xf32>)
func @slice(%operand: memref<?x?xf32>, %result: memref<?x?xf32>) {
  "lmhlo.slice"(%operand, %result) {
    start_indices = dense<[0,1]> : tensor<2xi64>,
    limit_indices = dense<[2,3]> : tensor<2xi64>,
    strides = dense<[1,1]> : tensor<2xi64>
  } : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}
// CHECK: %[[RESULT:.*]] = memref.subview %[[IN]][0, 1] [2, 2] [1, 1] : memref<?x?xf32> to memref<2x2xf32, #{{.*}}>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK: func @slice_with_strides(%[[IN:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?xf32>)
func @slice_with_strides(%operand: memref<?xf32>, %result: memref<?xf32>) {
  "lmhlo.slice"(%operand, %result) {
    limit_indices = dense<12> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (memref<?xf32>, memref<?xf32>) -> ()
  return
}
// CHECK: %[[RESULT:.*]] = memref.subview %[[IN]][0] [6] [2] : memref<?xf32> to memref<6xf32, #{{.*}}>
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK-LABEL: func @reshape_3D_2D
func @reshape_3D_2D(%arg0: memref<12x1x42xi32>, %arg1 : memref<12x42xi32>) {
  "lmhlo.reshape"(%arg0, %arg1)
    : (memref<12x1x42xi32>, memref<12x42xi32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1], [2]]
// CHECK-NEXT: linalg.copy

// -----

// CHECK-LABEL: func @reshape_4D_2D
func @reshape_4D_2D(%arg0: memref<12x42x1x1xi32>, %arg1 : memref<12x42xi32>) {
  "lmhlo.reshape"(%arg0, %arg1)
    : (memref<12x42x1x1xi32>, memref<12x42xi32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]
// CHECK-NEXT: linalg.copy

// -----

// CHECK-LABEL: func @reshape_2D_4D
func @reshape_2D_4D(%arg0: memref<12x42xi32>, %arg1 : memref<12x1x42x1xi32>) {
  "lmhlo.reshape"(%arg0, %arg1)
    : (memref<12x42xi32>, memref<12x1x42x1xi32>) -> ()
  return
}
// CHECK: memref.expand_shape %{{.*}} {{\[}}[0, 1], [2, 3]]
// CHECK-NEXT: linalg.copy

// -----

// CHECK-LABEL: func @reshape_3D_4D
func @reshape_3D_4D(%arg0: memref<1x49x16xf32>, %arg1: memref<1x784x1x1xf32>) {
  "lmhlo.reshape"(%arg0, %arg1)
   : (memref<1x49x16xf32>, memref<1x784x1x1xf32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK: memref.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]]
// CHECK: linalg.copy

// -----

// CHECK-LABEL: func @reshape_4D_3D
func @reshape_4D_3D(%arg0: memref<1x8x10x3xf32>, %arg1: memref<1x240x1xf32>) {
  "lmhlo.reshape"(%arg0, %arg1)
   : (memref<1x8x10x3xf32>, memref<1x240x1xf32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]]
// CHECK: memref.expand_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK: linalg.copy

// -----

// CHECK-LABEL: func @reshape1_4D_4D
func @reshape1_4D_4D(%arg0: memref<4x512x1x1xi32>,
                     %arg1: memref<1x4x1x512xi32>) {
  "lmhlo.reshape"(%arg0, %arg1)
   : (memref<4x512x1x1xi32>, memref<1x4x1x512xi32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]]
// CHECK: memref.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape2_4D_4D
func @reshape2_4D_4D(%arg0: memref<4x1x1x1024xi32>,
                     %arg1: memref<4x1024x1x1xi32>) {
  "lmhlo.reshape"(%arg0, %arg1)
   : (memref<4x1x1x1024xi32>, memref<4x1024x1x1xi32>) -> ()
  return
}
// CHECK: memref.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]]
// CHECK: memref.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reverse
func @reverse(%arg0: memref<2x3xf32>, %arg1: memref<2x3xf32>) {
  "lmhlo.reverse"(%arg0, %arg1) {
    dimensions = dense<1> : tensor<1xi64>
  } : (memref<2x3xf32>, memref<2x3xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]


// -----

func @conv(%input: memref<3x5x5x3xf32>, %filter: memref<2x2x3x4xf32>, %output: memref<3x5x5x4xf32>) {
  %c0 = constant 0 : index
  %0 = memref.alloc() : memref<3x5x5x4xf32>
  // CHECK: linalg.conv(%{{.+}}, %{{.+}}, %{{.+}})
  // CHECK-SAME: dilations = [1, 2]
  // CHECK-SAME: padding = dense<{{\[\[}}0, 1], [0, 1]]> : tensor<2x2xi64>
  // CHECK-SAME: strides = [2, 1]}
  // With all atributes explicitly specified.
  "lmhlo.convolution"(%filter, %input, %0) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, padding = dense<[[0, 1], [0, 1]]> : tensor<2x2xi64>, rhs_dilation = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (memref<2x2x3x4xf32>, memref<3x5x5x3xf32>, memref<3x5x5x4xf32>) -> ()

  // Dilation left unspecified, sets default dilation since linalg expects it.
  // CHECK: linalg.conv(%{{.+}}, %{{.+}}, %{{.+}})
  // CHECK-SAME: dilations = [1, 1]
  // Padding is not set if it's zero.
  // CHECK-NOT: padding
  "lmhlo.convolution"(%filter, %input, %0) {batch_group_count = 1 : i64, dimension_numbers = {input_batch_dimension = 0 : i64, input_feature_dimension = 3 : i64, input_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>, kernel_input_feature_dimension = 2 : i64, kernel_output_feature_dimension = 3 : i64, kernel_spatial_dimensions = dense<[0, 1]> : tensor<2xi64>, output_batch_dimension = 0 : i64, output_feature_dimension = 3 : i64, output_spatial_dimensions = dense<[1, 2]> : tensor<2xi64>}, feature_group_count = 1 : i64, window_strides = dense<[2, 1]> : tensor<2xi64>} : (memref<2x2x3x4xf32>, memref<3x5x5x3xf32>, memref<3x5x5x4xf32>) -> ()

  "lmhlo.copy"(%0, %output) : (memref<3x5x5x4xf32>, memref<3x5x5x4xf32>) -> ()
  "lmhlo.terminator"() : () -> ()
}

// -----

// CHECK-DAG: #[[TRANSPOSE_INPUT_MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[TRANSPOSE_OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @transpose
func @transpose(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
  "lmhlo.transpose"(%arg0, %arg1) {
    permutation = dense<[1, 0]> : tensor<2xi64>
  } : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[TRANSPOSE_INPUT_MAP]], #[[TRANSPOSE_OUTPUT_MAP]]]

// -----

// CHECK-DAG: #[[REDUCE_INPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[REDUCE_OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @reduce_add
func @reduce_add(%arg: memref<100x10xf32>,
             %init: memref<f32>,
             %result: memref<100xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.add"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<100x10xf32>, memref<f32>, memref<100xf32>) -> ()
  return
}
// CHECK: %[[INIT_VAL:.*]] = memref.load %arg1[] : memref<f32>
// CHECK: linalg.fill(%[[INIT_VAL]], %arg2)
// CHECK: linalg.generic {
// CHECK-SAME: indexing_maps = [#[[REDUCE_INPUT_MAP]], #[[REDUCE_OUTPUT_MAP]]],
// CHECK-SAME: iterator_types = ["parallel", "reduction"]}
// CHECK-SAME: ins(%arg0 : memref<100x10xf32>) outs(%arg2 : memref<100xf32>) {
// CHECK: memref.alloca
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK-NEXT: addf
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.load
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: }

// -----

// CHECK-DAG: #[[REDUCE_INPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[REDUCE_OUTPUT_MAP:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @reduce_maximum
func @reduce_maximum(%arg: memref<100x10xf32>,
             %init: memref<f32>,
             %result: memref<100xf32>) {
  "lmhlo.reduce"(%arg, %init, %result) ( {
    ^bb0(%lhs: memref<f32>, %rhs: memref<f32>, %res: memref<f32>):
      "lmhlo.maximum"(%lhs, %rhs, %res)
        : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    } ) {dimensions = dense<[1]> : tensor<1xi64>}
      : (memref<100x10xf32>, memref<f32>, memref<100xf32>) -> ()
  return
}
// CHECK: %[[INIT_VAL:.*]] = memref.load %arg1[] : memref<f32>
// CHECK: linalg.fill(%[[INIT_VAL]], %arg2)
// CHECK: linalg.generic {
// CHECK-SAME: indexing_maps = [#[[REDUCE_INPUT_MAP]], #[[REDUCE_OUTPUT_MAP]]],
// CHECK-SAME: iterator_types = ["parallel", "reduction"]}
// CHECK-SAME: ins(%arg0 : memref<100x10xf32>) outs(%arg2 : memref<100xf32>) {
// CHECK: memref.alloca
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: memref.alloca
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.store
// CHECK-NEXT: memref.load
// CHECK-NEXT: memref.load
// CHECK: cmpf
// CHECK: select
// CHECK: memref.store
// CHECK-NEXT: memref.load
// CHECK-NEXT: linalg.yield
// CHECK-NEXT: }
