// RUN: xla-opt %s -lhlo-legalize-to-linalg -split-input-file | FileCheck %s --dump-input-on-failure

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @element_wise
func @element_wise(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.add"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = addf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @element_wise_with_dynamic_shape
func @element_wise_with_dynamic_shape(%lhs: memref<?x?xf32>, %rhs: memref<?x?xf32>,
          %result: memref<?x?xf32>) {
  "xla_lhlo.add"(%lhs, %rhs, %result)
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
// CHECK: %[[LHS:.*]] = load
// CHECK: %[[RHS:.*]] = load
// CHECK: %[[RES:.*]] = addf %[[LHS]], %[[RHS]]
// CHECK: store %[[RES]]
// CHECK-NEXT: return
  "xla_lhlo.add"(%lhs, %rhs, %result)
      : (memref<f32>, memref<f32>, memref<f32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @minf
func @minf(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.minimum"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf "olt", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxi
func @maxi(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.maximum"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi "sgt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @and
func @and(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.and"(%lhs, %rhs, %result)
      : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = and %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @exp
func @exp(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.exponential"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = exp %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @log
func @log(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.log"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = log %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @copy
func @copy(%input: memref<2x4x8xf32>,
           %result: memref<2x4x8xf32>) {
  "xla_lhlo.copy"(%input, %result)
      : (memref<2x4x8xf32>, memref<2x4x8xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   linalg.yield %[[OPERAND_IN]] : f32

// -----

// CHECK-LABEL: func @float_cmp
func @float_cmp(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
    %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "EQ"}
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf "oeq", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func @int_cmp(%lhs: memref<2x2xi32>, %rhs: memref<2x2xi32>,
          %result: memref<2x2xi1>) {
  "xla_lhlo.compare"(%lhs, %rhs, %result) {comparison_direction = "LT"} : (memref<2x2xi32>, memref<2x2xi32>, memref<2x2xi1>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi "slt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @select
func @select(%pred: memref<2x2xi1>, %lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.select"(%pred, %lhs, %rhs, %result)
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
  "xla_lhlo.iota"(%out) {iota_dimension = 1 : i64} : (memref<7x10xf32>) -> ()
  return
}
// CHECK: linalg.indexed_generic {{{.*}}indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[D0:.*]]: index, %[[D1:.*]]: index, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   %[[INT_CAST:.*]] = index_cast %[[D1]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota
func @iota(%out: memref<7x10xi64>) {
  "xla_lhlo.iota"(%out) {iota_dimension = 1 : i64} : (memref<7x10xi64>) -> ()
  return
}

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @dynamic_broadcast
func @dynamic_broadcast(%operand: memref<?x?x?xf32>,
                        %result: memref<?x?x?x?x?xf32>) {
  "xla_lhlo.broadcast_in_dim"(%operand, %result)
    {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
    : (memref<?x?x?xf32>, memref<?x?x?x?x?xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast
func @broadcast(%operand: memref<5x7x1xf32>, %result: memref<7x10x6x4x5xf32>) {
  "xla_lhlo.broadcast_in_dim"(%operand, %result)
    {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
    : (memref<5x7x1xf32>, memref<7x10x6x4x5xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[RESULT_MAP_0:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func @broadcast_scalar(%operand: memref<f32>, %result: memref<7x10x6xf32>) {
  "xla_lhlo.broadcast_in_dim"(%operand, %result)
    {broadcast_dimensions = dense<[]> : tensor<0xi64>}
    : (memref<f32>, memref<7x10x6xf32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[RESULT_MAP_0]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[CONST:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[CONST]] : f32

// -----

// CHECK-LABEL: func @constant
func @constant(%value: memref<i32>) {
  "xla_lhlo.constant"(%value) {value = dense<10> : tensor<i32>} : (memref<i32>) -> ()
  return
}
// CHECK: %[[CONSTANT:.*]] = constant 10 : i32
// CHECK: store %[[CONSTANT]], %{{.*}}[] : memref<i32>

// -----

// CHECK-LABEL: func @abs
func @abs(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.abs"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

func @abs(%input: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.abs"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[L0:.*]] = constant 0 : i32
// CHECK-NEXT:   %[[L1:.*]] = cmpi "sge", %[[OPERAND_IN]], %[[L0]] : i32
// CHECK-NEXT:   %[[L2:.*]] = subi %[[L0]], %[[OPERAND_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[L1]], %[[OPERAND_IN]], %[[L2]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @ceil
func @ceil(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.ceil"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = ceilf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i32_to_f32
func @convert_i32_to_f32(%input: memref<2x2xi32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xf32>) -> ()
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
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xi16>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = zexti %[[OPERAND_IN]] : i16 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_i16
func @convert_i32_to_i16(%input: memref<2x2xi32>,
          %result: memref<2x2xi16>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xi16>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i16):
// CHECK-NEXT:   %[[RESULT:.*]] = trunci %[[OPERAND_IN]] : i32 to i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16

// -----

// CHECK-LABEL: func @convert_f32_to_f64
func @convert_f32_to_f64(%input: memref<2x2xf32>,
          %result: memref<2x2xf64>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf64>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f64):
// CHECK-NEXT:   %[[RESULT:.*]] = fpext %[[OPERAND_IN]] : f32 to f64
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f64

// -----

// CHECK-LABEL: func @convert_f64_to_f32
func @convert_f64_to_f32(%input: memref<2x2xf64>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xf64>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f64, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = fptrunc %[[OPERAND_IN]] : f64 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i32_to_i32
func @convert_i32_to_i32(%input: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]: i32):
// CHECK-NEXT: linalg.yield %[[OPERAND_IN]] : i32

// -----

// CHECK-LABEL: func @convert_f32_to_f32
func @convert_f32_to_f32(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.convert"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]: f32):
// CHECK-NEXT: linalg.yield %[[OPERAND_IN]] : f32

// -----

// CHECK-LABEL: func @cos
func @cos(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.cosine"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = cos %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @neg
func @neg(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.negate"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = negf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @neg
func @neg(%input: memref<2x2xi32>,
          %result: memref<2x2xi32>) {
  "xla_lhlo.negate"(%input, %result)
      : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[L0:.*]] = constant 0 : i32 
// CHECK-NEXT:   %[[RESULT:.*]] = subi %[[L0]], %[[OPERAND_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @rem
func @remainder(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.remainder"(%lhs, %rhs, %result)
      : (memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = remf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @rsqrt
func @rsqrt(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.rsqrt"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = rsqrt %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @sign
func @sign(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.sign"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[CST:.*]] = constant 1.000000e+00 : f32
// CHECK-NEXT:   %[[RESULT:.*]] = copysign %[[CST]], %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @sqrt
func @sqrt(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.sqrt"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = sqrt %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @tanh
func @tanh(%input: memref<2x2xf32>,
          %result: memref<2x2xf32>) {
  "xla_lhlo.tanh"(%input, %result)
      : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %[[RESULT_OUT:.*]]):
// CHECK-NEXT:   %[[RESULT:.*]] = tanh %[[OPERAND_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32


// -----

// CHECK: func @slice(%[[IN:.*]]: memref<?x?xf32>, %[[OUT:.*]]: memref<?x?xf32>)
func @slice(%operand: memref<?x?xf32>, %result: memref<?x?xf32>) {
  "xla_lhlo.slice"(%operand, %result) {
    start_indices = dense<[0,1]> : tensor<2xi64>,
    limit_indices = dense<[2,3]> : tensor<2xi64>,
    strides = dense<[1,1]> : tensor<2xi64>
  } : (memref<?x?xf32>, memref<?x?xf32>) -> ()
  return
}
// CHECK: %[[L0:.*]] = constant 0 : index
// CHECK: %[[L2:.*]] = constant 2 : index
// CHECK: %[[L1:.*]] = constant 1 : index
// CHECK: %[[LHS:.*]] = linalg.range %[[L0]] : %[[L2]] : %[[L1]]
// CHECK: %[[R0:.*]] = constant 1 : index
// CHECK: %[[R2:.*]] = constant 3 : index
// CHECK: %[[R1:.*]] = constant 1 : index
// CHECK: %[[RHS:.*]] = linalg.range %[[R0]] : %[[R2]] : %[[R1]]
// CHECK: %[[RESULT:.*]] = linalg.slice %[[IN]][%[[LHS]], %[[RHS]]]
// CHECK: linalg.copy(%[[RESULT]], %[[OUT]])

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, 0, d1)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reshape_3D_2D
func @reshape_3D_2D(%arg0: memref<12x1x42xi32>, %arg1 : memref<12x42xi32>) {
  "xla_lhlo.reshape"(%arg0, %arg1) : (memref<12x1x42xi32>, memref<12x42xi32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reshape_4D_2D
func @reshape_4D_2D(%arg0: memref<12x42x1x1xi32>, %arg1 : memref<12x42xi32>) {
  "xla_lhlo.reshape"(%arg0, %arg1) : (memref<12x42x1x1xi32>, memref<12x42xi32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @reshape_2D_4D
func @reshape_2D_4D(%arg0: memref<12x42xi32>, %arg1 : memref<12x1x42x1xi32>) {
  "xla_lhlo.reshape"(%arg0, %arg1) : (memref<12x42xi32>, memref<12x1x42x1xi32>) -> ()
  return
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
