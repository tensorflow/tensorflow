// RUN: xla-opt %s -verify-diagnostics -split-input-file | xla-opt | FileCheck %s

// -----

// CHECK-LABEL: func @ceil
func @ceil(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "xla_lhlo.ceil"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}

// -----

func @ceil(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point values}}
  "xla_lhlo.ceil"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @cos
func @cos(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "xla_lhlo.cosine"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @cos
func @cos(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "xla_lhlo.cosine"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  return
}

// -----

func @cos(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.cosine"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sin
func @sin(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "xla_lhlo.sine"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sin
func @sin(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "xla_lhlo.sine"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  return
}

// -----

func @sin(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.sine"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @add_memrefs
func @add_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.add"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @abs_memref
func @abs_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.abs"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @convert_memref
func @convert_memref(%in: memref<10xf32>, %out: memref<10xi32>) -> () {
  "xla_lhlo.convert"(%in, %out) : (memref<10xf32>, memref<10xi32>) -> ()
  return
}

// -----

func @convert_memref(%in: memref<10xf32>, %out: memref<9xi32>) -> () {
  // expected-error@+1{{requires the same shape for all operands}}
  "xla_lhlo.convert"(%in, %out) : (memref<10xf32>, memref<9xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exp
func @exp(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "xla_lhlo.exponential"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exp
func @exp(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "xla_lhlo.exponential"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  return
}

// -----

func @exp(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.exponential"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log_memref
func @log_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.log"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log_memref
func @log_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "xla_lhlo.log"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  return
}

// -----

func @log_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.log"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @neg_memref
func @neg_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.negate"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @rsqrt_memref
func @rsqrt_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.rsqrt"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @rsqrt_memref
func @rsqrt_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "xla_lhlo.rsqrt"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  return
}

// -----

func @rsqrt_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.rsqrt"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sqrt_memref
func @sqrt_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.sqrt"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sqrt_memref
func @sqrt_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "xla_lhlo.sqrt"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  return
}

// -----

func @sqrt_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.sqrt"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sign_memref
func @sign_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.sign"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @tanh_memref
func @tanh_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.tanh"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @tanh_memref
func @tanh_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "xla_lhlo.tanh"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  return
}

// -----

func @tanh_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.tanh"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

func @tanh_memref(%arg0: memref<1xf32>, %arg1: memref<2xf32>) -> () {
  // expected-error@+1{{'xla_lhlo.tanh' op requires all operands to have the same type}}
  "xla_lhlo.tanh"(%arg0, %arg1) : (memref<1xf32>, memref<2xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @add_memref
func @add_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.add"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @div_memref
func @div_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.divide"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @max_memref
func @max_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.maximum"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @min_memref
func @min_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.minimum"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @mul_memref
func @mul_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.multiply"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sub_memref
func @sub_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.subtract"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @and_memref
func @and_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "xla_lhlo.and"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @and_memref
func @and_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "xla_lhlo.and"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  return
}

// -----

func @and_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "xla_lhlo.and"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @or_memref
func @or_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "xla_lhlo.or"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @or_memref
func @or_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "xla_lhlo.or"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  return
}

// -----

func @or_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "xla_lhlo.or"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @xor_memref
func @xor_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "xla_lhlo.xor"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @xor_memref
func @xor_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "xla_lhlo.xor"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  return
}

// -----

func @xor_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "xla_lhlo.xor"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_memref
func @broadcast_in_dim_memref(%arg0: memref<1x2xi32>, %out: memref<1x2x2xi32>) -> () {
  "xla_lhlo.broadcast_in_dim"(%arg0, %out) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (memref<1x2xi32>, memref<1x2x2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_zero_rank_memref
func @broadcast_in_dim_zero_rank_memref(%arg0: memref<i32>, %out: memref<1x2x3xi32>) -> () {
  "xla_lhlo.broadcast_in_dim"(%arg0, %out) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<i32>, memref<1x2x3xi32>) -> ()
  return
}

// -----


// CHECK-LABEL: func @reduce_memref
func @reduce_memref(%input: memref<10xf32>, %init: memref<f32>, %out: memref<1xf32>) -> () {
  "xla_lhlo.reduce"(%input, %init, %out) ( {
  ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %result: memref<f32>):
    "xla_lhlo.add"(%arg1, %arg2, %result) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "xla_lhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<10xf32>, memref<f32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @fusion_memref
func @fusion_memref(%input1: memref<10xf32>, %input2: memref<10xf32>, %input3: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.fusion"() ( {
    %0 = tensor_load %input1 : memref<10xf32>
    %1 = tensor_load %input2 : memref<10xf32>
    %2 = "xla_hlo.add"(%0, %1) {name = "add"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %3 = tensor_load %input3 : memref<10xf32>
    %4 = "xla_hlo.multiply"(%2, %3) {name = "multiply"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    tensor_store %4, %out : memref<10xf32>
    "xla_lhlo.terminator"() : () -> ()
  } ) : () -> ()
  return
}

// -----

// CHECK-LABEL: func @case_memref
func @case_memref(%index: memref<i32>, %operand_1: memref<f32>, %operand_2: memref<f32>, %operand_3: memref<f32>, %out: memref<f32>) -> () {
  "xla_lhlo.case"(%index, %operand_1, %operand_2, %operand_3, %out) ( {
    ^bb0(%arg0: memref<f32>):
      "xla_lhlo.negate"(%arg0, %out) : (memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
    },  {
    ^bb0(%arg0: memref<f32>):
      "xla_lhlo.copy"(%arg0, %out) : (memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
    },  {
    ^bb0(%arg0: memref<f32>):
      "xla_lhlo.add"(%arg0, %arg0, %out) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "xla_lhlo.terminator"() : () -> ()
    }
  ) : (memref<i32>, memref<f32>, memref<f32>, memref<f32>, memref<f32>) -> ()
  return
}

// -----

func @static_memref_cast(%in: memref<10x1xf32>) {
  %out = xla_lhlo.static_memref_cast %in
           : memref<10x1xf32> -> memref<10xf32, offset: 0, strides: [1]>
  return
}
// CHECK-LABEL: func @static_memref_cast

// -----

func @static_memref_cast_dynamic_operand(%in: memref<10x?xf32>) {
  // expected-error @+1 {{operand must have static shape}}
  %out = xla_lhlo.static_memref_cast %in
           : memref<10x?xf32> -> memref<10x1xf32, offset: 0, strides: [10, 1]>
  return
}

// -----

func @static_memref_cast_dynamic_result(%in: memref<10x1xf32>) {
  // expected-error @+1 {{result must have static shape}}
  %out = xla_lhlo.static_memref_cast %in
           : memref<10x1xf32> -> memref<10x?xf32, offset: 0, strides: [?, ?]>
  return
}

// -----

func @dynamic_memref_cast(%in: memref<?xf32>) {
  %size = constant 10 : index
  %step = constant 1 : index
  %out = xla_lhlo.dynamic_memref_cast %in(%size)[%step]
           : memref<?xf32> -> memref<?xf32, offset: 0, strides: [?]>
  return
}
// CHECK-LABEL: func @dynamic_memref_cast

// -----

func @dynamic_memref_cast_incompatible_result_type(%in: memref<?xf32>) {
  // expected-error @+3 {{`sizes` args count must be equal to the rank of the output memref}}
  %size = constant 10 : index
  %step = constant 1 : index
  %out = xla_lhlo.dynamic_memref_cast %in(%size)[%step]
           : memref<?xf32> -> memref<?x?xf32, offset: 0, strides: [?, ?]>
  return
}

// -----

// CHECK-LABEL: func @atan2_memrefs
func @atan2_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @atan2_memrefs
func @atan2_memrefs(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "xla_lhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  return
}

// -----

func @atan2_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @bitcast_convert_memrefs
func @bitcast_convert_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.bitcast_convert"(%arg0, %arg_out) : (memref<1xf32>, memref<1xi32>) -> ()
  return
}

// -----

func @bitcast_convert_memrefs(%arg0: memref<1xf32>, %arg_out: memref<2xi32>) -> () {
  // expected-error@+1{{requires the same shape for all operands}}
  "xla_lhlo.bitcast_convert"(%arg0, %arg_out) : (memref<1xf32>, memref<2xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @clz_memrefs
func @clz_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.count_leading_zeros"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @expm1_memrefs
func @expm1_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.exponential_minus_one"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @expm1_memrefs
func @expm1_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "xla_lhlo.exponential_minus_one"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @floor_memrefs
func @floor_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.floor"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

func @floor_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point values}}
  "xla_lhlo.floor"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @imag_memrefs
func @imag_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.imag"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xf32>) -> ()
  return
}

// -----

func @imag_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error@+1{{must be memref of complex-type values}}
  "xla_lhlo.imag"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @real_memrefs
func @real_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.real"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xf32>) -> ()
  return
}

// -----

func @real_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error@+1{{must be memref of complex-type values}}
  "xla_lhlo.real"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @is_finite_memrefs
func @is_finite_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xi1>) -> () {
  "xla_lhlo.is_finite"(%arg0, %arg_out) : (memref<1xf32>, memref<1xi1>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log1p_memrefs
func @log1p_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.log_plus_one"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log1p_memrefs
func @log1p_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "xla_lhlo.log_plus_one"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  return
}

// -----

func @log1p_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "xla_lhlo.log_plus_one"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @not_memrefs
func @not_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.not"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @not_memrefs
func @not_memrefs(%arg0: memref<1xi1>, %arg_out: memref<1xi1>) -> () {
  "xla_lhlo.not"(%arg0, %arg_out) : (memref<1xi1>, memref<1xi1>) -> ()
  return
}

// -----

func @not_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "xla_lhlo.not"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @popcnt_memrefs
func @popcnt_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.popcnt"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

func @popcnt_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values}}
  "xla_lhlo.popcnt"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @reduce_precision_memrefs
func @reduce_precision_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.reduce_precision"(%arg0, %arg_out) { exponent_bits = 4 : i32, mantissa_bits = 4 : i32 } : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @round_memrefs
func @round_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "xla_lhlo.round_nearest_afz"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

func @round_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point values}}
  "xla_lhlo.round_nearest_afz"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_left_memrefs
func @shift_left_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.shift_left"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

func @shift_left_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values}}
  "xla_lhlo.shift_left"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic_memrefs
func @shift_right_arithmetic_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.shift_right_arithmetic"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

func @shift_right_arithmetic_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values}}
  "xla_lhlo.shift_right_arithmetic"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @shift_right_logical_memrefs
func @shift_right_logical_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "xla_lhlo.shift_right_logical"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  return
}

// -----

func @shift_right_logical_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 8/16/32/64-bit signless integer or 8/16/32/64-bit unsigned integer values}}
  "xla_lhlo.shift_right_logical"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @all_reduce_memrefs
func @all_reduce_memrefs(%arg0: memref<10xf32>, %arg_out: memref<10xf32>) -> () {
  "xla_lhlo.all_reduce"(%arg0, %arg_out) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = xla_hlo.maximum %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%max) : (tensor<f32>) -> ()
  })
  { replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64> }: (memref<10xf32>, memref<10xf32>) -> ()

  "xla_lhlo.all_reduce"(%arg0, %arg_out) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = xla_hlo.maximum %lhs, %rhs : tensor<f32>
    "xla_hlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_id = { handle = 5 : i64, type = 2 : i64 },
    constrain_layout = true,
    use_global_device_ids = true
  }: (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @collective_permute_memrefs
func @collective_permute_memrefs(%arg0: memref<128x32xf32>, %arg_out: memref<128x32xf32>) -> () {
  "xla_lhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()

  "xla_lhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_id = { handle = 5 : i64, type = 2 : i64 }
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @fft_memrefs
func @fft_memrefs(%arg0: memref<3x9xf32>, %arg_out: memref<3x5xcomplex<f32>>) -> () {
  "xla_lhlo.fft"(%arg0, %arg_out) {fft_length = dense<9> : tensor<1xi64>, fft_type = "RFFT"} : (memref<3x9xf32>, memref<3x5xcomplex<f32>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_grad_memrefs
func @batch_norm_grad_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                              %arg3: memref<8xf32>, %arg4: memref<8x8x8x8xf32>,
                              %arg_out: tuple<memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>>) -> () {
  "xla_lhlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg_out) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>,
         tuple<memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_inference_memrefs
func @batch_norm_inference_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                   %arg3: memref<8xf32>, %arg4: memref<8xf32>, %arg_out: memref<8x8x8x8xf32>) -> () {
  "xla_lhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg_out) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @batch_norm_training_memrefs
func @batch_norm_training_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                  %arg_out: tuple<memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>>) -> () {
  "xla_lhlo.batch_norm_training"(%arg0, %arg1, %arg2, %arg_out) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, tuple<memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>>) -> ()
  return
}

// -----

// CHECK-LABEL: func @cholesky_memrefs
func @cholesky_memrefs(%arg0: memref<1x291x291xf32>, %arg_out: memref<1x291x291xf32>) -> () {
  "xla_lhlo.cholesky"(%arg0, %arg_out) : (memref<1x291x291xf32>, memref<1x291x291xf32>) -> ()
  "xla_lhlo.cholesky"(%arg0, %arg_out) { lower = true } : (memref<1x291x291xf32>, memref<1x291x291xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @infeed_memrefs
func @infeed_memrefs(%arg_out: memref<3xf32>) -> () {
  "xla_lhlo.infeed"(%arg_out) { config = "x" } : (memref<3xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @outfeed_memrefs
func @outfeed_memrefs(%arg0: memref<3xf32>) -> () {
  "xla_lhlo.outfeed"(%arg0) { config = "x" } : (memref<3xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @replica_id_memrefs
func @replica_id_memrefs(%arg_out: memref<ui32>) -> () {
  "xla_lhlo.replica_id"(%arg_out) : (memref<ui32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @triangular_solve_memrefs
func @triangular_solve_memrefs(%arg0: memref<4x4xf32>, %arg1: memref<3x4xf32>, %arg_out: memref<3x4xf32>) -> () {
  "xla_lhlo.triangular_solve"(%arg0, %arg1, %arg_out) {left_side = true, lower = true, transpose_a = "NO_TRANSPOSE", unit_diagonal = true}
      : (memref<4x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @while_memrefs
func @while_memrefs(%arg0: memref<i64>, %arg_out: memref<i64>) -> () {
  "xla_lhlo.while"(%arg0, %arg_out) (
    { ^bb0(%arg: memref<i64>, %cond: memref<i1>): "xla_lhlo.terminator"() : () -> () },
    { ^bb0(%arg: memref<i64>, %body_out: memref<i64>): "xla_lhlo.terminator"() : () -> () }
  ) : (memref<i64>, memref<i64>) -> ()
  return
}
