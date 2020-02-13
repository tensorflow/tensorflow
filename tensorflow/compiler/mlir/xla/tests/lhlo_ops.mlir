// RUN: tf-opt %s -verify-diagnostics -split-input-file | tf-opt | FileCheck %s

func @enforce_same_shape(%arg0: memref<1xf32>, %arg1: memref<2xf32>) -> () {
  // expected-error@+1{{'xla_lhlo.tanh' op requires all operands to have the same type}}
  "xla_lhlo.tanh"(%arg0, %arg1) : (memref<1xf32>, memref<2xf32>) -> ()
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
func @convert_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.convert"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @exp_memref
func @exp_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.exp"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @log_memref
func @log_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.log"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @neg_memref
func @neg_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.neg"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @rsqrt_memref
func @rsqrt_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.rsqrt"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
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

// CHECK-LABEL: func @add_memref
func @add_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.add"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @div_memref
func @div_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.div"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @max_memref
func @max_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.max"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @min_memref
func @min_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.min"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @mul_memref
func @mul_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.mul"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @sub_memref
func @sub_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.sub"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @and_memref
func @and_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "xla_lhlo.and"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
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
    %4 = "xla_hlo.mul"(%2, %3) {name = "multiply"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    tensor_store %4, %out : memref<10xf32>
    "xla_lhlo.terminator"() : () -> ()
  } ) : () -> ()
  return
}
