// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file | mlir-hlo-opt | FileCheck %s

// -----

func.func @invalid_allreduce(%input0: memref<2xf32>, %input1: memref<3xf32>) {
  // expected-error@+1 {{requires operand #1 (type: 'memref<3xf32>') and result #1 (type: 'memref<2xf32>') to have same type}}
  "lmhlo.all_reduce"(%input0, %input1, %input0, %input0) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = mhlo.add %arg0, %arg1 : tensor<f32>
      "mhlo.return"(%add) : (tensor<f32>) -> ()
    })
  {channel_id = #mhlo.channel_handle<handle = 1, type = 0>, constrain_layout = false,
   replica_groups = dense<[[0, 1, 2, 3], [5, 6, 7, 4]]> : tensor<2x4xi64>,
   use_global_device_ids = false} : (memref<2xf32>, memref<3xf32>, memref<2xf32>, memref<2xf32>) -> ()
  func.return
}

// -----

func.func @invalid_allreduce(%input0: memref<2xf32>, %input1: memref<3xf16>) {
  // expected-error@+1 {{requires the same element type for all operands}}
  "lmhlo.all_reduce"(%input0, %input1, %input0, %input1) ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %add = mhlo.add %arg0, %arg1 : tensor<f32>
      "mhlo.return"(%add) : (tensor<f32>) -> ()
    })
  {channel_id = #mhlo.channel_handle<handle = 1, type = 0>, constrain_layout = false,
   replica_groups = dense<[[0, 1, 2, 3], [5, 6, 7, 8]]> : tensor<2x4xi64>,
   use_global_device_ids = false} : (memref<2xf32>, memref<3xf16>, memref<2xf32>, memref<3xf16>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%data: memref<4x16xf32>, %result:memref<4x4xf32>) {
  "lmhlo.reduce_scatter"(%data, %result) ({
    // reduction computation
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (memref<4x16xf32>, memref<4x4xf32>) -> ()
  func.return
}
// -----

// CHECK-LABEL: func @mixed_types_allgather
func.func @mixed_types_allgather(%a0: memref<1x1xf32>, %a1:memref<1x1xi32>) {
  "lmhlo.all_gather"(%a0, %a1, %a0, %a1) {all_gather_dimension = 0 : i64,
    constrain_layout = false, replica_groups = dense<0> : tensor<1x1xi64>,
    use_global_device_ids = false} : (memref<1x1xf32>, memref<1x1xi32>, memref<1x1xf32>, memref<1x1xi32>) -> ()
  func.return
}

// -----

func.func @invalid_allgather(%input0: memref<2xf32>, %output: memref<8xf32>) {
  // expected-error@+1 {{replica id #1 seen more than once}}
  "lmhlo.all_gather"(%input0, %output)
    {channel_id = #mhlo.channel_handle<handle = 1, type = 0>, constrain_layout = false,
     replica_groups = dense<[[0, 1, 1, 3], [5, 6, 7, 8]]> : tensor<2x4xi64>,
     use_global_device_ids = false, all_gather_dimension = 0 : i64} : (memref<2xf32>, memref<8xf32>) -> ()
  func.return
}

// -----

func.func @invalid_alltoall(%input0: memref<2xf32>, %output: memref<8xf32>) {
  // expected-error@+1 {{replica id #4 not seen in replica groups}}
  "lmhlo.all_to_all"(%input0, %output)
    {channel_id = #mhlo.channel_handle<handle = 1, type = 0>, constrain_layout = false,
     replica_groups = dense<[[0, 1, 2, 3], [5, 6, 7, 8]]> : tensor<2x4xi64>,
     use_global_device_ids = false} : (memref<2xf32>, memref<8xf32>) -> ()
  func.return
}

// -----

func.func @invalid_alltoall(%input0: memref<2xf32>, %output: memref<8xf32>) {
  // expected-error@+1 {{replica groups should be a rank 2 tensor of 64 bit integers}}
  "lmhlo.all_to_all"(%input0, %output)
    {channel_id = #mhlo.channel_handle<handle = 1, type = 0>, constrain_layout = false,
     replica_groups = dense<0> : tensor<1xi64>,
     use_global_device_ids = false} : (memref<2xf32>, memref<8xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @ceil
func.func @ceil(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.ceil"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  func.return
}

// -----

func.func @ceil(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point values}}
  "lmhlo.ceil"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @cos
func.func @cos(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.cosine"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @cos
func.func @cos(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.cosine"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @cos(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.cosine"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sin
func.func @sin(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.sine"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sin
func.func @sin(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.sine"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @sin(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.sine"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @add_memrefs
func.func @add_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.add"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @abs_memref
func.func @abs_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.abs"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @convert_memref
func.func @convert_memref(%in: memref<10xf32>, %out: memref<10xi32>) -> () {
  "lmhlo.convert"(%in, %out) : (memref<10xf32>, memref<10xi32>) -> ()
  func.return
}

// -----

func.func @convert_memref(%in: memref<10xf32>, %out: memref<9xi32>) -> () {
  // expected-error@+1{{requires the same shape for all operands}}
  "lmhlo.convert"(%in, %out) : (memref<10xf32>, memref<9xi32>) -> ()
  func.return
}

// -----
// CHECK-LABEL: func @exp
func.func @exp(%input: memref<2x2xf32>, %result: memref<2x2xf32>) {
  "lmhlo.exponential"(%input, %result) : (memref<2x2xf32>, memref<2x2xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @exp
func.func @exp(%input: memref<2x2xcomplex<f32>>, %result: memref<2x2xcomplex<f32>>) {
  "lmhlo.exponential"(%input, %result) : (memref<2x2xcomplex<f32>>, memref<2x2xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @exp(%input: memref<2x2xi32>, %result: memref<2x2xi32>) {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.exponential"(%input, %result) : (memref<2x2xi32>, memref<2x2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @log_memref
func.func @log_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.log"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @log_memref
func.func @log_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "lmhlo.log"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @log_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.log"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @neg_memref
func.func @neg_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.negate"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_memref
func.func @rsqrt_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.rsqrt"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @rsqrt_memref
func.func @rsqrt_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "lmhlo.rsqrt"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @rsqrt_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.rsqrt"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sqrt_memref
func.func @sqrt_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.sqrt"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sqrt_memref
func.func @sqrt_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "lmhlo.sqrt"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @sqrt_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.sqrt"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sign_memref
func.func @sign_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.sign"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @tanh_memref
func.func @tanh_memref(%in: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.tanh"(%in, %out) : (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @tanh_memref
func.func @tanh_memref(%in: memref<10xcomplex<f32>>, %out: memref<10xcomplex<f32>>) -> () {
  "lmhlo.tanh"(%in, %out) : (memref<10xcomplex<f32>>, memref<10xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @tanh_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.tanh"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

func.func @tanh_memref(%arg0: memref<1xf32>, %arg1: memref<2xf32>) -> () {
  // expected-error@+1{{'lmhlo.tanh' op requires all operands to have the same type}}
  "lmhlo.tanh"(%arg0, %arg1) : (memref<1xf32>, memref<2xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @add_memref
func.func @add_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.add"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @div_memref
func.func @div_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.divide"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @max_memref
func.func @max_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.maximum"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @min_memref
func.func @min_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.minimum"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @mul_memref
func.func @mul_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.multiply"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sub_memref
func.func @sub_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.subtract"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @and_memref
func.func @and_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "lmhlo.and"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @and_memref
func.func @and_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "lmhlo.and"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  func.return
}

// -----

func.func @and_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "lmhlo.and"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @or_memref
func.func @or_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "lmhlo.or"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @or_memref
func.func @or_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "lmhlo.or"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  func.return
}

// -----

func.func @or_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "lmhlo.or"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @xor_memref
func.func @xor_memref(%lhs: memref<10xi32>, %rhs: memref<10xi32>, %out: memref<10xi32>) -> () {
  "lmhlo.xor"(%lhs, %rhs, %out) : (memref<10xi32>, memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @xor_memref
func.func @xor_memref(%lhs: memref<10xi1>, %rhs: memref<10xi1>, %out: memref<10xi1>) -> () {
  "lmhlo.xor"(%lhs, %rhs, %out) : (memref<10xi1>, memref<10xi1>, memref<10xi1>) -> ()
  func.return
}

// -----

func.func @xor_memref(%lhs: memref<10xf32>, %rhs: memref<10xf32>, %out: memref<10xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "lmhlo.xor"(%lhs, %rhs, %out) : (memref<10xf32>, memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_memref
func.func @broadcast_in_dim_memref(%arg0: memref<1x2xi32>, %out: memref<1x2x2xi32>) -> () {
  "lmhlo.broadcast_in_dim"(%arg0, %out) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (memref<1x2xi32>, memref<1x2x2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_zero_rank_memref
func.func @broadcast_in_dim_zero_rank_memref(%arg0: memref<i32>, %out: memref<1x2x3xi32>) -> () {
  "lmhlo.broadcast_in_dim"(%arg0, %out) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (memref<i32>, memref<1x2x3xi32>) -> ()
  func.return
}

// -----


// CHECK-LABEL: func @reduce_memref
func.func @reduce_memref(%input: memref<10xf32>, %init: memref<f32>, %out: memref<1xf32>) -> () {
  "lmhlo.reduce"(%input, %init, %out) ({
  ^bb0(%arg1: memref<f32>, %arg2: memref<f32>, %result: memref<f32>):
    "lmhlo.add"(%arg1, %arg2, %result) : (memref<f32>, memref<f32>, memref<f32>) -> ()
    "lmhlo.terminator"() : () -> ()
  } ) {dimensions = dense<[0]> : tensor<1xi64>} : (memref<10xf32>, memref<f32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @fusion_memref
func.func @fusion_memref(%input1: memref<10xf32>, %input2: memref<10xf32>, %input3: memref<10xf32>, %out: memref<10xf32>) -> () {
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %input1 : memref<10xf32>
    %1 = bufferization.to_tensor %input2 : memref<10xf32>
    %2 = "mhlo.add"(%0, %1) {name = "add"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %3 = bufferization.to_tensor %input3 : memref<10xf32>
    %4 = "mhlo.multiply"(%2, %3) {name = "multiply"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    memref.tensor_store %4, %out : memref<10xf32>
    "lmhlo.terminator"() : () -> ()
  } ) : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @case_memref
func.func @case_memref(%index: memref<i32>, %operand_1: memref<f32>, %operand_2: memref<f32>, %operand_3: memref<f32>, %out: memref<f32>) -> () {
  "lmhlo.case"(%index) ({
    ^bb0:
      "lmhlo.negate"(%operand_1, %out) : (memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    },  {
    ^bb0:
      "lmhlo.copy"(%operand_2, %out) : (memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    },  {
    ^bb0:
      "lmhlo.add"(%operand_3, %operand_3, %out) : (memref<f32>, memref<f32>, memref<f32>) -> ()
      "lmhlo.terminator"() : () -> ()
    }
  ) : (memref<i32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @atan2_memrefs
func.func @atan2_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @atan2_memrefs
func.func @atan2_memrefs(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "lmhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @atan2_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.atan2"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @bitcast_convert_memrefs
func.func @bitcast_convert_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.bitcast_convert"(%arg0, %arg_out) : (memref<1xf32>, memref<1xi32>) -> ()
  func.return
}

// -----

func.func @bitcast_convert_memrefs(%arg0: memref<1xf32>, %arg_out: memref<2xi32>) -> () {
  // expected-error@+1{{requires the same shape for all operands}}
  "lmhlo.bitcast_convert"(%arg0, %arg_out) : (memref<1xf32>, memref<2xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @clz_memrefs
func.func @clz_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.count_leading_zeros"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @expm1_memrefs
func.func @expm1_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.exponential_minus_one"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @expm1_memrefs
func.func @expm1_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "lmhlo.exponential_minus_one"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @floor_memrefs
func.func @floor_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.floor"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @floor_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point values}}
  "lmhlo.floor"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @imag_memrefs
func.func @imag_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.imag"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @imag_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error@+1{{must be memref of complex-type values}}
  "lmhlo.imag"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @real_memrefs
func.func @real_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.real"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @real_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error@+1{{must be memref of complex-type values}}
  "lmhlo.real"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @is_finite_memrefs
func.func @is_finite_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xi1>) -> () {
  "lmhlo.is_finite"(%arg0, %arg_out) : (memref<1xf32>, memref<1xi1>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @log1p_memrefs
func.func @log1p_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.log_plus_one"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @log1p_memrefs
func.func @log1p_memrefs(%arg0: memref<1xcomplex<f32>>, %arg_out: memref<1xcomplex<f32>>) -> () {
  "lmhlo.log_plus_one"(%arg0, %arg_out) : (memref<1xcomplex<f32>>, memref<1xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @log1p_memref(%in: memref<10xi32>, %out: memref<10xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point or complex-type values}}
  "lmhlo.log_plus_one"(%in, %out) : (memref<10xi32>, memref<10xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @not_memrefs
func.func @not_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.not"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @not_memrefs
func.func @not_memrefs(%arg0: memref<1xi1>, %arg_out: memref<1xi1>) -> () {
  "lmhlo.not"(%arg0, %arg_out) : (memref<1xi1>, memref<1xi1>) -> ()
  func.return
}

// -----

func.func @not_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or pred (AKA boolean or 1-bit integer) values}}
  "lmhlo.not"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @popcnt_memrefs
func.func @popcnt_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.popcnt"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

func.func @popcnt_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values}}
  "lmhlo.popcnt"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @reduce_precision_memrefs
func.func @reduce_precision_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.reduce_precision"(%arg0, %arg_out) { exponent_bits = 4 : i32, mantissa_bits = 4 : i32 } : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @round_memrefs
func.func @round_memrefs(%arg0: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  "lmhlo.round_nearest_afz"(%arg0, %arg_out) : (memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @round_memrefs(%arg0: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  // expected-error@+1{{must be memref of floating-point values}}
  "lmhlo.round_nearest_afz"(%arg0, %arg_out) : (memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @shift_left_memrefs
func.func @shift_left_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.shift_left"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

func.func @shift_left_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values}}
  "lmhlo.shift_left"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @shift_right_arithmetic_memrefs
func.func @shift_right_arithmetic_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.shift_right_arithmetic"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

func.func @shift_right_arithmetic_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values}}
  "lmhlo.shift_right_arithmetic"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @shift_right_logical_memrefs
func.func @shift_right_logical_memrefs(%arg0: memref<1xi32>, %arg1: memref<1xi32>, %arg_out: memref<1xi32>) -> () {
  "lmhlo.shift_right_logical"(%arg0, %arg1, %arg_out) : (memref<1xi32>, memref<1xi32>, memref<1xi32>) -> ()
  func.return
}

// -----

func.func @shift_right_logical_memrefs(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg_out: memref<1xf32>) -> () {
  // expected-error @+1 {{must be memref of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values}}
  "lmhlo.shift_right_logical"(%arg0, %arg1, %arg_out) : (memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @all_reduce_memrefs
func.func @all_reduce_memrefs(%arg0: memref<10xf32>, %arg_out: memref<10xf32>) -> () {
  "lmhlo.all_reduce"(%arg0, %arg_out) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  { replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64> }: (memref<10xf32>, memref<10xf32>) -> ()

  "lmhlo.all_reduce"(%arg0, %arg_out) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_id = #mhlo.channel_handle<handle = 5, type = 2>,
    constrain_layout = true,
    use_global_device_ids = true
  }: (memref<10xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @collective_permute_memrefs
func.func @collective_permute_memrefs(%arg0: memref<128x32xf32>, %arg_out: memref<128x32xf32>) -> () {
  "lmhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()

  "lmhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_id = #mhlo.channel_handle<handle = 5, type = 2>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()
  func.return
}

// -----

func.func @invalid_collective_permute(%arg0: memref<128x32xf32>, %arg_out: memref<128x32xf32>) -> () {
  // expected-error@+1{{expect source_target_pairs attribute of shape (N, 2), but got (1, 3)}}
  "lmhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[2, 3, 4]]> : tensor<1x3xi64>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()
  func.return
}

// -----

func.func @invalid_collective_permute(%arg0: memref<128x32xf32>, %arg_out: memref<128x32xf32>) -> () {
  // expected-error@+1{{duplicate sources not allowed.}}
  "lmhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[1,2], [1,3]]> : tensor<2x2xi64>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()
  func.return
}

// -----

func.func @invalid_collective_permute(%arg0: memref<128x32xf32>, %arg_out: memref<128x32xf32>) -> () {
  // expected-error@+1{{duplicate targets not allowed.}}
  "lmhlo.collective_permute"(%arg0, %arg_out) {
    source_target_pairs = dense<[[1,2], [0,2]]> : tensor<2x2xi64>
  } : (memref<128x32xf32>, memref<128x32xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @fft_memrefs
func.func @fft_memrefs(%arg0: memref<3x9xf32>, %arg_out: memref<3x5xcomplex<f32>>) -> () {
  "lmhlo.fft"(%arg0, %arg_out) {fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<"fft_type RFFT">} : (memref<3x9xf32>, memref<3x5xcomplex<f32>>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @batch_norm_grad_memrefs
func.func @batch_norm_grad_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                              %arg3: memref<8xf32>, %arg4: memref<8x8x8x8xf32>,
                              %grad_operand: memref<8x8x8x8xf32>, %grad_scale: memref<8xf32>,
                              %grad_offset: memref<8xf32>) -> () {
  "lmhlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4, %grad_operand, %grad_scale, %grad_offset) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>,
         memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @batch_norm_inference_memrefs
func.func @batch_norm_inference_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                   %arg3: memref<8xf32>, %arg4: memref<8xf32>, %arg_out: memref<8x8x8x8xf32>) -> () {
  "lmhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg_out) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @batch_norm_training_memrefs
func.func @batch_norm_training_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                  %output: memref<8x8x8x8xf32>, %batch_mean: memref<8xf32>,
                                  %batch_var: memref<8xf32>) -> () {
  "lmhlo.batch_norm_training"(%arg0, %arg1, %arg2, %output, %batch_mean, %batch_var) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @cholesky_memrefs
func.func @cholesky_memrefs(%arg0: memref<1x291x291xf32>, %arg_out: memref<1x291x291xf32>) -> () {
  "lmhlo.cholesky"(%arg0, %arg_out) : (memref<1x291x291xf32>, memref<1x291x291xf32>) -> ()
  "lmhlo.cholesky"(%arg0, %arg_out) { lower = true } : (memref<1x291x291xf32>, memref<1x291x291xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @infeed_memrefs
func.func @infeed_memrefs(%arg_out: memref<3xf32>) -> () {
  "lmhlo.infeed"(%arg_out) { config = "x" } : (memref<3xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @outfeed_memrefs
func.func @outfeed_memrefs(%arg0: memref<3xf32>) -> () {
  "lmhlo.outfeed"(%arg0) { config = "x" } : (memref<3xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @replica_id_memrefs
func.func @replica_id_memrefs(%arg_out: memref<ui32>) -> () {
  "lmhlo.replica_id"(%arg_out) : (memref<ui32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @triangular_solve_memrefs
func.func @triangular_solve_memrefs(%arg0: memref<4x4xf32>, %arg1: memref<3x4xf32>, %arg_out: memref<3x4xf32>) -> () {
  "lmhlo.triangular_solve"(%arg0, %arg1, %arg_out)
       {layout_a = dense<[1, 0]> : tensor<2xindex>,
        layout_b = dense<[1, 0]> : tensor<2xindex>,
        layout_output = dense<[1, 0]> : tensor<2xindex>,
        left_side = true, lower = true, transpose_a = #mhlo<"transpose NO_TRANSPOSE">,
        unit_diagonal = true}
      : (memref<4x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @while_memrefs
func.func @while_memrefs(%arg0: memref<i64>, %arg_out: memref<i64>, %cond: memref<i1>) -> () {
  "lmhlo.while"(%cond) (
    { ^bb0: "lmhlo.terminator"() : () -> () },
    { ^bb0: "lmhlo.terminator"() : () -> () }
  ) : (memref<i1>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @while_memrefs
func.func @while_memrefs(%arg0: memref<i64>, %arg1: memref<5xf32>, %arg0_out: memref<i64>, %arg1_out: memref<5xf32>, %cond: memref<i1>) -> () {
  "lmhlo.while"(%cond) (
    { ^bb0: "lmhlo.terminator"() : () -> () },
    { ^bb0: "lmhlo.terminator"() : () -> () }
  ) : (memref<i1>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @scatter_memrefs
func.func @scatter_memrefs(%input: memref<200x100x300xf32>, %indices: memref<10x2xi32>,
                      %updates: memref<10x300xf32>, %arg_out: memref<200x100x300xf32>) -> () {
  "lmhlo.scatter" (%input, %indices, %updates, %arg_out) ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %add = mhlo.add %lhs, %rhs : tensor<f32>
    "mhlo.return"(%add) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      inserted_window_dims = [0, 1],
      index_vector_dim = 1,
      update_window_dims = [1],
      scatter_dims_to_operand_dims = [0, 1],
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (memref<200x100x300xf32>, memref<10x2xi32>, memref<10x300xf32>, memref<200x100x300xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @map_memrefs
func.func @map_memrefs(%arg0: memref<20xf32>, %arg1: memref<20xf32>, %arg_out: memref<20xf32>) -> () {
  "lmhlo.map"(%arg0, %arg1, %arg_out) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %c = mhlo.add %a, %b : tensor<f32>
    "mhlo.return"(%c) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (memref<20xf32>, memref<20xf32>, memref<20xf32>) -> ()
  func.return
}

// -----

func.func @map_memrefs(%arg0: memref<20xf32>, %arg1: memref<20xf32>, %arg_out: memref<10xf32>) -> () {
  // expected-error@+1{{requires the same shape for all operands}}
  "lmhlo.map"(%arg0, %arg1, %arg_out) ({
    ^bb0(%a: tensor<f32>, %b: tensor<f32>):
    %c = mhlo.add %a, %b : tensor<f32>
    "mhlo.return"(%c) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (memref<20xf32>, memref<20xf32>, memref<10xf32>) -> ()
  func.return
}

// -----

func.func @pad_output_rank_error(%arg0: memref<1x2x3xf16>, %arg1: memref<f16>, %arg2: memref<4x5xf16>) {
  // expected-error@+1{{output's rank(2) is not same as operand's rank(3)}}
  "lmhlo.pad"(%arg0, %arg1, %arg2) {
         edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
         edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
         interior_padding = dense<0> : tensor<3xi64>
   } : (memref<1x2x3xf16>, memref<f16>, memref<4x5xf16>) -> ()
   func.return
}

// -----

func.func @pad_config_rank_error(%arg0: memref<1x2x3xf16>, %arg1: memref<f16>, %arg2: memref<2x4x5xf16>) {
  // expected-error@+1{{pad configurations to be specified for all 3 dimensions}}
  "lmhlo.pad"(%arg0, %arg1, %arg2) {
         edge_padding_high = dense<[1, 1]> : tensor<2xi64>,
         edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
         interior_padding = dense<0> : tensor<3xi64>
   } : (memref<1x2x3xf16>, memref<f16>, memref<2x4x5xf16>) -> ()
   func.return
}

// -----

func.func @pad_config_error(%arg0: memref<2x2x3xf16>, %arg1: memref<f16>, %arg2: memref<2x4x5xf16>) {
  // expected-error@+1{{expected 0-th dimension size after padding is 6 but found 2}}
  "lmhlo.pad"(%arg0, %arg1, %arg2) {
         edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
         edge_padding_low = dense<[1, 1, 2]> : tensor<3xi64>,
         interior_padding = dense<[2, 0, 0]> : tensor<3xi64>
   } : (memref<2x2x3xf16>, memref<f16>, memref<2x4x5xf16>) -> ()
   func.return
}

// -----

// CHECK-LABEL: func @rng_get_and_update_state_memrefs
func.func @rng_get_and_update_state_memrefs(%state: memref<1xui64>) -> () {
  "lmhlo.rng_get_and_update_state"(%state) { delta = 1 : i64 } : (memref<1xui64>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sort_memrefs
func.func @sort_memrefs(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf16>,
                   %out0: memref<16x16xf32>, %out1: memref<16x16xf16>) -> () {
  "lmhlo.sort"(%arg0, %arg1, %out0, %out1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f16>, %d: tensor<f16>):
    %7 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (memref<16x16xf32>, memref<16x16xf16>, memref<16x16xf32>, memref<16x16xf16>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sort_memrefs
func.func @sort_memrefs(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf16>,
                   %out0: memref<16x16xf32>, %out1: memref<16x16xf16>) -> () {
  "lmhlo.sort"(%arg0, %arg1, %out0, %out1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f16>, %d: tensor<f16>):
    %7 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64} : (memref<16x16xf32>, memref<16x16xf16>, memref<16x16xf32>, memref<16x16xf16>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sort_memrefs
func.func @sort_memrefs(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf16>,
                   %out0: memref<16x16xf32>, %out1: memref<16x16xf16>) -> () {
  "lmhlo.sort"(%arg0, %arg1, %out0, %out1) ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<f16>, %d: tensor<f16>):
    %7 = "mhlo.compare"(%a, %b) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) : (memref<16x16xf32>, memref<16x16xf16>, memref<16x16xf32>, memref<16x16xf16>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @valid_custom_call
func.func @valid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0,3],
      results_to_target_results = [1,2]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{number of entries in the mapping for args (1) should match the number of args for the operation (2)}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0],
      results_to_target_results = [1,2]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{number of entries in the mapping for results (1) should match the number of results for the operation (2)}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0, 3],
      results_to_target_results = [1]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{entry 0 cannot appear more than once in the mapping for args}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0, 0],
      results_to_target_results = [1, 2]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{entry 1 cannot appear more than once in the mapping for results}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0, 1],
      results_to_target_results = [1, 1]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{entries in mapping for args must be >= 0 and less than target's number of args (4)}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0, 6],
      results_to_target_results = [1, 2]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_custom_call(%arg0:memref<1xf32>, %arg1:memref<1xf32>) -> () {
  // expected-error @+1 {{entries in mapping for results must be >= 0 and less than target's number of results (3)}}
  "lmhlo.custom_call"(%arg0, %arg0, %arg1, %arg1) {
    backend_config = "",
    call_target_name = "foo",
    has_side_effects = false,
    operand_segment_sizes = dense<2> : vector<2xi32>,
    target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<
      num_args = 4,
      num_results = 3,
      args_to_target_args = [0, 1],
      results_to_target_results = [1, 3]
    >
  } : (memref<1xf32>, memref<1xf32>, memref<1xf32>, memref<1xf32>) -> ()
  func.return
}

// -----

func.func @invalid_complex_abs_call(%input:memref<2xcomplex<f32>>, %result:memref<2xcomplex<f32>>) -> () {
  // expected-error @+1 {{requires output type to be the same as the element type of the input}}
  "lmhlo.abs"(%input, %result)
      : (memref<2xcomplex<f32>>, memref<2xcomplex<f32>>) -> ()
  func.return
}

// -----

func.func @invalid_float_abs_call(%input:memref<2xf32>, %result:memref<2xf64>) -> () {
  // expected-error @+1 {{requires all operands to have the same type}}
  "lmhlo.abs"(%input, %result) : (memref<2xf32>, memref<2xf64>) -> ()
  func.return
}
