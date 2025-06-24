// RUN: emitters_opt  %s -split-input-file -verify-diagnostics

#map0 = #xla.indexing_map<"(d0, d1)[s0] -> (d0, d1 + s0), domain: d0 in [1, 2], d1 in [5, 8], s0 in [0, 32]">
func.func @apply_indexing(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{operand count 1 does not match the sum of dimensions 2 and symbols 1 in the affine map}}
  %0:2 = xla.apply_indexing #map0 (%d0)
  func.return %0#0, %0#1 : index, index
}

// -----

#map0 = #xla.indexing_map<"(d0, d1)[s0] -> (d0, d1 + s0), domain: d0 in [1, 2], d1 in [5, 8], s0 in [0, 32], d0 mod 2 in [0, 1], d0 + s0 in [1, 10]">
func.func @cannot_have_constraints(%d0: index, %d1: index, %s0: index) -> (index, index) {
  // expected-error @+1 {{apply indexing op cannot have any constraints}}
  %0:2 = xla.apply_indexing #map0 (%d0, %d1)[%s0]
  func.return %0#0, %0#1 : index, index
}

// -----

#map = #xla.indexing_map<"()[s0, s1] -> (s0, s1), domain: s0 in [0, 1024], s1 in [0, 32]">
func.func @loop_result_num_mismatch(%input: tensor<1024x32xf32>,
    %init: f32) -> (f32) {
  // expected-error @+1 {{mismatch in number of loop-carried values and results}}
   %sum:2 = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (f32, f32)
  func.return %sum#0 : f32
}

// -----

#map = #xla.indexing_map<"()[s0] -> (s0, s0), domain: s0 in [0, 1024]">
func.func @loop_iv_num_mismatch(%input: tensor<1024x32xf32>,
    %init: f32) -> (f32) {
  // expected-error @+1 {{mismatch in number of induction variables 2 and RangeVars}}
   %sum = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (f32)
  func.return %sum : f32
}

// -----

#map = #xla.indexing_map<"()[s0, s1] -> (s0, s1), domain: s0 in [0, 1024], s1 in [0, 32]">

func.func @loop_types_mismatch(%input: tensor<1024x32xf32>, %init: f32) -> (i32) {
  // expected-error @+1 {{block iter arg type = 'f32', result type = 'i32' and init operand type = 'f32' should match}}
   %sum = "xla.loop"(%init) <{
      indexing_map_attr = #map,
      operandSegmentSizes = array<i32: 0, 1>
    }> ({
    ^bb0(%i: index, %j: index, %r0: index, %r1: index, %iter: f32):
      %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
      %add = arith.addf %iter, %t : f32
      xla.yield %add : f32
    }) : (f32) -> (i32)
  func.return %sum : i32
}

// -----

#map = #xla.indexing_map<"(d0)[s0, s1] -> (s0, s1), domain: d0 in [0, 3], s0 in [0, 1024], s1 in [0, 32]">

func.func @loop_op(%input: tensor<1024x32xf32>, %init: f32, %dim: index) -> (f32) {
  // expected-error @+1 {{mismatch in number of dims operands 0 and DimVars in the indexing map}}
  %sum = xla.loop ()[%i, %j] -> (%r0, %r1)
     in #map iter_args(%sum_ = %init) -> (f32) {
    %t = tensor.extract %input[%i, %j] : tensor<1024x32xf32>
    %add = arith.addf %sum_, %t : f32
    xla.yield %add : f32
  } {xla.range = [0 : index, 42 : index]}
  func.return %sum : f32
}

// -----

func.func @atomic_rmw_mismatch_yield(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  // expected-error @+1 {{mismatch in the block argument type and the yielded type}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : f32):
      %c42 = arith.constant 420 : i32
      xla.yield %c42 : i32
  }
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_rmw_mismatch_block_arg_type(%in: tensor<2x3xf32>, %i: index, %j: index)
    -> (tensor<2x3xf32>) {
  // expected-error @+1 {{mismatch in the result element type and the block argument type}}
  %ret = xla.atomic_rmw %in[%i, %j] : tensor<2x3xf32> {
    ^bb0(%current : i32):
      %c42 = arith.constant 420 : i32
      xla.yield %c42 : i32
  }
  return %ret : tensor<2x3xf32>
}

// -----

func.func @atomic_rmw_mismatch_block_arg_vector_type(%in: tensor<16xf32>, %i: index) -> (tensor<16xf32>) {
  // expected-error @+1 {{mismatch in the result element type and the block argument vector element type}}
  %ret = xla.atomic_rmw %in[%i] : tensor<16xf32> {
    ^bb0(%current : vector<2xi32>):
      %c42 = arith.constant dense<42> : vector<2xi32>
      %add = arith.addi %current, %c42 : vector<2xi32>
      xla.yield %add : vector<2xi32>
  }
  return %ret : tensor<16xf32>
}

// -----

// expected-error @+2 {{expected ::xla::BackendKind to be one of: cpu, gpu, tpu}}
// expected-error @+1 {{failed to parse XLA_BackendKindAttr parameter 'value' which is to be a `::xla::BackendKind`}}
func.func @test_backend_kind(%arg0: f32) attributes { xla.backend_kind = #xla.backend_kind<foo> } {
  func.return
}
