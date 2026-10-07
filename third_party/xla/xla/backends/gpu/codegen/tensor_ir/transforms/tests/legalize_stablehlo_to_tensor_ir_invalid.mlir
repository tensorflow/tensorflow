// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --verify-diagnostics --split-input-file

func.func @unsupported_xor(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.xor'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.xor %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

func.func @unsupported_totalorder(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.compare'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.compare LT, %arg0, %arg1, TOTALORDER : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// nv_tensor_ir.matmul has nowhere to record a precision request, so anything
// other than DEFAULT is rejected instead of being silently dropped.
func.func @dot_general_non_default_precision(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.dot_general'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [HIGHEST, HIGHEST] : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

// Likewise for an explicit dot algorithm.
func.func @dot_general_algorithm(%arg0: tensor<8x16xf32>, %arg1: tensor<16x32xf32>) -> tensor<8x32xf32> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.dot_general'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], algorithm = <lhs_precision_type = tf32, rhs_precision_type = tf32, accumulation_type = f32, lhs_component_count = 1, rhs_component_count = 1, num_primitive_operations = 1, allow_imprecise_accumulation = false> : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// -----

func.func @reduce_non_constant_init(%arg0: tensor<4x8xf32>, %arg1: tensor<f32>) -> tensor<4xf32> {
  // expected-error @+4 {{failed to legalize operation 'stablehlo.reduce'}}
  // expected-error @+3 {{'stablehlo.add' op operation was not legalized to nv_tensor_ir dialect}}
  // expected-error @+2 {{'stablehlo.return' op operation was not legalized to nv_tensor_ir dialect}}
  // expected-error @+1 {{'stablehlo.reduce' op operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// stablehlo.subtract is not one of the built-in nv_tensor_ir reduction modes.
func.func @reduce_unsupported_combiner(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // expected-error @+2 {{failed to legalize operation 'stablehlo.reduce'}}
  // expected-error @+1 {{'stablehlo.reduce' op operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    %1 = stablehlo.subtract %lhs, %rhs : tensor<f32>
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    stablehlo.return %1 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

// -----

// arith has no total-order comparison, in a reduction body just as much as in
// an elementwise position.
func.func @reduce_totalorder_combiner(%arg0: tensor<4x8xf32>) -> tensor<4xf32> {
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // expected-error @+2 {{failed to legalize operation 'stablehlo.reduce'}}
  // expected-error @+1 {{'stablehlo.reduce' op operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.reduce(%arg0 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
   reducer(%lhs: tensor<f32>, %rhs: tensor<f32>) {
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    %1 = stablehlo.compare LT, %lhs, %rhs, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    %2 = stablehlo.select %1, %lhs, %rhs : tensor<i1>, tensor<f32>
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    stablehlo.return %2 : tensor<f32>
  }
  return %0 : tensor<4xf32>
}

// -----

// Only single-input, single-result reductions are supported.
func.func @reduce_variadic(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %init = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  // expected-error @+2 {{failed to legalize operation 'stablehlo.reduce'}}
  // expected-error @+1 {{'stablehlo.reduce' op operation was not legalized to nv_tensor_ir dialect}}
  %0:2 = stablehlo.reduce(%arg0 init: %init), (%arg1 init: %init) across dimensions = [1] : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<f32>, tensor<f32>) -> (tensor<4xf32>, tensor<4xf32>)
   reducer(%lhs0: tensor<f32>, %rhs0: tensor<f32>) (%lhs1: tensor<f32>, %rhs1: tensor<f32>) {
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    %1 = stablehlo.add %lhs0, %rhs0 : tensor<f32>
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    %2 = stablehlo.add %lhs1, %rhs1 : tensor<f32>
    // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
    stablehlo.return %1, %2 : tensor<f32>, tensor<f32>
  }
  return %0#0, %0#1 : tensor<4xf32>, tensor<4xf32>
}

// -----

// The nv_tensor_ir dialect has no pad operation.
func.func @unsupported_pad(%arg0: tensor<4xf32>, %arg1: tensor<f32>) -> tensor<6xf32> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.pad'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.pad %arg0, %arg1, low = [1], high = [1], interior = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// -----

// StableHLO allows rank-0 clamp bounds that are broadcast against the operand,
// but nv_tensor_ir.max and nv_tensor_ir.min require matching shapes. The form
// is rejected rather than lowered into something that would not verify.
func.func @clamp_scalar_bounds(%arg0: tensor<f32>, %arg1: tensor<4xf32>, %arg2: tensor<f32>) -> tensor<4xf32> {
  // expected-error @+2 {{failed to legalize operation 'stablehlo.clamp'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<f32>, tensor<4xf32>, tensor<f32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// source_layout and result_layout are discardable attributes, so they can be
// dropped by any rewrite that does not copy them. Without them the physical
// element order is unknowable and assuming the default layout would silently
// miscompile, so the bitcast is rejected instead.
func.func @bitcast_without_layouts(%arg0: tensor<8x16xf32>) -> tensor<8x4x4xf32> {
  // expected-error @+2 {{failed to legalize operation 'mhlo.bitcast'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = mhlo.bitcast %arg0 : (tensor<8x16xf32>) -> tensor<8x4x4xf32>
  return %0 : tensor<8x4x4xf32>
}

// -----

//===----------------------------------------------------------------------===//
// Dialect leakage. The conversion target is illegal by default, so an op from
// any dialect other than nv_tensor_ir/arith/func fails the pass instead of
// surviving inside the nv_tensor_ir.graph body.
//
// The motivating producers are the mhlo ops that HloFunctionImporter emits for
// constructs StableHLO cannot express. mhlo.bitcast and mhlo.erf are handled
// by this pass now, so two unhandled ops with the same shape stand in for the
// general case here.
//===----------------------------------------------------------------------===//

func.func @unsupported_math_dialect(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error @+2 {{failed to legalize operation 'math.absf'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = math.absf %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

func.func @unsupported_tensor_dialect(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  // expected-error @+2 {{failed to legalize operation 'tensor.bitcast'}}
  // expected-error @+1 {{operation was not legalized to nv_tensor_ir dialect}}
  %0 = tensor.bitcast %arg0 : tensor<4xi32> to tensor<4xf32>
  return %0 : tensor<4xf32>
}
