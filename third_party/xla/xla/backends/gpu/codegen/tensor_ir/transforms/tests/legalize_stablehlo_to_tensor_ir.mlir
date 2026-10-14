// RUN: xla-opt %s --legalize-stablehlo-to-tensor-ir --split-input-file | FileCheck %s

// CHECK-LABEL: nv_tensor_ir.graph @identity
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK-NEXT: results %[[IN]] : tensor<8x16xf32>
func.func @identity(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  return %arg0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @attributes_preserved
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(16,1)"})
// CHECK-SAME: -> (tensor<8x16xf32> {nv_tensor_ir.alignment = 32 : i64, nv_tensor_ir.stride = "(16,1)"})
// CHECK-NEXT: results %[[IN]] : tensor<8x16xf32>
func.func @attributes_preserved(
    %arg0: tensor<8x16xf32> {nv_tensor_ir.alignment = 16 : i64, nv_tensor_ir.stride = "(16,1)"})
    -> (tensor<8x16xf32> {nv_tensor_ir.alignment = 32 : i64, nv_tensor_ir.stride = "(16,1)"}) {
  return %arg0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @multiple_results
// CHECK-SAME: (%[[IN0:.*]]: tensor<8x16xf32>, %[[IN1:.*]]: tensor<4xf32>)
// CHECK-SAME: -> (tensor<8x16xf32>, tensor<4xf32>)
// CHECK-NEXT: results %[[IN0]], %[[IN1]] : tensor<8x16xf32>, tensor<4xf32>
func.func @multiple_results(%arg0: tensor<8x16xf32>, %arg1: tensor<4xf32>) -> (tensor<8x16xf32>, tensor<4xf32>) {
  return %arg0, %arg1 : tensor<8x16xf32>, tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @envelope_i32
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xsi32>) -> tensor<8x16xsi32>
// CHECK-NEXT: results %[[IN]] : tensor<8x16xsi32>
func.func @envelope_i32(%arg0: tensor<8x16xi32>) -> tensor<8x16xi32> {
  return %arg0 : tensor<8x16xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @envelope_ui32
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xui32>) -> tensor<8x16xui32>
// CHECK-NEXT: results %[[IN]] : tensor<8x16xui32>
func.func @envelope_ui32(%arg0: tensor<8x16xui32>) -> tensor<8x16xui32> {
  return %arg0 : tensor<8x16xui32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @envelope_i1
// CHECK-SAME: (%[[IN:.*]]: tensor<8x16xi1>) -> tensor<8x16xi1>
// CHECK-NEXT: results %[[IN]] : tensor<8x16xi1>
func.func @envelope_i1(%arg0: tensor<8x16xi1>) -> tensor<8x16xi1> {
  return %arg0 : tensor<8x16xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @abs
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = abs %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @abs(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.abs %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @ceil
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = ceil %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @ceil(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.ceil %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @floor
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = floor %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @floor(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.floor %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @negate
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = neg %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @negate(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.negate %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @cosine
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = cos %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @cosine(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.cosine %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @sine
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = sin %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @sine(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.sine %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @tan
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = tan %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @tan(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.tan %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @exponential
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = exp %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @exponential(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.exponential %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @log
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = log %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @log(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.log %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @rsqrt
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = rsqrt %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @rsqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.rsqrt %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @sqrt
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = sqrt %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @sqrt(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.sqrt %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @tanh
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = tanh_fwd %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @tanh(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.tanh %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @logistic
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = sigmoid_fwd %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @logistic(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.logistic %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @not
// CHECK-SAME: (%[[IN:.*]]: tensor<4xi1>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = not %[[IN]] : tensor<4xi1>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @not(%arg0: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.not %arg0 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// StableHLO has no erf operation, so the HLO importer emits the mhlo one and
// this pass has to accept it.
// CHECK-LABEL: nv_tensor_ir.graph @erf
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = erf %[[IN]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @erf(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = mhlo.erf %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// nv_tensor_ir has no expm1, so it is decomposed into exp(x) - 1.
// CHECK-LABEL: nv_tensor_ir.graph @exponential_minus_one
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[ONE:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT: %[[EXP:.*]] = exp %[[IN]] : tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = sub %[[EXP]], %[[ONE]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @exponential_minus_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.exponential_minus_one %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// Likewise, log1p is decomposed into log(x + 1).
// CHECK-LABEL: nv_tensor_ir.graph @log_plus_one
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[ONE:.*]] = constant dense<1.000000e+00> : tensor<4xf32>
// CHECK-NEXT: %[[ADD:.*]] = add %[[IN]], %[[ONE]] : tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = log %[[ADD]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @log_plus_one(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.log_plus_one %arg0 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @add
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = add %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @add(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @subtract
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = sub %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @subtract(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @multiply
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = mul %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @multiply(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @divide
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = div %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @divide(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.divide %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @maximum
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = max %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @maximum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @minimum
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = min %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @minimum(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @power
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = pow %[[LHS]], %[[RHS]] : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @power(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.power %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @remainder
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xsi32>
// CHECK-NEXT: %[[RES:.*]] = rem %[[LHS]], %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @remainder(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @atan2
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = atan2 %[[LHS]], %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @atan2(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.atan2 %arg0, %arg1 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @and
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xi1>, %[[RHS:.*]]: tensor<4xi1>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = and %[[LHS]], %[[RHS]] : tensor<4xi1>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @and(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.and %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @or
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xi1>, %[[RHS:.*]]: tensor<4xi1>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = or %[[LHS]], %[[RHS]] : tensor<4xi1>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @or(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = stablehlo.or %arg0, %arg1 : tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @select
// CHECK-SAME: (%[[PRED:.*]]: tensor<4xi1>, %[[ON_TRUE:.*]]: tensor<4xf32>, %[[ON_FALSE:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = binary_select %[[PRED]], %[[ON_TRUE]], %[[ON_FALSE]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @select(%arg0: tensor<4xi1>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<4xi1>, tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// nv_tensor_ir has no clamp, so it is expressed as min(max, max(min, x)). The
// nesting and operand order match the string-building converter this pass
// replaces.
// CHECK-LABEL: nv_tensor_ir.graph @clamp
// CHECK-SAME: (%[[MIN:.*]]: tensor<4xf32>, %[[IN:.*]]: tensor<4xf32>, %[[MAX:.*]]: tensor<4xf32>) -> tensor<4xf32>
// CHECK-NEXT: %[[LOWER:.*]] = max %[[MIN]], %[[IN]] : tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = min %[[MAX]], %[[LOWER]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @clamp(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> tensor<4xf32> {
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// nv_tensor_ir.max and nv_tensor_ir.min pick the signed comparison from the
// signed element type, so no extra conversion is needed.
// CHECK-LABEL: nv_tensor_ir.graph @clamp_signed
// CHECK-SAME: (%[[MIN:.*]]: tensor<4xsi32>, %[[IN:.*]]: tensor<4xsi32>, %[[MAX:.*]]: tensor<4xsi32>) -> tensor<4xsi32>
// CHECK-NEXT: %[[LOWER:.*]] = max %[[MIN]], %[[IN]] : tensor<4xsi32>
// CHECK-NEXT: %[[RES:.*]] = min %[[MAX]], %[[LOWER]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @clamp_signed(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>, %arg2: tensor<4xi32>) -> tensor<4xi32> {
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_float
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xf32>, %[[RHS:.*]]: tensor<4xf32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] olt %[[RHS]] : tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_float(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  %0 = stablehlo.compare LT, %arg0, %arg1, FLOAT : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_signed
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xsi32>, %[[RHS:.*]]: tensor<4xsi32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] ge %[[RHS]] : tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_signed(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi1> {
  %0 = stablehlo.compare GE, %arg0, %arg1, SIGNED : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @compare_unsigned
// CHECK-SAME: (%[[LHS:.*]]: tensor<4xui32>, %[[RHS:.*]]: tensor<4xui32>) -> tensor<4xi1>
// CHECK-NEXT: %[[RES:.*]] = cmp %[[LHS]] eq %[[RHS]] : tensor<4xui32>
// CHECK-NEXT: results %[[RES]] : tensor<4xi1>
func.func @compare_unsigned(%arg0: tensor<4xui32>, %arg1: tensor<4xui32>) -> tensor<4xi1> {
  %0 = stablehlo.compare EQ, %arg0, %arg1, UNSIGNED : (tensor<4xui32>, tensor<4xui32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @convert_f32_to_i32
// CHECK-SAME: (%[[IN:.*]]: tensor<4xf32>) -> tensor<4xsi32>
// CHECK-NEXT: %[[RES:.*]] = convert %[[IN]] : tensor<4xf32> -> tensor<4xsi32>
// CHECK-NEXT: results %[[RES]] : tensor<4xsi32>
func.func @convert_f32_to_i32(%arg0: tensor<4xf32>) -> tensor<4xi32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: nv_tensor_ir.graph @convert_i32_to_f32
// CHECK-SAME: (%[[IN:.*]]: tensor<4xsi32>) -> tensor<4xf32>
// CHECK-NEXT: %[[RES:.*]] = convert %[[IN]] : tensor<4xsi32> -> tensor<4xf32>
// CHECK-NEXT: results %[[RES]] : tensor<4xf32>
func.func @convert_i32_to_f32(%arg0: tensor<4xi32>) -> tensor<4xf32> {
  %0 = stablehlo.convert %arg0 : (tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
