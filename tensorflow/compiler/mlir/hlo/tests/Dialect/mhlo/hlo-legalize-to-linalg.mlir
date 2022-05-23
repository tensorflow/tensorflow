// RUN: mlir-hlo-opt %s -hlo-legalize-to-linalg -split-input-file | FILECHECK_OPTS="" FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: integer_add
func.func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: addi
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_add
func.func @complex_add(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.add
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: mulf
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
func.func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: muli
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_mul
func.func @complex_mul(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.mul
  %0 = "mhlo.multiply"(%lhs, %rhs)
          : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_remainder
func.func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: remf
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
func.func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.remsi
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @population_count_integer
func.func @population_count_integer(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctpop
  %0 = "mhlo.popcnt"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.sqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_rsqrt
func.func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: rsqrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_cbrt
func.func @float_cbrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.cbrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[ABS:.+]] = math.abs %arg1
  // CHECK: %[[THIRD:.+]] = arith.constant 0.333333343
  // CHECK: %[[POW:.+]] = math.powf %[[ABS]], %[[THIRD]]
  // CHECK: %[[RESULT:.+]] = math.copysign %[[POW]], %arg1
  // CHECK: linalg.yield %[[RESULT]]
  func.return %tensor_result : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func @float_sub
func.func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: subf
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_sub
func.func @complex_sub(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sub
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: math.abs
  %0 = "mhlo.abs"(%arg0) {someattr} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
func.func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_exp
func.func @complex_exp(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                 -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: expm1
  %0 = "mhlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_log
func.func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log
func.func @complex_log(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xcomplex<f32>>)
                         -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log1p
func.func @complex_log1p(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                  -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_logistic
func.func @float_logistic(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK: %[[C1:.*]] = arith.constant 1.{{.*}}e+00
  // CHECK: %[[NEG_ARG:.*]] = arith.negf %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = math.exp %[[NEG_ARG]]
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = arith.addf %[[C1]], %[[EXP_NEG_ARG]]
  // CHECK: %[[RESULT:.*]] = arith.divf %[[C1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.ceil
  %0 = "mhlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.floor
  %0 = "mhlo.floor"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: negf
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_neg
func.func @complex_neg(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.neg
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xcomplex<f32>>)
                            -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sign
  %0 = "mhlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>>)
                          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = "mhlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_tanh
func.func @complex_tanh(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.tanh"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.tanh
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @integer_and
func.func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: and
  %0 = "mhlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_or
func.func @integer_or(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: or
  %0 = "mhlo.or"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_xor
func.func @integer_xor(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: xor
  %0 = "mhlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctlz
  %0 = "mhlo.count_leading_zeros"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_cmp
func.func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction EQ">}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cmp_ne
func.func @float_cmp_ne(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction NE">}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func.func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction LT">}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  func.return %0 : tensor<2x2xi1>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_eq
func.func @complex_cmp_eq(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction EQ">}
          : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: linalg.init_tensor [2] : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_neq
func.func @complex_cmp_neq(%lhs: tensor<2xcomplex<f64>>,
                      %rhs: tensor<2xcomplex<f64>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction NE">}
          : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: linalg.init_tensor [2] : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cos
func.func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_cos
func.func @complex_cos(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_sin
func.func @complex_sin(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @copy(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "mhlo.copy"(%input) : (tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>)
  func.return %0 : tensor<2x4x8xf32>
}
// CHECK: return [[ARG]] : tensor<2x4x8xf32>

// -----

// CHECK-LABEL: func @is_finte
func.func @is_finte(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %0 = "mhlo.is_finite"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32
// CHECK-NEXT:   %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK-NEXT:   %[[ABS_X:.+]] = math.abs %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = arith.cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @round
func.func @round(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[HALF:.+]] = arith.constant 5.000000e-01
  // CHECK: %[[ABS:.+]] = math.abs %arg1
  // CHECK: %[[ADD:.+]] = arith.addf %[[ABS]], %[[HALF]]
  // CHECK: %[[CEIL:.+]] = math.floor %[[ADD]]
  // CHECK: %[[COPY:.+]] = math.copysign %[[CEIL]], %arg1
  // CHECK: linalg.yield %[[COPY]]
  %0 = "mhlo.round_nearest_afz"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-DAG:   #[[SCALAR_MAP:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG:   #[[ID_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @select_scalar_pred_dyn
// CHECK-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
func.func @select_scalar_pred_dyn(%pred : tensor<i1>, %lhs: tensor<2x?xf32>, %rhs: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = "mhlo.select"(%pred, %lhs, %rhs) {someattr} : (tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>) -> (tensor<2x?xf32>)
  func.return %0 : tensor<2x?xf32>
}
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[SHAPE:.*]] = shape.shape_of %[[LHS]] : tensor<2x?xf32> -> tensor<2xindex>
// CHECK-DAG: %[[DIM:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
// CHECK-DAG:  %[[DST:.*]] = linalg.init_tensor [2, %[[DIM]]]
// CHECK:      linalg.generic
// CHECK-SAME:   indexing_maps = [#[[SCALAR_MAP]], #[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel"]
// CHECK-SAME:   ins(%[[PRED]], %[[LHS]], %[[RHS]] : tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-SAME:   {someattr}
// CHECK:      ^bb0(%[[PRED_:.*]]: i1, %[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %{{.*}}: f32):
// CHECK:        %[[RES:.*]] = arith.select %[[PRED_]], %[[LHS_]], %[[RHS_]] : f32
// CHECK:        linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func.func @broadcast_scalar(%arg: tensor<f32>) -> tensor<4x2x1xf32> {
  %0 = "mhlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<f32>) -> tensor<4x2x1xf32>
  func.return %0: tensor<4x2x1xf32>
}
// CHECK: linalg.init_tensor [4, 2, 1] : tensor<4x2x1xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg: tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32> {
  %0 = "mhlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32>
  func.return %0: tensor<4x2x1x4x?x16xf32>
}
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C4_0:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C1_1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1_1]] : tensor<4x?x16xf32>
// CHECK-DAG: %[[C2_2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK: %[[T:.*]] = tensor.from_elements %[[C4]], %[[C2]], %[[C1]], %[[C4_0]], %[[DIM]], %[[C16]] : tensor<6xindex>
// CHECK: %[[C4_3:.*]] = arith.constant 4 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[T]][%[[C4_3]]] : tensor<6xindex>
// CHECK: %3 = linalg.init_tensor [4, 2, 1, 4, %2, 16] : tensor<4x2x1x4x?x16xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%operand: tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32>
  func.return %0 : tensor<7x10x6x4x5xf32>
}
// CHECK: linalg.init_tensor [7, 10, 6, 4, 5] : tensor<7x10x6x4x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast_in_dim_ui32
func.func @broadcast_in_dim_ui32(%operand: tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xui32>) -> tensor<7x10x6x4x5xui32>
  func.return %0 : tensor<7x10x6x4x5xui32>
}
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<5x7x1xui32> to tensor<5x7x1xi32>
// CHECK: linalg.init_tensor [7, 10, 6, 4, 5] : tensor<7x10x6x4x5xi32>
// CHECK: %[[RES:.*]] = linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @broadcast_in_dim_with_one_to_one
func.func @broadcast_in_dim_with_one_to_one(
         %operand: tensor<1xf32>) -> tensor<1x5xf32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
         : (tensor<1xf32>) -> tensor<1x5xf32>
  func.return %0 : tensor<1x5xf32>
}
// CHECK: linalg.init_tensor [1, 5] : tensor<1x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func.func @broadcast_scalar(%operand: tensor<f32>) -> tensor<7x10x6xf32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
        {broadcast_dimensions = dense<[]> : tensor<0xi64>}
        : (tensor<f32>) -> tensor<7x10x6xf32>
  func.return %0 : tensor<7x10x6xf32>
}
// CHECK: linalg.init_tensor [7, 10, 6] : tensor<7x10x6xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}
        : (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  func.return %0 : tensor<3x2x5x9xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose_dynamic
func.func @transpose_dynamic(%arg0: tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>, someattr}
        : (tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32>
  func.return %0 : tensor<?x?x?x9xi32>
}
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[A0:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?x9x?xi32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[A1:.*]] = tensor.dim %arg0, %[[C1]] : tensor<?x?x9x?xi32>
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C9:.*]] = arith.constant 9 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[A3:.*]] = tensor.dim %arg0, %[[C3]] : tensor<?x?x9x?xi32>
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[A1]], %[[A0]], %[[A3]], %[[C9]] : tensor<4xindex>
// CHECK: %[[C0_0:.*]] = arith.constant 0 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C0_0]]] : tensor<4xindex>
// CHECK: %[[C1_1:.*]] = arith.constant 1 : index
// CHECK: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C1_1]]] : tensor<4xindex>
// CHECK: %[[C2_2:.*]] = arith.constant 2 : index
// CHECK: %[[D3:.*]] = tensor.extract %[[SHAPE]][%[[C2_2]]] : tensor<4xindex>
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D1]], %[[D0]], %[[D3]], 9] : tensor<?x?x?x9xi32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins(%arg0 : tensor<?x?x9x?xi32>) outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-SAME: {someattr}

// -----

// CHECK-LABEL: func @real_dynamic_slice
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<256x?xf32>, %[[START_INDICES:.*]]: tensor<2xindex>, %[[LIMIT_INDICES:.*]]: tensor<2xindex>, %[[STRIDES:.*]]: tensor<2xindex>)
func.func @real_dynamic_slice(%input: tensor<256x?xf32>, %start_indices: tensor<2xindex>, %limit_indices: tensor<2xindex>, %strides: tensor<2xindex>) -> tensor<256x?xf32> {
  %0 = "mhlo.real_dynamic_slice"(%input, %start_indices, %limit_indices, %strides) : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
  func.return %0 : tensor<256x?xf32>
}
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64

// 1.   For dimension 0.
// CHECK-NEXT: %[[DIM0:.*]] = arith.constant 0 : index

// 1.1. Fetch start index, limit index and stride.
// CHECK-NEXT: %[[START0:.*]] = tensor.extract %[[START_INDICES]][%[[DIM0]]] : tensor<2xindex>
// CHECK-NEXT: %[[LIMIT0:.*]] = tensor.extract %[[LIMIT_INDICES]][%[[DIM0]]] : tensor<2xindex>
// CHECK-NEXT: %[[STRIDE0:.*]] = tensor.extract %[[STRIDES]][%[[DIM0]]] : tensor<2xindex>

// 1.2. Since 0-th dimension size of the result is known, we set it as size[0]
// CHECK-NEXT: %[[SIZE0:.*]] = arith.constant 256 : index

// 1.3. Compute upper bound for starting index = operand_dim[0] - size[0].
//      where, size[0] is computed at step 1.2
// CHECK-NEXT: %[[OPERAND_DIM0:.*]] = arith.constant 256 : index
// CHECK-NEXT: %[[UB:.*]] = arith.constant 0 : index

// 1.4. Clamp starting index : 0 <= start <= ub
//      where upper bound (ub) is computed at step 1.3
// CHECK-NEXT: %[[START0_Int:.*]] = arith.index_cast %[[START0]] : index to i64
// CHECK-NEXT: %[[UB0:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[MIN0:.*]] = arith.minsi %[[START0_Int]], %[[UB0]] : i64
// CHECK-NEXT: %[[MAX0:.*]] = arith.maxsi %[[MIN0]], %[[ZERO]] : i64
// CHECK-NEXT: %[[CLAMPED_START0:.*]] = arith.index_cast %[[MAX0]] : i64 to index

// 2.   For dimension 1.
// CHECK-NEXT: %[[DIM1:.*]] = arith.constant 1 : index

// 2.1. Fetch start index, limit index and stride.
// CHECK-NEXT: %[[START1:.*]] = tensor.extract %[[START_INDICES]][%[[DIM1]]] : tensor<2xindex>
// CHECK-NEXT: %[[LIMIT1:.*]] = tensor.extract %[[LIMIT_INDICES]][%[[DIM1]]] : tensor<2xindex>
// CHECK-NEXT: %[[STRIDE1:.*]] = tensor.extract %[[STRIDES]][%[[DIM1]]] : tensor<2xindex>

// 2.2. Since 1-th dimension of result is unknown we compute result size at 1-th
//      dimension as size[1] = (limit - start)/stride
// CHECK-NEXT: %[[DELTA1:.*]] = arith.subi %[[LIMIT1]], %[[START1]] : index
// CHECK-NEXT: %[[SIZE1:.*]] = arith.ceildivui %[[DELTA1]], %[[STRIDE1]] : index

// 2.3. Compute upper bound for starting index = operand_dim[1] - size[1].
//      where, size[1] is computed at step 2.2
// CHECK-NEXT: %[[OPERAND_DIM1:.*]] = tensor.dim %[[OPERAND]], %[[DIM1]] : tensor<256x?xf32>
// CHECK-NEXT: %[[UB:.*]] = arith.subi %[[OPERAND_DIM1]], %[[SIZE1]] : index

// 2.4. Clamp starting index : 0 <= start <= ub
//      where upper bound (ub) is computed at step 2.3
// CHECK-NEXT: %[[START1_Int:.*]] = arith.index_cast %[[START1]] : index to i64
// CHECK-NEXT: %[[UB1:.*]] = arith.index_cast %[[UB]] : index to i64
// CHECK-NEXT: %[[MIN1:.*]] = arith.minsi %[[START1_Int]], %[[UB1]] : i64
// CHECK-NEXT: %[[MAX1:.*]] = arith.maxsi %[[MIN1]], %[[ZERO]] : i64
// CHECK-NEXT: %[[CLAMPED_START1:.*]] = arith.index_cast %[[MAX1]] : i64 to index

// CHECK-NEXT: %[[SLICE:.*]] = tensor.extract_slice %[[OPERAND]][%[[CLAMPED_START0]], %[[CLAMPED_START1]]] [256, %[[SIZE1]]] [%[[STRIDE0]], %[[STRIDE1]]] : tensor<256x?xf32> to tensor<256x?xf32>
// CHECK-NEXT: return %[[SLICE]] : tensor<256x?xf32>

// -----

// CHECK-LABEL: real_dynamic_slice_with_int
// Verify that legalization of real_dynamic_slice legalization with integer
// dims work & passes verification.
func.func public @real_dynamic_slice_with_int(%arg0: tensor<10xi32> , %arg1: tensor<1xi32> ) -> tensor<?xi32> {
  %0 = mhlo.constant dense<0> : tensor<1xi32>
  %1 = mhlo.constant dense<1> : tensor<1xi32>
  %2 = mhlo.constant dense<0> : tensor<i32>
// Verify cast is inserted for extracted value feeding into subtraction with
// result of dim.
// CHECK: %[[SIZE1:.*]] = arith.ceildivui
// CHECK: %[[SIZE2:.*]] = arith.index_cast %[[SIZE1]] : i32 to index
// CHECK: arith.subi %{{.*}}, %[[SIZE2]] : index
  %4 = "mhlo.real_dynamic_slice"(%arg0, %0, %arg1, %1) : (tensor<10xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
  func.return %4 : tensor<?xi32>
}

// -----

// CHECK-LABEL: func @reshape_0D_1D
func.func @reshape_0D_1D(%arg0: tensor<i32>) -> tensor<1xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<i32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}
// CHECK: tensor.expand_shape %{{.*}} [] : tensor<i32> into tensor<1xi32>

// -----

func.func @reshape_0D_1D_unsigned(%arg0: tensor<ui32>) -> tensor<1xui32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<ui32>) -> tensor<1xui32>
  func.return %0 : tensor<1xui32>
}
// CHECK-LABEL: func @reshape_0D_1D_unsigned
// CHECK-SAME:    %[[ARG_UNSIGNED:[a-zA-Z0-9_]*]]
// CHECK:         %[[ARG_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[ARG_UNSIGNED]] : tensor<ui32> to tensor<i32>
// CHECK:         %[[RET_SIGNLESS:.*]] = tensor.expand_shape %[[ARG_SIGNLESS]] [] : tensor<i32> into tensor<1xi32>
// CHECK:         %[[RET_UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[RET_SIGNLESS]] : tensor<1xi32> to tensor<1xui32>
// CHECK:         return %[[RET_UNSIGNED]] : tensor<1xui32>

// -----

// CHECK-LABEL: func @reshape_1D_0D
func.func @reshape_1D_0D(%arg0: tensor<1xi32>) -> tensor<i32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: tensor.collapse_shape %{{.*}} [] : tensor<1xi32> into tensor<i32>

// -----

func.func @reshape_1D_0D_unsigned(%arg0: tensor<1xui32>) -> tensor<ui32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1xui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: func @reshape_1D_0D_unsigned
// CHECK-SAME:    %[[ARG_UNSIGNED:[a-zA-Z0-9_]*]]
// CHECK:         %[[ARG_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[ARG_UNSIGNED]] : tensor<1xui32> to tensor<1xi32>
// CHECK:         %[[RET_SIGNLESS:.*]] = tensor.collapse_shape %[[ARG_SIGNLESS]] [] : tensor<1xi32> into tensor<i32>
// CHECK:         %[[RET_UNSIGNED:.*]] = builtin.unrealized_conversion_cast %[[RET_SIGNLESS]] : tensor<i32> to tensor<ui32>
// CHECK:         return %[[RET_UNSIGNED]] : tensor<ui32>

// -----

// CHECK-LABEL: func @reshape_3D_2D
func.func @reshape_3D_2D(%arg0: tensor<12x1x42xi32>) -> tensor<12x42xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<12x1x42xi32>) -> tensor<12x42xi32>
  func.return %0 : tensor<12x42xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2]]

// -----

// CHECK-LABEL: func @reshape_4D_2D
func.func @reshape_4D_2D(%arg0: tensor<12x42x1x1xi32>) -> tensor<12x42xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<12x42x1x1xi32>) -> tensor<12x42xi32>
  func.return %0 : tensor<12x42xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_2D_4D
func.func @reshape_2D_4D(%arg0: tensor<12x42xi32>) -> tensor<12x1x42x1xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<12x42xi32>) -> tensor<12x1x42x1xi32>
  func.return %0 : tensor<12x1x42x1xi32>
}
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_3D_4D
func.func @reshape_3D_4D(%arg0: tensor<1x49x16xf32>) -> tensor<1x784x1x1xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x49x16xf32>) -> tensor<1x784x1x1xf32>
  func.return %0 : tensor<1x784x1x1xf32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2]]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]]

// -----

// CHECK-LABEL: func @reshape_4D_3D
func.func @reshape_4D_3D(%arg0: tensor<1x8x10x3xf32>) -> tensor<1x240x1xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x8x10x3xf32>) -> tensor<1x240x1xf32>
  func.return %0 : tensor<1x240x1xf32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2]

// -----

// CHECK-LABEL: func @reshape1_4D_4D
func.func @reshape1_4D_4D(%arg0: tensor<4x512x1x1xi32>) -> tensor<1x4x1x512xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<4x512x1x1xi32>) -> tensor<1x4x1x512xi32>
  func.return %0 : tensor<1x4x1x512xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]

// -----

// CHECK-LABEL: func @reshape2_4D_4D
func.func @reshape2_4D_4D(%arg0: tensor<4x1x1x1024xi32>) -> tensor<4x1024x1x1xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<4x1x1x1024xi32>) -> tensor<4x1024x1x1xi32>
  func.return %0 : tensor<4x1024x1x1xi32>
}
// CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0, 1, 2, 3]
// CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2, 3]

// -----

// CHECK-LABEL: func @reshape_dynamic_in
func.func @reshape_dynamic_in(%arg0: tensor<?x?xf32>) -> tensor<2x4x5xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<?x?xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}
// CHECK: %[[FLATTEN:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0, 1]] : tensor<?x?xf32> into tensor<?xf32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[FLATTEN]] : tensor<?xf32> to tensor<40xf32>
// CHECK: tensor.expand_shape %[[CAST]] {{\[}}[0, 1, 2]] : tensor<40xf32> into tensor<2x4x5xf32>

// -----

// CHECK-LABEL: func @reshape_1D_2D_dynamic
func.func @reshape_1D_2D_dynamic(%arg0: tensor<?xi32>) -> tensor<1x3xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<?xi32>) -> tensor<1x3xi32>
  func.return %0 : tensor<1x3xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?xi32> to tensor<3xi32>
// CHECK: tensor.expand_shape %[[CAST]] {{\[}}[0, 1]] : tensor<3xi32> into tensor<1x3xi32>

// -----

// CHECK-LABEL: func @reshape_2D_1D_dynamic
func.func @reshape_2D_1D_dynamic(%arg0: tensor<?x?xi32>) -> tensor<3xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<?x?xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
// CHECK: %[[FLATTEN:.*]] = tensor.collapse_shape %{{.*}} {{\[}}[0, 1]] : tensor<?x?xi32> into tensor<?xi32>
// CHECK: %[[CAST:.*]] = tensor.cast %[[FLATTEN]] : tensor<?xi32> to tensor<3xi32>
// CHECK: return %[[CAST:.*]] : tensor<3xi32>

// -----
// CHECK-LABEL: func @reshape_2D_1D_semidynamic
func.func @reshape_2D_1D_semidynamic(%arg0: tensor<1x?xi32>) -> tensor<1xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x?xi32>) -> tensor<1xi32>
  func.return %0 : tensor<1xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<1x?xi32> to tensor<1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}[0, 1]] : tensor<1x1xi32> into tensor<1xi32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<1xi32>

// -----

// CHECK-LABEL: func @reshape_1D_0D_dynamic
func.func @reshape_1D_0D_dynamic(%arg0: tensor<?xi32>) -> tensor<i32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<?xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?xi32> to tensor<1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}] : tensor<1xi32> into tensor<i32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_2D_0D_dynamic
func.func @reshape_2D_0D_dynamic(%arg0: tensor<?x?xi32>) -> tensor<i32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<?x?xi32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<?x?xi32> to tensor<1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}] : tensor<1x1xi32> into tensor<i32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<i32>

// -----

// CHECK-LABEL: func @reshape_3D_1D_semidynamic
func.func @reshape_3D_1D_semidynamic(%arg0: tensor<16x1x?xi32>) -> tensor<16xi32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<16x1x?xi32>) -> tensor<16xi32>
  func.return %0 : tensor<16xi32>
}
// CHECK: %[[CAST:.*]] = tensor.cast %{{.*}} : tensor<16x1x?xi32> to tensor<16x1x1xi32>
// CHECK: %[[COLLAPSE:.*]] = tensor.collapse_shape %[[CAST]] {{\[}}[0, 1, 2]] : tensor<16x1x1xi32> into tensor<16xi32>
// CHECK: return %[[COLLAPSE:.*]] : tensor<16xi32>

// -----

// CHECK-LABEL: func @minf
func.func @minf(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.minimum"(%lhs, %rhs) {someattr}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.minf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxi
func.func @maxi(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @maxu
func.func @maxu(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  %0 = "mhlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}
// CHECK: linalg.init_tensor [2, 2] : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxui %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-DAG: #[[MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @add_scalar
func.func @add_scalar(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[RESULT:.*]] = arith.addf %[[LHS]], %[[RHS]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

func.func @reshape_collapse_single_dim
  (%arg0: tensor<1x28x28x1xf32>) -> tensor<1x784xf32> {
  %0 = "mhlo.reshape"(%arg0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
  func.return %0 : tensor<1x784xf32>
}
// CHECK-LABEL: func @reshape_collapse_single_dim
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2, 3]]

// -----

func.func @reshape_collapse(%arg0: tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32>
    func.return %0 : tensor<2x4x3xf32>
}
// CHECK-LABEL: func @reshape_collapse
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2], [3]]

// -----

func.func @reshape_expand(%arg0: tensor<2x8xf32>) -> tensor<2x4x2xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    func.return %0 : tensor<2x4x2xf32>
}
// CHECK-LABEL: func @reshape_expand
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0], [1, 2]]

// -----

func.func @reshape_single_expand(%arg0 : tensor<8xf32>) -> tensor<1x4x2xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<8xf32>) -> tensor<1x4x2xf32>
    func.return %0 : tensor<1x4x2xf32>
}
// CHECK-LABEL: func @reshape_single_expand
//       CHECK: tensor.expand_shape %{{.*}} {{\[}}[0, 1, 2]]

// -----

func.func @reshape_multiple_collapse
  (%arg0 : tensor<1x2x2x5x3x2xf32>) -> tensor<1x4x5x6xf32> {
    %0 = "mhlo.reshape"(%arg0) : (tensor<1x2x2x5x3x2xf32>) -> tensor<1x4x5x6xf32>
    func.return %0 : tensor<1x4x5x6xf32>
}
// CHECK-LABEL: func @reshape_multiple_collapse
//       CHECK: tensor.collapse_shape %{{.*}} {{\[}}[0], [1, 2], [3], [4, 5]]


// -----

// CHECK-LABEL: func @bitcast_convert
func.func @bitcast_convert(%input: tensor<2x2xi32>) -> tensor<2x2xf32> {
  %result = "mhlo.bitcast_convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @bitcast_convert_dynamic
func.func @bitcast_convert_dynamic(%input: tensor<?x?xi32>) -> tensor<?x?xf32> {
  %result = "mhlo.bitcast_convert"(%input) : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i1_to_f32
func.func @convert_i1_to_f32(%input: tensor<2x2xi1>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi1>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i1, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.uitofp %[[OPERAND_IN]] : i1 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_ui8_to_f32
func.func @convert_ui8_to_f32(%input: tensor<2x2xui8>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xui8>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<2x2xui8> to tensor<2x2xi8>
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i8, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.uitofp %[[OPERAND_IN]] : i8 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i1_to_i32
func.func @convert_i1_to_i32(%input: tensor<2x2xi1>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi1>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i1, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.extui %[[OPERAND_IN]] : i1 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_f32
func.func @convert_i32_to_f32(%input: tensor<2x2xi32>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.sitofp %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_ui32_to_f32
func.func @convert_ui32_to_f32(%input: tensor<2x2xui32>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xui32>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.uitofp %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i16_to_i32
func.func @convert_i16_to_i32(%input: tensor<2x2xi16>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi16>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.extsi %[[OPERAND_IN]] : i16 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_ui16_to_i32
func.func @convert_ui16_to_i32(%input: tensor<2x2xui16>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xui16>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<2x2xui16> to tensor<2x2xi16>
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.extui %[[OPERAND_IN]] : i16 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_i16
func.func @convert_i32_to_i16(%input: tensor<2x2xi32>) -> tensor<2x2xi16> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xi16>
  func.return %result : tensor<2x2xi16>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: i16):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.trunci %[[OPERAND_IN]] : i32 to i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16

// -----

// CHECK-LABEL: func @convert_f32_to_f64
func.func @convert_f32_to_f64(%input: tensor<2x2xf32>) -> tensor<2x2xf64> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf64>
  func.return %result : tensor<2x2xf64>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: f64):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.extf %[[OPERAND_IN]] : f32 to f64
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f64

// -----

// CHECK-LABEL: func @convert_f64_to_f32
func.func @convert_f64_to_f32(%input: tensor<2x2xf64>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf64>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f64, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.truncf %[[OPERAND_IN]] : f64 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i32_to_i1
func.func @convert_i32_to_i1(%input: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi ne, %[[OPERAND_IN]], %[[ZERO]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_ui32_to_i1
func.func @convert_ui32_to_i1(%input: tensor<2x2xui32>) -> tensor<2x2xi1> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xui32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi ne, %[[OPERAND_IN]], %[[ZERO]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i1
func.func @convert_f32_to_i1(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[OPERAND_IN]], %[[ZERO]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i32
func.func @convert_f32_to_i32(%input: tensor<2x2xf32>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.fptosi %[[OPERAND_IN]] : f32 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_c64_to_c128
func.func @convert_c64_to_c128(%input: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f64>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f64>>
  func.return %result : tensor<2x2xcomplex<f64>>
}
// CHECK:      linalg.init_tensor
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %{{.*}}: complex<f64>):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK-DAG:  %[[IMAG:.*]] = complex.im %[[OPERAND_IN]]
// CHECK-DAG:  %[[REAL_RESULT:.*]] = arith.extf %[[REAL]] : f32 to f64
// CHECK-DAG:  %[[IMAG_RESULT:.*]] = arith.extf %[[IMAG]] : f32 to f64
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[REAL_RESULT]], %[[IMAG_RESULT]]
// CHECK:      linalg.yield %[[RESULT]] : complex<f64>

// CHECK-LABEL: func @convert_c128_to_c64
func.func @convert_c128_to_c64(%input: tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f32>>
  func.return %result : tensor<2x2xcomplex<f32>>
}
// CHECK:      linalg.init_tensor
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f64>, %{{.*}}: complex<f32>):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK-DAG:  %[[IMAG:.*]] = complex.im %[[OPERAND_IN]]
// CHECK-DAG:  %[[REAL_RESULT:.*]] = arith.truncf %[[REAL]] : f64 to f32
// CHECK-DAG:  %[[IMAG_RESULT:.*]] = arith.truncf %[[IMAG]] : f64 to f32
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[REAL_RESULT]], %[[IMAG_RESULT]]
// CHECK:      linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @convert_f32_to_c64
func.func @convert_f32_to_c64(%input: tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  func.return %result : tensor<2x2xcomplex<f32>>
}
// CHECK:      linalg.init_tensor
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: complex<f32>):
// CHECK-DAG:  %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[OPERAND_IN]], %[[IMAG]] 
// CHECK:      linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @convert_i32_to_c128
func.func @convert_i32_to_c128(%input: tensor<2x2xi32>) -> tensor<2x2xcomplex<f64>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xcomplex<f64>>
  func.return %result : tensor<2x2xcomplex<f64>>
}
// CHECK:      linalg.init_tensor
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: complex<f64>):
// CHECK-DAG:  %[[REAL:.*]] = arith.sitofp %[[OPERAND_IN]] : i32 to f64
// CHECK-DAG:  %[[IMAG:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[REAL]], %[[IMAG]] 
// CHECK:      linalg.yield %[[RESULT]] : complex<f64>

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reverse
func.func @reverse(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "mhlo.reverse"(%input) {
    dimensions = dense<1> : tensor<1xi64>, someattr
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %result : tensor<2x3xf32>
}
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: {someattr}

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_f32
func.func @iota_f32() -> tensor<7x10xf32> {
  %result = "mhlo.iota"() {iota_dimension = 1 : i64, someattr} : () -> (tensor<7x10xf32>)
  func.return %result : tensor<7x10xf32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_i32
func.func @iota_i32() -> tensor<7x10xi32> {
  %result = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xi32>)
  func.return %result : tensor<7x10xi32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_ui32
func.func @iota_ui32() -> tensor<7x10xui32> {
  %result = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xui32>)
  func.return %result : tensor<7x10xui32>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[INT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<7x10xi32> to tensor<7x10xui32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @iota_complexf32
func.func @iota_complexf32() -> tensor<7x10xcomplex<f32>> {
  %result = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> (tensor<7x10xcomplex<f32>>)
  func.return %result : tensor<7x10xcomplex<f32>>
}
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: complex<f32>):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[COMPLEX_CAST:.*]] = complex.create %[[FLOAT_CAST]], %[[ZERO]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[COMPLEX_CAST]] : complex<f32>

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @dynamic_iota_f32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dynamic_iota_f32(%shape: tensor<?xi32>) -> tensor<?x?x8xf32> {
  %result = "mhlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xf32>)
  func.return %result : tensor<?x?x8xf32>
}
// CHECK: %[[CAST:.*]] = arith.index_cast %arg0 : tensor<?xi32> to tensor<?xindex>
// CHECK: %[[I1:.*]] = tensor.extract %[[CAST]][%c0] : tensor<?xindex>
// CHECK: %[[I2:.*]] = tensor.extract %[[CAST]][%c1] : tensor<?xindex>
// CHECK: linalg.init_tensor [%[[I1]], %[[I2]], 8] : tensor<?x?x8xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: f32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : f32

// -----

// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @dyanmic_iota_ui32
// CHECK-SAME: %[[SHAPE:.*]]: tensor<?xi32>
func.func @dyanmic_iota_ui32(%shape: tensor<?xi32>) -> tensor<?x?x8xui32> {
  %result = "mhlo.dynamic_iota"(%shape) {iota_dimension = 1 : i64} : (tensor<?xi32>) -> (tensor<?x?x8xui32>)
  func.return %result : tensor<?x?x8xui32>
}
// CHECK: %[[CAST:.*]] = arith.index_cast %arg0 : tensor<?xi32> to tensor<?xindex>
// CHECK: %[[I1:.*]] = tensor.extract %[[CAST]][%c0] : tensor<?xindex>
// CHECK: %[[I2:.*]] = tensor.extract %[[CAST]][%c1] : tensor<?xindex>
// CHECK: linalg.init_tensor [%[[I1]], %[[I2]], 8] : tensor<?x?x8xi32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: i32):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   linalg.yield %[[FLOAT_CAST]] : i32
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<?x?x8xi32> to tensor<?x?x8xui32>

// -----

func.func @shift_left(%lhs: tensor<2x2xi32>,
                 %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.shift_left"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_left
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.shli %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_arithmetic(%lhs: tensor<2x2xi32>,
                             %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_arithmetic"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_arithmetic
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.shrsi %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_logical(%lhs: tensor<2x2xi32>,
                          %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_logical"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_logical
// CHECK: linalg.init_tensor
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.shrui %[[LHS]], %[[RHS]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @constant
func.func @constant() {
  %result = "mhlo.constant"() {
    value = dense<10> : tensor<i32>
  } : () -> (tensor<i32>)
  func.return
}
// CHECK: %[[CONSTANT:.*]] = arith.constant dense<10> : tensor<i32>

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_pow
func.func @float_pow(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = math.powf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.power"(%lhs, %rhs) : (tensor<2x2xf32>,
                                   tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_pow
func.func @complex_pow(%lhs: tensor<2x2xcomplex<f32>>,
                %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: complex<f32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: complex<f32>
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = complex.pow %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.power"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
                                   tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @integer_pow
func.func @integer_pow(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
                    // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: i32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: i32
  // CHECK: %[[FOR_RESULT:[a-zA-Z0-9_]*]]:3 = scf.for {{.*}} to %c6 step %c1
  // CHECK-SAME: iter_args(
  // CHECK-SAME:   %[[ITER0:.*]] = %c1
  // CHECK-SAME:   %[[ITER1:.*]] = %[[ARG0]]
  // CHECK-SAME:   %[[ITER2:.*]] = %[[ARG1]]
  // CHECK-SAME: ) -> (i32, i32, i32) {
  //   CHECK: %[[AND:[a-zA-Z0-9_]*]] = arith.andi %[[ITER2]], %c1
  //   CHECK: %[[COND:[a-zA-Z0-9_]*]] = arith.cmpi eq, %[[AND]], %c1
  //   CHECK: %[[MUL:[a-zA-Z0-9_]*]] = arith.muli %[[ITER0]], %[[ITER1]]
  //   CHECK: %[[ACCUM:[a-zA-Z0-9_]*]] = arith.select %[[COND]], %[[MUL]], %[[ITER0]]
  //   CHECK: %[[BASE:[a-zA-Z0-9_]*]] = arith.muli %[[ITER1]], %[[ITER1]]
  //   CHECK: %[[EXP:[a-zA-Z0-9_]*]] = arith.shrui %[[ITER2]], %c1
  //   CHECK: scf.yield %[[ACCUM]], %[[BASE]], %[[EXP]]
  // CHECK: %[[RHS_PARITY:.*]] = arith.remsi %[[ARG1]], %c2
  // CHECK: %[[RHS_EVEN:.*]] = arith.cmpi eq, %[[RHS_PARITY]], %c0
  // CHECK: %[[RHS_NEG:.*]] = arith.cmpi slt, %[[ARG1]], %c0
  // CHECK: %[[LHS_ONE:.*]] = arith.cmpi eq, %[[ARG0]], %c1
  // CHECK: %[[LHS_NEG_ONE:.*]] = arith.cmpi eq, %[[ARG0]], %c-1
  // CHECK: %[[VAL5:.*]] = arith.select %[[LHS_ONE]], %c1_i32, %c0
  // CHECK: %[[VAL6:.*]] = arith.select %[[RHS_EVEN]], %c1{{.*}}, %c-1
  // CHECK: %[[VAL7:.*]] = arith.select %[[LHS_NEG_ONE]], %[[VAL6]], %[[VAL5]]
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_NEG]], %[[VAL7]], %[[FOR_RESULT]]#0
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.power"(%lhs, %rhs) : (tensor<2x2xi32>,
                                   tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-SAME: [[SHAPE:%.*]]: tensor<1xindex>
func.func @dynamic_broadcast_in_dim(%shape: tensor<1xindex>) -> tensor<?xf32> {
  %cst = mhlo.constant dense<0x7F800000> : tensor<f32>
  %result = "mhlo.dynamic_broadcast_in_dim"(%cst, %shape) {
     broadcast_dimensions = dense<> : tensor<0xi64>, someattr
  } : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %result : tensor<?xf32>
}
// CHECK: [[CST:%.*]] = arith.constant
// CHECK: [[INIT:%.*]] = linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[CST]] : tensor<f32>) outs([[INIT]] : tensor<?xf32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-SAME: [[SCALAR:%.*]]: tensor<f32>
// CHECK-SAME: [[SHAPE:%.*]]: tensor<2xindex>
func.func @dynamic_broadcast_in_dim(%scalar: tensor<f32>, %shape: tensor<2xindex>)
    -> tensor<?x32xf32> {
  %result = "mhlo.dynamic_broadcast_in_dim"(%scalar, %shape) {
     broadcast_dimensions = dense<> : tensor<0xi64>
  } : (tensor<f32>, tensor<2xindex>) -> tensor<?x32xf32>
  func.return %result : tensor<?x32xf32>
}
// CHECK: [[INIT:%.*]] = linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[SCALAR]] : tensor<f32>) outs([[INIT]] : tensor<?x32xf32>)
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-SAME: [[VECTOR:%.*]]: tensor<42xf32>
// CHECK-SAME: [[SHAPE:%.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%vector: tensor<42xf32>, %shape: tensor<3xindex>)
    -> tensor<?x?x?xf32> {
  %result = "mhlo.dynamic_broadcast_in_dim"(%vector, %shape) {
     broadcast_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<42xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %result : tensor<?x?x?xf32>
}
// CHECK: [[INIT:%.*]] = linalg.init_tensor
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[VECTOR]] : tensor<42xf32>) outs([[INIT]] : tensor<?x?x?xf32>)
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// Note: this test requires no checks. The linalg init_tensor verifier will
// fail if the %shape i32 -> index cast is not performed properly.
func.func @dynamic_broadcast_in_dim(%scalar: tensor<f32>, %shape: tensor<2xi32>)
    -> tensor<?x32xf32> {
  %result = "mhlo.dynamic_broadcast_in_dim"(%scalar, %shape) {
     broadcast_dimensions = dense<> : tensor<0xi64>
  } : (tensor<f32>, tensor<2xi32>) -> tensor<?x32xf32>
  func.return %result : tensor<?x32xf32>
}

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-SAME: [[SHAPE:%.*]]: tensor<1xindex>, [[CSTARG:%.*]]: tensor<ui32>
func.func @dynamic_broadcast_in_dim(%shape: tensor<1xindex>, %cst: tensor<ui32>) -> tensor<?xui32> {
  %result = "mhlo.dynamic_broadcast_in_dim"(%cst, %shape) {
     broadcast_dimensions = dense<> : tensor<0xi64>
  } : (tensor<ui32>, tensor<1xindex>) -> tensor<?xui32>
  func.return %result : tensor<?xui32>
}
// CHECK: [[CST:%.*]] = builtin.unrealized_conversion_cast [[CSTARG]] : tensor<ui32> to tensor<i32>
// CHECK: [[INIT:%.*]] = linalg.init_tensor
// CHECK: [[GENERIC:%.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[CST]] : tensor<i32>) outs([[INIT]] : tensor<?xi32>)
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %[[RESULT:.*]]: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: [[RES:%.*]] = builtin.unrealized_conversion_cast [[GENERIC]] : tensor<?xi32> to tensor<?xui32>
// CHECK: return [[RES]] : tensor<?xui32>

// -----

// CHECK: #[[ARG_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (0, 0, d3, d4, 0, d6)>
// CHECK: #[[RES_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?x?x1x42xf32>, %[[SHAPE:.*]]: tensor<7xindex>
func.func @dynamic_broadcast_in_dim(%arg: tensor<?x?x?x?x1x42xf32>,
    %shape: tensor<7xindex>) -> tensor<?x?x?x?x?x?x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[C2:.*]] = arith.constant 2
  // CHECK-DAG:  %[[C3:.*]] = arith.constant 3
  // CHECK-DAG:  %[[C4:.*]] = arith.constant 4
  // CHECK-DAG:  %[[C5:.*]] = arith.constant 5
  // CHECK-DAG:  %[[C6:.*]] = arith.constant 6
  // CHECK-DAG:  %[[DIM0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG:  %[[DIM1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG:  %[[DIM2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG:  %[[DIM3:.*]] = tensor.extract %[[SHAPE]][%[[C3]]]
  // CHECK-DAG:  %[[DIM4:.*]] = tensor.extract %[[SHAPE]][%[[C4]]]
  // CHECK-DAG:  %[[DIM5:.*]] = tensor.extract %[[SHAPE]][%[[C5]]]
  // CHECK-DAG:  %[[DIM6:.*]] = tensor.extract %[[SHAPE]][%[[C6]]]
  // CHECK-DAG:  %[[INIT:.*]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[DIM4]], %[[DIM5]], %[[DIM6]]] : tensor<?x?x?x?x?x?x?xf32>
  // CHECK:      %[[RES:.*]] = linalg.generic {
  // CHECK-SAME:     indexing_maps = [#[[ARG_MAP]], #[[RES_MAP]]],
  // CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME:     ins(%[[ARG]] : tensor<?x?x?x?x1x42xf32>) outs(%[[INIT]] : tensor<?x?x?x?x?x?x?xf32>) {
  // CHECK:      ^bb0(%[[ARG_:.*]]: f32, %{{.*}}: f32):
  // CHECK:        linalg.yield %[[ARG_]]
  // CHECK:      return %[[RES]]
  %result = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape) {
      broadcast_dimensions = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>,
      known_expanding_dimensions = dense<[0, 1]> : tensor<2xi64>,
      known_nonexpanding_dimensions = dense<[2, 3]> : tensor<2xi64> }
      : (tensor<?x?x?x?x1x42xf32>, tensor<7xindex>) -> tensor<?x?x?x?x?x?x?xf32>
  func.return %result : tensor<?x?x?x?x?x?x?xf32>
}

// -----

func.func @dot_matmul(%arg0: tensor<2x3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<2x?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xf32>, tensor<3x?xf32>) -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}
// CHECK-LABEL: func @dot_matmul(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, %[[D1]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xf32>)

// -----

func.func @dot_matmul_complex(%arg0: tensor<2x3xcomplex<f32>>,
                 %arg1: tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>> {
  %0 = "mhlo.dot"(%arg0, %arg1) {someattr}
           : (tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>) -> tensor<2x?xcomplex<f32>>
  func.return %0 : tensor<2x?xcomplex<f32>>
}
// CHECK-LABEL: func @dot_matmul_complex(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xcomplex<f32>>, %[[ARG1:.*]]: tensor<3x?xcomplex<f32>>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, %[[D1]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xcomplex<f32>>, tensor<3x?xcomplex<f32>>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xcomplex<f32>>)

// -----

func.func @dot_matmul_i8_i8_i32(%arg0: tensor<2x3xi8>,
                 %arg1: tensor<3x?xi8>) -> tensor<2x?xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3xi8>,
                                   tensor<3x?xi8>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i8_i8_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi8>, %[[ARG1:.*]]: tensor<3x?xi8>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, %[[D1]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi8>, tensor<3x?xi8>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i16_i16_i32(%arg0: tensor<2x3xi16>,
                 %arg1: tensor<3x?xi16>) -> tensor<2x?xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3xi16>,
                                   tensor<3x?xi16>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i16_i16_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi16>, %[[ARG1:.*]]: tensor<3x?xi16>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, %[[D1]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi16>, tensor<3x?xi16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matmul_i32_i32_i32(%arg0: tensor<2x3xi32>,
                 %arg1: tensor<3x?xi32>) -> tensor<2x?xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x3xi32>,
                                   tensor<3x?xi32>) -> tensor<2x?xi32>
  func.return %0 : tensor<2x?xi32>
}
// CHECK-LABEL: func @dot_matmul_i32_i32_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x3xi32>, %[[ARG1:.*]]: tensor<3x?xi32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, %[[D1]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x3xi32>, tensor<3x?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x?xi32>)

// -----

func.func @dot_matvec(%arg0: tensor<?x3xf32>,
                 %arg1: tensor<3xf32>) -> tensor<?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>,
                                   tensor<3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_matvec(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x3xf32>, %[[ARG1:.*]]: tensor<3xf32>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D0]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.matvec
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x3xf32>, tensor<3xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_vecmat(%arg0: tensor<3xf32>,
                 %arg1: tensor<3x?xf32>) -> tensor<?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3xf32>,
                                   tensor<3x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @dot_vecmat(
// CHECK-SAME: %[[ARG0:.*]]: tensor<3xf32>, %[[ARG1:.*]]: tensor<3x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG1]], %[[C1]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D1]]]
// CHECK: linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.vecmat
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3xf32>, tensor<3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?xf32>)

// -----

func.func @dot_dot(%arg0: tensor<?xf32>,
              %arg1: tensor<?xf32>) -> tensor<f32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: func @dot_dot(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xf32>, %[[ARG1:.*]]: tensor<?xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor []
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?xf32>, tensor<?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<f32>)

// -----

func.func @dot_dot_unsigned(%arg0: tensor<?xui32>,
              %arg1: tensor<?xui32>) -> tensor<ui32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?xui32>, tensor<?xui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: func @dot_dot_unsigned(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xui32>, %[[ARG1:.*]]: tensor<?xui32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor []
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}outs(%[[INIT]]
// CHECK: linalg.dot
// CHECK-SAME: ins(%{{.*}} : tensor<?xi32>, tensor<?xi32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<i32>)

// -----

func.func @dot_general_batch_matmul(%arg0: tensor<?x?x3xf32>,
                  %arg1: tensor<?x3x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<?x?x3xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @dot_general_batch_matmul(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xf32>, %[[ARG1:.*]]: tensor<?x3x?xf32>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.batch_matmul
// CHECK-SAME: {someattr}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x3xf32>, tensor<?x3x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)

// -----

func.func @dot_general_batch_matmul_unsigned(%arg0: tensor<?x?x3xui32>,
                  %arg1: tensor<?x3x?xui32>) -> tensor<?x?x?xui32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<?x?x3xui32>, tensor<?x3x?xui32>) -> tensor<?x?x?xui32>
  func.return %0 : tensor<?x?x?xui32>
// CHECK-LABEL: func @dot_general_batch_matmul_unsigned(
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins({{.*}} : tensor<?x?x3xi32>, tensor<?x3x?xi32>)
// CHECK-SAME: outs({{.*}} : tensor<?x?x?xi32>)
}

// -----

func.func @dot_general_batch_matmul_i8_i8_i32(%arg0: tensor<?x?x3xi8>,
                  %arg1: tensor<?x3x?xi8>) -> tensor<?x?x?xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<?x?x3xi8>, tensor<?x3x?xi8>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_i8_i8_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xi8>, %[[ARG1:.*]]: tensor<?x3x?xi8>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x3xi8>, tensor<?x3x?xi8>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xi32>)

// -----

func.func @dot_general_batch_matmul_i16_i16_i32(%arg0: tensor<?x?x3xi16>,
                  %arg1: tensor<?x3x?xi16>) -> tensor<?x?x?xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]
  } : (tensor<?x?x3xi16>, tensor<?x3x?xi16>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_i16_i16_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xi16>, %[[ARG1:.*]]: tensor<?x3x?xi16>)
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.batch_matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x3xi16>, tensor<?x3x?xi16>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xi32>)

// -----

func.func @dot_general_batch_matmul_large
  (%arg0: tensor<2x16x32xf32>, %arg1: tensor<2x32x32xf32>) -> tensor<2x16x32xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]}
    : (tensor<2x16x32xf32>, tensor<2x32x32xf32>) -> tensor<2x16x32xf32>
  func.return %0 : tensor<2x16x32xf32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_large(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: tensor<2x16x32xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: tensor<2x32x32xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor [2, 16, 32]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: %[[DOT:.*]] = linalg.batch_matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x16x32xf32>, tensor<2x32x32xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x16x32xf32>)

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_basic
func.func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm", someattr}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x5x6xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor [3, 4, 6] : tensor<3x4x6xf32>
// CHECk: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x5x6xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x6xf32>)
// CHECK-SAME: {someattr}
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @einsum_pointwisemul
func.func @einsum_pointwisemul(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "abc,abc->abc"} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x4x5xf32>)
// CHECK: linalg.init_tensor [3, 4, 5] : tensor<3x4x5xf32>
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP0]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5xf32>, tensor<3x4x5xf32>)
// CHECK-SAME: outs(%[[DST:.+]] : tensor<3x4x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[RES:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: func @einsum_matmul
func.func @einsum_matmul(%arg0: tensor<7x9xf32>, %arg1: tensor<9x5xf32>) -> tensor<7x5xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ae,ed->ad"}: (tensor<7x9xf32>, tensor<9x5xf32>) -> tensor<7x5xf32>
  func.return %0 : tensor<7x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<7x9xf32>, %[[RHS:.*]]: tensor<9x5xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor [7, 5] : tensor<7x5xf32>
// CHECk: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<7x9xf32>, tensor<9x5xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<7x5xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d5)>
// CHECK: func @einsum_broadcast4
func.func @einsum_broadcast4(%arg0: tensor<3x4x5x6x7xf32>, %arg1: tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "abcdh,hg->abcdg"}: (tensor<3x4x5x6x7xf32>, tensor<7x8xf32>) -> tensor<3x4x5x6x8xf32>
  func.return %0 : tensor<3x4x5x6x8xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5x6x7xf32>, %[[RHS:.*]]: tensor<7x8xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor [3, 4, 5, 6, 8] : tensor<3x4x5x6x8xf32>
// CHECk: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<3x4x5x6x7xf32>, tensor<7x8xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<3x4x5x6x8xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_ellipsis
func.func @einsum_ellipsis(%arg0: tensor<1x512x128xf32>, %arg1: tensor<128x256xf32>) -> tensor<1x512x256xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "...x,xy->...y"} : (tensor<1x512x128xf32>, tensor<128x256xf32>) -> tensor<1x512x256xf32>
  func.return %0 : tensor<1x512x256xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<1x512x128xf32>, %[[RHS:.*]]: tensor<128x256xf32>)
// CHECK: %[[INIT:.*]] = linalg.init_tensor [1, 512, 256] : tensor<1x512x256xf32>
// CHECk: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<1x512x128xf32>, tensor<128x256xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<1x512x256xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK: func @einsum_dynamic_size_broadcast_dot
func.func @einsum_dynamic_size_broadcast_dot(%arg0: tensor<?x?x4xf32>, %arg1: tensor<4x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "abc,cd->abd"} : (tensor<?x?x4xf32>, tensor<4x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<?x?x4xf32>, %[[RHS:.*]]: tensor<4x?xf32>)
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[DIM0:.+]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x4xf32>
// CHECK: %[[C1:.+]] = arith.constant 1 : index
// CHECK: %[[DIM1:.+]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x4xf32>
// CHECK: %[[C2:.+]] = arith.constant 1 : index
// CHECK: %[[DIM2:.+]] = tensor.dim %[[RHS]], %[[C2:.+]] : tensor<4x?xf32>
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[DIM0]], %[[DIM1]], %[[DIM2]]] : tensor<?x?x?xf32>
// CHECk: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[LHS]], %[[RHS]] : tensor<?x?x4xf32>, tensor<4x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK: ^bb0(%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %[[OUT_:.*]]: f32):
// CHECK:   %[[MUL:.*]] = arith.mulf %[[LHS_]], %[[RHS_]] : f32
// CHECK:   %[[RES:.*]] = arith.addf %[[OUT_]], %[[MUL]] : f32
// CHECK:   linalg.yield %[[RES]]

// -----

// CHECK-LABEL: @clamp
// CHECK-SAME: %[[LB:.*]]: tensor<4xf32>, %[[X:.*]]: tensor<4xf32>, %[[UB:.*]]: tensor<4xf32>
func.func @clamp(%lb : tensor<4xf32>, %x : tensor<4xf32>, %ub : tensor<4xf32>)
    -> tensor<4xf32> {
  // CHECK: %[[INIT:.*]] = linalg.init_tensor
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<4xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MIN:.*]] = arith.minf %[[SCALAR_X]], %[[SCALAR_UB]] : f32
  // CHECK:   %[[MAX_X2_LB:.*]] = arith.maxf %[[MIN]], %[[SCALAR_LB]] : f32
  // CHECK:   linalg.yield %[[MAX_X2_LB]]
  // CHECK: } -> tensor<4xf32>
  // CHECK: return %[[RESULT]] : tensor<4xf32>
  %0 = "mhlo.clamp"(%lb, %x, %ub) : (tensor<4xf32>, tensor<4xf32>,
      tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @clamp_dynamic
// CHECK-SAME: %[[LB:.*]]: tensor<?xf32>, %[[X:.*]]: tensor<?xf32>, %[[UB:.*]]: tensor<?xf32>
func.func @clamp_dynamic(%lb : tensor<?xf32>, %x : tensor<?xf32>, %ub : tensor<?xf32>)
    -> tensor<?xf32> {
  // CHECK: %[[INIT:.*]] = linalg.init_tensor
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%[[INIT]] : tensor<?xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MIN:.*]] = arith.minf %[[SCALAR_X]], %[[SCALAR_UB]] : f32
  // CHECK:   %[[MAX_X2_LB:.*]] = arith.maxf %[[MIN]], %[[SCALAR_LB]] : f32
  // CHECK:   linalg.yield %[[MAX_X2_LB]]
  // CHECK: } -> tensor<?xf32>
  // CHECK: return %[[RESULT]] : tensor<?xf32>
  %0 = "mhlo.clamp"(%lb, %x, %ub) : (tensor<?xf32>, tensor<?xf32>,
      tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @reduce_add(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<5xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<5x4xi32>, tensor<i32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_add
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_add_complex(%arg0: tensor<5x4xcomplex<f32>>) -> tensor<5xcomplex<f32>> {
  %cst = mhlo.constant dense<(0.000000e+00,0.000000e+00)> : tensor<complex<f32>>
  %0 = "mhlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg3: tensor<complex<f32>>, %arg4 : tensor<complex<f32>>):
    %1 = mhlo.add %arg3, %arg4 : tensor<complex<f32>>
    "mhlo.return"(%1) : (tensor<complex<f32>>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<5x4xcomplex<f32>>, tensor<complex<f32>>) -> tensor<5xcomplex<f32>>
  func.return %0 : tensor<5xcomplex<f32>>
}

// CHECK-LABEL: @reduce_add_complex
// CHECK: %[[INIT:.*]] = complex.constant [0.000000e+00 : f32, 0.000000e+00 : f32] : complex<f32>
// CHECK: linalg.fill ins(%[[INIT]]
// CHECK: complex.add

// -----

func.func @reduce_minimum_unsigned(%arg0: tensor<5x4xui32>, %arg1: tensor<ui32>) -> tensor<5xui32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<ui32>, %arg4 : tensor<ui32>):
    %1 = mhlo.minimum %arg3, %arg4 : tensor<ui32>
    "mhlo.return"(%1) : (tensor<ui32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<5x4xui32>, tensor<ui32>) -> tensor<5xui32>
  func.return %0 : tensor<5xui32>
}
// CHECK-LABEL: @reduce_minimum_unsigned
// CHECK: linalg.generic
// CHECK-NEXT: ^bb{{.}}(%[[ARG2:.+]]: i32, %[[ARG3:.+]]: i32)
// CHECK-NEXT: arith.minui %[[ARG2]], %[[ARG3]] : i32

// -----

func.func @reduce_minimum(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<5xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.minimum %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_minimum
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.minsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_maximum(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<5xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.maximum %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_maximum
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_and(%arg0: tensor<5x4xi1>, %arg1: tensor<i1>) -> tensor<5xi1> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i1>, %arg4 : tensor<i1>):
    %1 = mhlo.and %arg3, %arg4 : tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xi1>, tensor<i1>) -> tensor<5xi1>
  func.return %0 : tensor<5xi1>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_and
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i1>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi1>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi1>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i1, %[[RHS_IN:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.andi %[[LHS_IN]], %[[RHS_IN]] : i1
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

func.func @reduce_or(%arg0: tensor<5x4xi1>, %arg1: tensor<i1>) -> tensor<5xi1> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i1>, %arg4 : tensor<i1>):
    %1 = mhlo.or %arg3, %arg4 : tensor<i1>
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xi1>, tensor<i1>) -> tensor<5xi1>
  func.return %0 : tensor<5xi1>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_or
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i1>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi1>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi1>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i1, %[[RHS_IN:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.ori %[[LHS_IN]], %[[RHS_IN]] : i1
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

func.func @reduce_or_of_zero_constant() -> tensor<5xi1> {
  %0 = mhlo.constant dense<false> : tensor<5x4xi1>
  %1 = mhlo.constant dense<false> : tensor<i1>
  %2 = "mhlo.reduce"(%0, %1) ({
  ^bb0(%arg1: tensor<i1>, %arg2 : tensor<i1>):
    "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<5x4xi1>, tensor<i1>) -> tensor<5xi1>
  func.return %2 : tensor<5xi1>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_or_of_zero_constant
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<false> : tensor<5x4xi1>
// CHECK-DAG: %[[INIT:.*]] = arith.constant dense<false> : tensor<i1>
// CHECK-DAG: %[[FALSE:.*]] = arith.constant false
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [5]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[FALSE]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi1>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi1>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i1, %[[RHS_IN:.*]]: i1):
// CHECK-NEXT:   %[[EXTRACT:.*]] = tensor.extract %[[INIT]][] : tensor<i1>
// CHECK-NEXT:   linalg.yield %[[EXTRACT]] : i1

// -----

func.func @reduce_dim0(%arg0: tensor<5x4xi32>, %arg1: tensor<i32>) -> tensor<4xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.maximum %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<5x4xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_dim0
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [4]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_init_const(%arg0: tensor<1x10xf32>) -> tensor<1xf32> {
  %cst = arith.constant dense<0xFF800000> : tensor<f32>
  %0 = "mhlo.reduce"(%arg0, %cst) ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    %1 = mhlo.add %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<f32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: @reduce_init_const
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [1]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<1x10xf32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<1xf32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

func.func @reduce_multi_dimensions(%arg0: tensor<5x4x3xi32>,
                              %arg1: tensor<i32>) -> tensor<4xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<5x4x3xi32>, tensor<i32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-LABEL: @reduce_multi_dimensions
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [4]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4x3xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}
// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: func @reduce_dynamic(%[[ARG0:.*]]: tensor<?x?xi32>
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xi32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = linalg.init_tensor [%[[DIM1]]]
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<?x?xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<?xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @variadic_reduce(%arg0: tensor<9x2xi32>, %arg1: tensor<9x2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %cst0 = mhlo.constant dense<-2147483648> : tensor<i32>
  %cst1 = mhlo.constant dense<0> : tensor<i32>
  %res0, %res1 = "mhlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg15: tensor<i32>, %arg16: tensor<i32>):
    %669 = "mhlo.compare"(%arg2, %arg15) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %670 = "mhlo.select"(%669, %arg2, %arg15) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %671 = "mhlo.compare"(%arg2, %arg15) {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %672 = mhlo.minimum %arg3, %arg16 : tensor<i32>
    %673 = "mhlo.select"(%669, %arg3, %arg16) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %674 = "mhlo.select"(%671, %672, %673) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%670, %674) : (tensor<i32>, tensor<i32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<9x2xi32>, tensor<9x2xi32>, tensor<i32>, tensor<i32>) -> (tensor<2xi32>, tensor<2xi32>)
  func.return %res0, %res1 : tensor<2xi32>, tensor<2xi32>
}
// CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func @variadic_reduce
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:        %[[CST0:.*]] = arith.constant -2147483648 : i32
// CHECK:        %[[INIT0:.*]] = linalg.init_tensor [2] : tensor<2xi32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[CST1:.*]] = arith.constant 0 : i32
// CHECK:        %[[INIT1:.*]] = linalg.init_tensor [2] : tensor<2xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<9x2xi32>, tensor<9x2xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<2xi32>, tensor<2xi32>)
// CHECK-NEXT:   ^bb0(%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %[[OUT0:.*]]: i32, %[[OUT1:.*]]: i32):
// CHECK-NEXT:     %[[T1:.*]] = arith.cmpi sge, %[[IN0]], %[[OUT0]] : i32
// CHECK-NEXT:     %[[T2:.*]] = arith.select %[[T1]], %[[IN0]], %[[OUT0]] : i32
// CHECK-NEXT:     %[[T3:.*]] = arith.cmpi eq, %[[IN0]], %[[OUT0]] : i32
// CHECK-NEXT:     %[[T4:.*]] = arith.minsi %[[IN1:.*]], %[[OUT1]] : i32
// CHECK-NEXT:     %[[T5:.*]] = arith.select %[[T1]], %[[IN1]], %[[OUT1]] : i32
// CHECK-NEXT:     %[[T6:.*]] = arith.select %[[T3]], %[[T4]], %[[T5]] : i32
// CHECK-NEXT:     linalg.yield %[[T2]], %[[T6]]

// -----

func.func @variadic_diff_type_reduce(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128xf32>, tensor<128xi32>) {
  %cst0 = mhlo.constant dense<1.0> : tensor<f32>
  %cst1 = mhlo.constant dense<1> : tensor<i32>
  %res0, %res1 = "mhlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<f32>, %arg10: tensor<i32>):
    %0 = "mhlo.compare"(%arg7, %arg9) {comparison_direction = #mhlo<"comparison_direction GE">} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = "mhlo.select"(%0, %arg7, %arg9) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.select"(%0, %arg8, %arg10) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<128x10xf32>, tensor<128x10xi32>, tensor<f32>, tensor<i32>) ->(tensor<128xf32>, tensor<128xi32>)
  func.return %res0, %res1 : tensor<128xf32>, tensor<128xi32>
}
// CHECK-DAG:  #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK:      func @variadic_diff_type_reduce
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:        %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:        %[[INIT0:.*]] = linalg.init_tensor [128] : tensor<128xf32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:        %[[INIT1:.*]] = linalg.init_tensor [128] : tensor<128xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<128x10xf32>, tensor<128x10xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<128xf32>, tensor<128xi32>)
// CHECK-NEXT:   ^bb0(%[[LHS0:.*]]: f32, %[[LHS1:.*]]: i32, %[[RHS0:.*]]: f32, %[[RHS1:.*]]: i32):
// CHECK-NEXT:      %[[B0:.*]] = arith.cmpf oge, %[[LHS0]], %[[RHS0]] : f32
// CHECK-NEXT:      %[[RES0:.*]] = arith.select %[[B0]], %[[LHS0]], %[[RHS0]] : f32
// CHECK-NEXT:      %[[RES1:.*]] = arith.select %[[B0]], %[[LHS1]], %[[RHS1]] : i32
// CHECK-NEXT:      linalg.yield %[[RES0]], %[[RES1]] : f32, i32

// -----

func.func @slice_whole_stride(%arg0: tensor<3x4xi32>) -> tensor<1x4xi32> {
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}
// CHECK-LABEL: func @slice_whole_stride
//       CHECK:   tensor.extract_slice %{{.*}}[1, 0] [1, 4] [1, 1] : tensor<3x4xi32> to tensor<1x4xi32>

// -----

func.func @slice_stride_part(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 3]> : tensor<2xi64>,
    strides = dense<1> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}
// CHECK-LABEL: func @slice_stride_part
//       CHECK:   tensor.extract_slice %{{.*}}[1, 1] [1, 2] [1, 1]  : tensor<3x4xi32> to tensor<1x2xi32>

// -----

func.func @slice_with_strides(%arg0: tensor<13xi32>) -> tensor<6xi32> {
  %0 = "mhlo.slice"(%arg0) {
    limit_indices = dense<12> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (tensor<13xi32>) -> tensor<6xi32>
  func.return %0 : tensor<6xi32>
}
// CHECK-LABEL: func @slice_with_strides
//       CHECK:   tensor.extract_slice %{{.*}}[0] [6] [2] : tensor<13xi32> to tensor<6xi32>

// -----

func.func @slice_with_strides2(%arg0: tensor<6xi32>) -> tensor<3xi32> {
  %0 = "mhlo.slice"(%arg0) {
    limit_indices = dense<5> : tensor<1xi64>,
    start_indices = dense<0> : tensor<1xi64>,
    strides = dense<2> : tensor<1xi64>
  } : (tensor<6xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
// CHECK-LABEL: func @slice_with_strides
//       CHECK:   tensor.extract_slice %{{.*}}[0] [3] [2] : tensor<6xi32> to tensor<3xi32>

// -----

func.func @dynamic_slice(%arg: tensor<3x4xf32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xf32> {
  %0 = "mhlo.dynamic-slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func @dynamic_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.*]] = arith.constant 0 : i64
// CHECK:         %[[SCALAR1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[UB1:.*]] = arith.constant 2 : i64
// CHECK:         %[[T1:.*]] = arith.minsi %[[SCALAR1]], %[[UB1]] : i64
// CHECK:         %[[CLAMPED1:.*]] = arith.maxsi %[[T1]], %[[C0]] : i64
// CHECK:         %[[START1:.*]] = arith.index_cast %[[CLAMPED1]] : i64 to index
// CHECK:         %[[SCALAR2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[UB2:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.minsi %[[SCALAR2]], %[[UB2]] : i64
// CHECK:         %[[CLAMPED2:.*]] = arith.maxsi %[[T2]], %[[C0]] : i64
// CHECK:         %[[START2:.*]] = arith.index_cast %[[CLAMPED2]] : i64 to index
// CHECK:           tensor.extract_slice %[[ARG0]][%[[START1]], %[[START2]]] [1, 4] [1, 1]

// -----

func.func @dynamic_slice_unsigned(%arg: tensor<3x4xui32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xui32> {
  %0 = "mhlo.dynamic-slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xui32>, tensor<i64>, tensor<i64>) -> tensor<1x4xui32>
  func.return %0 : tensor<1x4xui32>
}

// CHECK-LABEL: func @dynamic_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[SIGNLESS_ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x4xui32> to tensor<3x4xi32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : i64
// CHECK:         %[[SCALAR1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[UB1:.*]] = arith.constant 2 : i64
// CHECK:         %[[T1:.*]] = arith.minsi %[[SCALAR1]], %[[UB1]] : i64
// CHECK:         %[[CLAMPED1:.*]] = arith.maxsi %[[T1]], %[[C0]] : i64
// CHECK:         %[[START1:.*]] = arith.index_cast %[[CLAMPED1]] : i64 to index
// CHECK:         %[[SCALAR2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[UB2:.*]] = arith.constant 0 : i64
// CHECK:         %[[T2:.*]] = arith.minsi %[[SCALAR2]], %[[UB2]] : i64
// CHECK:         %[[CLAMPED2:.*]] = arith.maxsi %[[T2]], %[[C0]] : i64
// CHECK:         %[[START2:.*]] = arith.index_cast %[[CLAMPED2]] : i64 to index
// CHECK:           tensor.extract_slice %[[SIGNLESS_ARG0]][%[[START1]], %[[START2]]] [1, 4] [1, 1]

// -----

func.func @dynamic_update_slice(%target: tensor<3x3xi32>, %update: tensor<2x2xi32>, %c0: tensor<i32>) -> tensor<3x3xi32> {
  %0 = "mhlo.dynamic-update-slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}
// CHECK-LABEL: func @dynamic_update_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         %[[SCALAR1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB1:.*]] = arith.constant 1 : i32
// CHECK:         %[[T1:.*]] = arith.minsi %[[SCALAR1]], %[[UB1]] : i32
// CHECK:         %[[CLAMPED1:.*]] = arith.maxsi %[[T1]], %[[C0]] : i32
// CHECK:         %[[START1:.*]] = arith.index_cast %[[CLAMPED1]] : i32 to index
// CHECK:         %[[SCALAR2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB2:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = arith.minsi %[[SCALAR2]], %[[UB2]] : i32
// CHECK:         %[[CLAMPED2:.*]] = arith.maxsi %[[T2]], %[[C0]] : i32
// CHECK:         %[[START2:.*]] = arith.index_cast %[[CLAMPED2]] : i32 to index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[START1]], %[[START2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         return %[[RES]] : tensor<3x3xi32>

// -----

func.func @dynamic_update_slice_unsigned(%target: tensor<3x3xui32>, %update: tensor<2x2xui32>, %c0: tensor<i32>) -> tensor<3x3xui32> {
  %0 = "mhlo.dynamic-update-slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xui32>, tensor<2x2xui32>, tensor<i32>, tensor<i32>) -> tensor<3x3xui32>
  func.return %0 : tensor<3x3xui32>
}
// CHECK-LABEL: func @dynamic_update_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[SIGNLESS_UPDATE:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK-DAG:     %[[SIGNLESS_TARGET:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x3xui32> to tensor<3x3xi32>
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         %[[SCALAR1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB1:.*]] = arith.constant 1 : i32
// CHECK:         %[[T1:.*]] = arith.minsi %[[SCALAR1]], %[[UB1]] : i32
// CHECK:         %[[CLAMPED1:.*]] = arith.maxsi %[[T1]], %[[C0]] : i32
// CHECK:         %[[START1:.*]] = arith.index_cast %[[CLAMPED1]] : i32 to index
// CHECK:         %[[SCALAR2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB2:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = arith.minsi %[[SCALAR2]], %[[UB2]] : i32
// CHECK:         %[[CLAMPED2:.*]] = arith.maxsi %[[T2]], %[[C0]] : i32
// CHECK:         %[[START2:.*]] = arith.index_cast %[[CLAMPED2]] : i32 to index
// CHECK:         %[[SIGNLESS_RES:.*]] = tensor.insert_slice %[[SIGNLESS_UPDATE]] into %[[SIGNLESS_TARGET]]
// CHECK-SAME:      [%[[START1]], %[[START2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         %[[RES:.*]] = builtin.unrealized_conversion_cast %[[SIGNLESS_RES]] : tensor<3x3xi32> to tensor<3x3xui32>
// CHECK:         return %[[RES]] : tensor<3x3xui32>

// -----

func.func @dynamic_update_slice_float(%target: tensor<3x3xf32>,
                                 %update: tensor<2x2xf32>,
                                 %c0: tensor<i32>) -> tensor<3x3xf32> {
  %0 = "mhlo.dynamic-update-slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @dynamic_update_slice_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.*]] = arith.constant 0 : i32
// CHECK:         %[[SCALAR1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB1:.*]] = arith.constant 1 : i32
// CHECK:         %[[T1:.*]] = arith.minsi %[[SCALAR1]], %[[UB1]] : i32
// CHECK:         %[[CLAMPED1:.*]] = arith.maxsi %[[T1]], %[[C0]] : i32
// CHECK:         %[[START1:.*]] = arith.index_cast %[[CLAMPED1]] : i32 to index
// CHECK:         %[[SCALAR2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[UB2:.*]] = arith.constant 1 : i32
// CHECK:         %[[T2:.*]] = arith.minsi %[[SCALAR2]], %[[UB2]] : i32
// CHECK:         %[[CLAMPED2:.*]] = arith.maxsi %[[T2]], %[[C0]] : i32
// CHECK:         %[[START2:.*]] = arith.index_cast %[[CLAMPED2]] : i32 to index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[START1]], %[[START2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xf32> into tensor<3x3xf32>
// CHECK:         return %[[RES]] : tensor<3x3xf32>

// -----

func.func @pad_cst(%arg0: tensor<12x4xf32>) -> tensor<18x12xf32> {
  %0 = arith.constant dense<0.0> : tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %1 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_cst
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00 : f32
//       CHECK: tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:  tensor.yield %[[CST]] : f32
//       CHECK: } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_tensor(%arg0: tensor<12x4xf32>, %arg1: tensor<f32>) -> tensor<18x12xf32> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<0> : tensor<2xi64>
  } : (tensor<12x4xf32>, tensor<f32>) -> tensor<18x12xf32>
  func.return %0 : tensor<18x12xf32>
}
// CHECK-LABEL: func @pad_tensor
//  CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG:   %[[PAD:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
//       CHECK:   tensor.pad %[[ARG0]] low[4, 5] high[2, 3]
//       CHECK:     tensor.yield %[[PAD]] : f32
//       CHECK:   } : tensor<12x4xf32> to tensor<18x12xf32>

// -----

func.func @pad_interior(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<29x15xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, 5]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<29x15xui32>
  func.return %1 : tensor<29x15xui32>
}
// CHECK-LABEL: func @pad_interior
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
//  CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
//   CHECK-DAG: %[[CAST0:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<12x4xui32> to tensor<12x4xi32>
//   CHECK-DAG: %[[CAST1:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<ui32> to tensor<i32>
//   CHECK-DAG: %[[PAD:.+]] = tensor.extract %[[CAST1]][] : tensor<i32>
//       CHECK: %[[INIT:.+]] = linalg.init_tensor [29, 15] : tensor<29x15xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PAD]] : i32) outs(%[[INIT]] : tensor<29x15xi32>) -> tensor<29x15xi32>
//       CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[CAST0]] into %[[FILL]][4, 5] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x15xi32>

// -----

func.func @linalg.conv_0d_nc(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> {

  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">]} : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @linalg.conv_0d_nc
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00
// CHECK-DAG: %[[INIT:.+]] = linalg.init_tensor [3, 3]
// CHECK-DAG: %[[FILL:.+]] = linalg.fill ins(%cst{{.*}}outs(%[[INIT]]
// CHECK-DAG: %[[PAD:.+]] = tensor.pad %arg0 low[0, 0] high[0, 0]
// CHECK: linalg.matmul ins(%[[PAD]], %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%[[FILL]] : tensor<3x3xf32>)

// -----

func.func @linalg.conv_1d_nwc(%arg0: tensor<?x8x?xf32>, %arg1: tensor<2x?x?xf32>)
  -> tensor<?x7x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 2,
      input_spatial_dimensions = [1],
      kernel_input_feature_dimension = 1,
      kernel_output_feature_dimension = 2,
      kernel_spatial_dimensions = [0],
      output_batch_dimension = 0,
      output_feature_dimension = 2,
      output_spatial_dimensions = [1]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0]]> : tensor<1x2xi64>,
    rhs_dilation = dense<1> : tensor<1xi64>,
    window_strides = dense<1> : tensor<1xi64>,
    someattr
  } : (tensor<?x8x?xf32>, tensor<2x?x?xf32>) -> tensor<?x7x?xf32>
  func.return %0 : tensor<?x7x?xf32>
}
// CHECK-LABEL: func @linalg.conv_1d_nwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x?xf32>
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<2x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], 7, %[[DIM2]]]
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_1d_nwc_wcf
// CHECK-SAME:      {dilations = dense<1> : tensor<1xi64>
// CHECK-SAME:       someattr
// CHECK-SAME:       strides = dense<1> : tensor<1xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x8x?xf32>, tensor<2x?x?xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<?x7x?xf32>) -> tensor<?x7x?xf32>

// -----

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<?x4x5x?xf32>, %arg1: tensor<3x2x?x?xf32>)
  -> tensor<?x2x4x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>) -> tensor<?x2x4x?xf32>
  func.return %0 : tensor<?x2x4x?xf32>
}
// CHECK-LABEL: func @conv_2d_nhwc_hwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x4x5x?xf32>
// CHECK:         %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[C3]] : tensor<3x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], 2, 4, %[[DIM3]]]
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_2d_nhwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x2x4x?xf32>) -> tensor<?x2x4x?xf32>

// -----

func.func @conv_3d_ndhwc_dhwcf(%arg0: tensor<?x8x8x8x?xf32>, %arg1: tensor<2x2x2x?x?xf32>)
  -> tensor<?x7x7x7x?xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2, 3],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1, 2],
      output_batch_dimension = 0,
      output_feature_dimension = 4,
      output_spatial_dimensions = [1, 2, 3]
    >,
    feature_group_count = 1 : i64,
    padding = dense<[[0, 0], [0, 0], [0, 0]]> : tensor<3x2xi64>,
    rhs_dilation = dense<1> : tensor<3xi64>,
    window_strides = dense<1> : tensor<3xi64>
  } : (tensor<?x8x8x8x?xf32>, tensor<2x2x2x?x?xf32>) -> tensor<?x7x7x7x?xf32>
  func.return %0 : tensor<?x7x7x7x?xf32>
}
// CHECK-LABEL: func @conv_3d_ndhwc_dhwcf
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x8x8x?xf32>
// CHECK:         %[[C4:.+]] = arith.constant 4 : index
// CHECK:         %[[DIM4:.+]] = tensor.dim %[[ARG1]], %[[C4]] : tensor<2x2x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[DIM0]], 7, 7, 7, %[[DIM4]]]
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_3d_ndhwc_dhwcf
// CHECK-SAME:      {dilations = dense<1> : tensor<3xi64>
// CHECK-SAME:       strides = dense<1> : tensor<3xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x8x8x8x?xf32>, tensor<2x2x2x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x7x7x7x?xf32>) -> tensor<?x7x7x7x?xf32>

// -----

func.func @conv2d_1452x2223_dilated_valid(%arg0: tensor<1x4x5x2xf32>, %arg1: tensor<2x2x2x3xf32>)
  -> tensor<1x2x4x3xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 1 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<[2, 1]> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>
  } : (tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>) -> tensor<1x2x4x3xf32>
  func.return %0 : tensor<1x2x4x3xf32>
}
// CHECK-LABEL: func @conv2d_1452x2223_dilated_valid
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 2, 4, 3] : tensor<1x2x4x3xf32>
// CHECK:         %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      {dilations = dense<[2, 1]> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<1x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<1x2x4x3xf32>) -> tensor<1x2x4x3xf32>

// -----

// CHECK-LABEL: func @linalg.conv_2D_padding_test1
// CHECK-SAME: (%[[FILTER:.*]]: tensor<1x33x1x1xf16>, %[[INPUT:.*]]: tensor<400x1024x1024x1xf16>)
func.func @linalg.conv_2D_padding_test1(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>)
  -> tensor<400x1024x1024x1xf16> {
  %0 = mhlo.convolution(%arg1, %arg0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[0, 0], [16, 16]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<400x1024x1024x1xf16>, tensor<1x33x1x1xf16>) -> (tensor<400x1024x1024x1xf16>)
  func.return %0 : tensor<400x1024x1024x1xf16>
}
// CHECK-NEXT: %[[INIT:.*]] = linalg.init_tensor [400, 1024, 1024, 1] : tensor<400x1024x1024x1xf16>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[0, 0, 16, 0] high[0, 0, 16, 0]  {
// CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK-NEXT:   tensor.yield %[[ZERO]] : f16
// CHECK-NEXT: } : tensor<400x1024x1024x1xf16> to tensor<400x1024x1056x1xf16>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[PAD]], %[[FILTER]] : tensor<400x1024x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%[[FILL]] : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<400x1024x1024x1xf16>

// -----

// CHECK-LABEL: func @linalg.conv_2D_padding_test2
// CHECK-SAME: (%[[FILTER:.*]]: tensor<1x33x1x1xf16>, %[[INPUT:.*]]: tensor<400x1024x1024x1xf16>)
func.func @linalg.conv_2D_padding_test2(%arg0: tensor<1x33x1x1xf16>, %arg1: tensor<400x1024x1024x1xf16>)
  -> tensor<400x1040x1024x1xf16> {
  %0 = mhlo.convolution(%arg1, %arg0) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f], window = {stride = [1, 1], pad = [[8, 8], [16, 16]], rhs_dilate = [1, 1]} {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<400x1024x1024x1xf16>, tensor<1x33x1x1xf16>) -> (tensor<400x1040x1024x1xf16>)
  return %0 : tensor<400x1040x1024x1xf16>
}
// CHECK-NEXT: %[[INIT:.*]] = linalg.init_tensor [400, 1040, 1024, 1] : tensor<400x1040x1024x1xf16>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[0, 8, 16, 0] high[0, 8, 16, 0]  {
// CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK-NEXT:   tensor.yield %[[ZERO]] : f16
// CHECK-NEXT: } : tensor<400x1024x1024x1xf16> to tensor<400x1040x1056x1xf16>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%2, %arg0 : tensor<400x1040x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%1 : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
// CHECK-NEXT: return %[[RESULT]] : tensor<400x1040x1024x1xf16>

// -----

func.func @depthwise_conv(%arg0: tensor<2x4x5x2xf32>,
                     %arg1: tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 2 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    someattr} : (tensor<2x4x5x2xf32>, tensor<2x2x1x6xf32>) -> tensor<2x3x4x6xf32>
  func.return %0 : tensor<2x3x4x6xf32>
}
// CHECK:      func @depthwise_conv
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]

// CHECK:       %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2, 3]] : tensor<2x2x1x6xf32> into tensor<24xf32>
// CHECK:       %[[CAST:.+]] = tensor.cast %[[COLLAPSE]] : tensor<24xf32> to tensor<24xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[CAST]] {{\[}}[0, 1, 2, 3]] : tensor<24xf32> into tensor<2x2x2x3xf32>
// CHECK:       %[[INIT:.+]] = linalg.init_tensor [2, 3, 4, 2, 3] : tensor<2x3x4x2x3xf32>
// CHECK:       %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
// CHECK:       %[[OUT:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:     {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[IN]], %[[EXPAND]] : tensor<2x4x5x2xf32>, tensor<2x2x2x3xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x3x4x2x3xf32>) -> tensor<2x3x4x2x3xf32>
// CHECK:       %{{.+}} = tensor.collapse_shape %[[OUT]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
// CHECK-SAME:     : tensor<2x3x4x2x3xf32> into tensor<2x3x4x6xf32>

// -----

func.func @depthwise_conv_with_padding(
    %arg0: tensor<2x4x5x2xf32>,
    %arg1: tensor<2x2x1x4xf32>) -> tensor<2x3x6x4xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 2 : i64,
    padding = dense<[[0, 0], [1, 1]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<1> : tensor<2xi64>,
    someattr} : (tensor<2x4x5x2xf32>, tensor<2x2x1x4xf32>) -> tensor<2x3x6x4xf32>
  func.return %0 : tensor<2x3x6x4xf32>
}
// CHECK:      func @depthwise_conv_with_padding
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]
// CHECK:        %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 0, 1, 0] high[0, 0, 1, 0]  {
// CHECK:        ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:          tensor.yield %[[ZERO]] : f32
// CHECK         } : tensor<2x4x5x2xf32> to tensor<2x4x7x2xf32>
// CHECK:        %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:    [0, 1, 2, 3]
// CHECK-SAME:    : tensor<2x2x1x4xf32> into tensor<16xf32>
// CHECK:       %[[CAST:.+]] = tensor.cast %[[COLLAPSE]] : tensor<16xf32> to tensor<16xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[CAST]]
// CHECK-SAME:   [0, 1, 2, 3]
// CHECK-SAME:   tensor<16xf32> into tensor<2x2x2x2xf32>
// CHECK:        %[[INIT:.+]] = linalg.init_tensor [2, 3, 6, 2, 2] : tensor<2x3x6x2x2xf32>
// CHECK:        %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
// CHECK:        %[[OUT:.+]] = linalg.depthwise_conv_2d_nhwc_hwcm
// CHECK-SAME:     {dilations = dense<1> : tensor<2xi64>, someattr, strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[PAD]], %[[EXPAND]] : tensor<2x4x7x2xf32>, tensor<2x2x2x2xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
// CHECK:        %{{.+}} = tensor.collapse_shape %[[OUT]]
// CHECK-SAME:     [0], [1], [2], [3, 4]
// CHECK-SAME:     : tensor<2x3x6x2x2xf32> into tensor<2x3x6x4xf32>

// -----

func.func @depthwise_conv_multiplier_1(%arg0: tensor<1x113x113x96xf32>,
                                  %arg1: tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 96 : i64,
    padding = dense<0> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x56x56x96xf32>
  func.return %0 : tensor<1x56x56x96xf32>
}
// CHECK:       func @depthwise_conv_multiplier_1
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9_]*]]
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 56, 56, 96] : tensor<1x56x56x96xf32>
// CHECK:         %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>
// CHECK:         %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:         %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:       ins(%[[IN]], %[[RESHAPED_FILTER]] : tensor<1x113x113x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x56x56x96xf32>) -> tensor<1x56x56x96xf32>

// -----

func.func @depthwise_conv_multiplier_1_with_padding(
    %arg0: tensor<1x113x113x96xf32>,
    %arg1: tensor<3x3x1x96xf32>) -> tensor<1x57x58x96xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [1, 2]
    >,
    feature_group_count = 96 : i64,
    padding = dense<[[1, 1], [2, 2]]> : tensor<2x2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_strides = dense<2> : tensor<2xi64>} : (tensor<1x113x113x96xf32>, tensor<3x3x1x96xf32>) -> tensor<1x57x58x96xf32>
  func.return %0 : tensor<1x57x58x96xf32>
}
// CHECK:       func @depthwise_conv_multiplier_1_with_padding
// CHECK-SAME:    %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[FILTER:[a-zA-Z0-9_]*]]
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 1, 2, 0] high[0, 1, 2, 0]  {
// CHECK:         ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK          } : tensor<1x113x113x96xf32> to tensor<1x115x117x96xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 57, 58, 96] : tensor<1x57x58x96xf32>
// CHECK:         %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>
// CHECK:         %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:         %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:       ins(%[[PAD]], %[[RESHAPED_FILTER]] : tensor<1x115x117x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:       outs(%[[FILL]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>

// -----

func.func @reduce_window_min_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.minimum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
      someattr} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_min_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_min
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       someattr,
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

func.func @reduce_window_max_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_max_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

func.func @reduce_window_sum_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> tensor<1x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %0 : tensor<1x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_sum_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

func.func @reduce_window_max_nhwc_with_cst(%arg0: tensor<1x17x17x64xf32>) -> tensor<1x8x8x64xf32> {
  %0 = arith.constant dense<0xFF800000> : tensor<f32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<f32>, %arg2 : tensor<f32>):
    %2 = mhlo.maximum %arg1, %arg2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x64xf32>
  func.return %1 : tensor<1x8x8x64xf32>
}

// CHECK-LABEL: func @reduce_window_max_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[CST:.+]] = arith.constant dense<0xFF800000> : tensor<f32>
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[CST]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>

// -----

func.func @reduce_window_sum_max_nhwc(%arg0: tensor<1x17x17x64xf32>,
                             %arg1: tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>) {
  %0:2 = "mhlo.reduce_window"(%arg0, %arg0, %arg1, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>, %arg4: tensor<f32>, %arg5 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg4 : tensor<f32>
    %2 = mhlo.maximum %arg3, %arg5 : tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<1x17x17x64xf32>, tensor<1x17x17x64xf32>, tensor<f32>, tensor<f32>) -> (tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>)
  func.return %0#0, %0#1 : tensor<1x8x8x64xf32>, tensor<1x8x8x64xf32>
}

// CHECK-LABEL: func @reduce_window_sum_max_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW0:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT0:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL0:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL0:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES0:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW0]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL0]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[WINDOW1:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[INIT1:.+]] = linalg.init_tensor [1, 8, 8, 64] : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL1:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL1:.+]] = linalg.fill ins(%[[INIT_VAL1]] : f32) outs(%[[INIT1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES1:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW1]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         return %[[RES0]], %[[RES1]]

// -----

func.func @dynamic_reduce_window_sum_nhwc(%arg0: tensor<?x?x?x?xf32>,
                                      %arg1: tensor<f32>) -> tensor<?x?x?x?xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 1]> : tensor<4xi64>,
      window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<?x?x?x?xf32>, tensor<f32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}
// CHECK-LABEL: func @dynamic_reduce_window_sum_nhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3] : tensor<3x3xf32>
// CHECK:         %[[C0:.+]] = arith.constant 0 : index
// CHECK:         %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[D1:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[C2:.+]] = arith.constant 2 : index
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[C1:.+]] = arith.constant 1 : index
// CHECK:         %[[D2:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[D3:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]], %[[D3]]] : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<?x?x?x?xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>

// -----

func.func @reduce_window_min_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.minimum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 3, 1]> : tensor<5xi64>,
      window_strides = dense<[1, 2, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_min_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3, 3] : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 8, 64] : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_min
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

func.func @reduce_window_max_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.maximum %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 3, 1]> : tensor<5xi64>,
      window_strides = dense<[1, 2, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_max_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3, 3] : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 8, 64] : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

func.func @reduce_window_sum_ndhwc(%arg0: tensor<1x17x17x17x64xf32>,
                              %arg1: tensor<f32>) -> tensor<1x8x8x8x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {window_dimensions = dense<[1, 3, 3, 3, 1]> : tensor<5xi64>,
      window_strides = dense<[1, 2, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x8x64xf32>
  func.return %0 : tensor<1x8x8x8x64xf32>
}
// CHECK-LABEL: func @reduce_window_sum_ndhwc
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK:         %[[WINDOW:.+]] = linalg.init_tensor [3, 3, 3] : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = linalg.init_tensor [1, 8, 8, 8, 64] : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2, d1 + d2 * 2)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @reduce_window_generic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]

// CHECK: %[[INIT:.+]] = linalg.init_tensor [4, 7] : tensor<4x7xf32>
// CHECK: %[[FILL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%[[INIT]] : tensor<4x7xf32>)
// CHECK: ^bb0(%arg2: f32, %arg3: f32):
// CHECK:   linalg.yield %arg2 : f32

// CHECK: %[[PADVAL:.+]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[PAD:.+]] = tensor.pad %arg0 low[0, 1] high[3, 2]
// CHECK: ^bb0(%arg2: index, %arg3: index):
// CHECK:   tensor.yield %[[PADVAL]] : f32

// CHECK: %[[WINDOW:.+]] = linalg.init_tensor [2] : tensor<2xf32>
// CHECK: %[[REDUCE:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[PAD]], %[[WINDOW]] : tensor<7x9xf32>, tensor<2xf32>) outs(%[[FILL]] : tensor<4x7xf32>) {
// CHECK: ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
// CHECK:   %[[ADD:.+]] = arith.addf %arg2, %arg4 : f32
// CHECK:   linalg.yield %[[ADD]]

// CHECK: return %[[REDUCE]]

func.func @reduce_window_generic(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}

// -----

func.func @gather(%operand : tensor<1x4x8xi32>, %start_indices : tensor<1x8x2xi32>) -> tensor<1x8x8xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>,
    someattr
  } : (tensor<1x4x8xi32>, tensor<1x8x2xi32>) -> tensor<1x8x8xi32>
  func.return %res : tensor<1x8x8xi32>
}

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   func @gather(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [1, 8, 8] : tensor<1x8x8xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           indexing_maps = [#[[MAP0]]],
// CHECK-SAME:           iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:           outs(%[[INIT]] : tensor<1x8x8xi32>)
// CHECK-SAME:           {someattr}
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[IDX1]], %[[C0]]] : tensor<1x8x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[IDX1]], %[[C1]]] : tensor<1x8x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[IN2:.+]] = arith.addi %[[C0]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IN2]]] : tensor<1x4x8xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK-DAG:       return %[[RES]]

// -----

func.func @gather_unsigned(%operand : tensor<1x4x8xui32>, %start_indices : tensor<1x8x2xi32>) -> tensor<1x8x8xui32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<1x4x8xui32>, tensor<1x8x2xi32>) -> tensor<1x8x8xui32>
  func.return %res : tensor<1x8x8xui32>
}

// CHECK-LABEL:   func @gather_unsigned(
// CHECK:           linalg.generic
// CHECK-SAME:           outs(%{{.*}} : tensor<1x8x8xi32>)

// -----

func.func @gather_no_collapse(%operand : tensor<6x3xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x4x2xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[4, 2]> : tensor<2xi64>
  } : (tensor<6x3xi32>, tensor<5x2xi32>) -> tensor<5x4x2xi32>
  func.return %res : tensor<5x4x2xi32>
}

// CHECK-LABEL:   func @gather_no_collapse(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [5, 4, 2] : tensor<5x4x2xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<5x4x2xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C0]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C1]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[IDX1]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[S1]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]]] : tensor<6x3xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]


// -----

func.func @gather_max_offset(%operand : tensor<?x?x?xi32>, %start_indices : tensor<5x2xi32>) -> tensor<2x3x4x5xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[2, 3, 4]> : tensor<3xi64>
  } : (tensor<?x?x?xi32>, tensor<5x2xi32>) -> tensor<2x3x4x5xi32>
  func.return %res : tensor<2x3x4x5xi32>
}

// CHECK-LABEL:   func @gather_max_offset(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [2, 3, 4, 5] : tensor<2x3x4x5xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<2x3x4x5xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[IDX3:.+]] = linalg.index 3
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX3]], %[[C0]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX3]], %[[C1]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[S1]], %[[IDX1]] : index
// CHECK-DAG:         %[[IN2:.+]] = arith.addi %[[C0]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IN2]]] : tensor<?x?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @gather_reorder_start_index(%operand : tensor<6x3x2x7xi32>, %start_indices : tensor<5x4xi32>) -> tensor<5x2x4xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 2],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [3, 1, 2, 0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 2, 1, 4]> : tensor<4xi64>
  } : (tensor<6x3x2x7xi32>, tensor<5x4xi32>) -> tensor<5x2x4xi32>
  func.return %res : tensor<5x2x4xi32>
}

// CHECK-LABEL:   func @gather_reorder_start_index(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [5, 2, 4] : tensor<5x2x4xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<5x2x4xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C0]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[S1_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C1]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S1:.+]] = arith.index_cast %[[S1_INT]] : i32 to index
// CHECK-DAG:         %[[S2_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C2]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S2:.+]] = arith.index_cast %[[S2_INT]] : i32 to index
// CHECK-DAG:         %[[S3_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX0]], %[[C3]]] : tensor<5x4xi32>
// CHECK-DAG:         %[[S3:.+]] = arith.index_cast %[[S3_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S3]], %[[C0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[S1]], %[[IDX1]] : index
// CHECK-DAG:         %[[IN2:.+]] = arith.addi %[[S2]], %[[C0]] : index
// CHECK-DAG:         %[[IN3:.+]] = arith.addi %[[S0]], %[[IDX2]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IN2]], %[[IN3]]] : tensor<6x3x2x7xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @gather_implicit_trailing_dim(%operand : tensor<?x?xi32>, %start_indices : tensor<5x2xi32>) -> tensor<3x4x5x2xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 2,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<?x?xi32>, tensor<5x2xi32>) -> tensor<3x4x5x2xi32>
  func.return %res : tensor<3x4x5x2xi32>
}

// CHECK-LABEL:   func @gather_implicit_trailing_dim(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [3, 4, 5, 2] : tensor<3x4x5x2xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x5x2xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[IDX3:.+]] = linalg.index 3
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[IDX3]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[C0]], %[[IDX1]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @gather_non_static(%operand : tensor<?x?xi32>, %start_indices : tensor<?x?xi32>) -> tensor<3x4x?xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<3x4x?xi32>
  func.return %res : tensor<3x4x?xi32>
}

// CHECK-LABEL:   func @gather_non_static(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[C4:.+]] = arith.constant 4

//                  Reify return shape
// CHECK-DAG:       %[[CI_3:.+]] = arith.index_cast %[[C3]] : index to i32
// CHECK-DAG:       %[[CI_4:.+]] = arith.index_cast %[[C4]] : index to i32
// CHECK-DAG:       %[[C0_REPEAT:.+]] = arith.constant 0
// CHECK-DAG:       %[[START_INDICES_DIM0_INT:.+]] = tensor.dim %[[START_INDICES]], %[[C0_REPEAT]] : tensor<?x?xi32>
// CHECK-DAG:       %[[START_INDICES_DIM0:.+]] = arith.index_cast %[[START_INDICES_DIM0_INT]] : index to i32
// CHECK-DAG:       %[[RES_SHAPE:.+]] = tensor.from_elements %[[CI_3]], %[[CI_4]], %[[START_INDICES_DIM0]] : tensor<3xi32>
// CHECK-DAG:       %[[DYN_DIM_INT:.+]] = tensor.extract %[[RES_SHAPE]][%[[C2]]] : tensor<3xi32>
// CHECK-DAG:       %[[DYN_DIM:.+]] = arith.index_cast %[[DYN_DIM_INT]] : i32 to index

// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [3, 4, %[[DYN_DIM]]] : tensor<3x4x?xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x?xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[C0]], %[[IDX1]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @gather_unranked(%operand : tensor<*xi32>, %start_indices : tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [0, 1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[3, 4]> : tensor<2xi64>
  } : (tensor<*xi32>, tensor<?x?xi32>) -> tensor<?x?x?xi32>
  func.return %res : tensor<?x?x?xi32>
}

// CHECK-LABEL:   func @gather_unranked(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[C4:.+]] = arith.constant 4

//                  Reify return shape
// CHECK-DAG:       %[[CI3:.+]] = arith.index_cast %[[C3]] : index to i32
// CHECK-DAG:       %[[CI4:.+]] = arith.index_cast %[[C4]] : index to i32
// CHECK-DAG:       %[[C0_REPEAT:.+]] = arith.constant 0
// CHECK-DAG:       %[[START_INDICES_DIM0_INT:.+]] = tensor.dim %[[START_INDICES]], %[[C0_REPEAT]] : tensor<?x?xi32>
// CHECK-DAG:       %[[START_INDICES_DIM0:.+]] = arith.index_cast %[[START_INDICES_DIM0_INT]] : index to i32
// CHECK-DAG:       %[[RES_SHAPE:.+]] = tensor.from_elements %[[CI3]], %[[CI4]], %[[START_INDICES_DIM0]] : tensor<3xi32>
// CHECK-DAG:       %[[RES_DIM0_INT:.+]] = tensor.extract %[[RES_SHAPE]][%[[C0]]] : tensor<3xi32>
// CHECK-DAG:       %[[RES_DIM0:.+]] = arith.index_cast %[[RES_DIM0_INT]] : i32 to index
// CHECK-DAG:       %[[RES_DIM1_INT:.+]] = tensor.extract %[[RES_SHAPE]][%[[C1]]] : tensor<3xi32>
// CHECK-DAG:       %[[RES_DIM1:.+]] = arith.index_cast %[[RES_DIM1_INT]] : i32 to index
// CHECK-DAG:       %[[RES_DIM2_INT:.+]] = tensor.extract %[[RES_SHAPE]][%[[C2]]] : tensor<3xi32>
// CHECK-DAG:       %[[RES_DIM2:.+]] = arith.index_cast %[[RES_DIM2_INT]] : i32 to index

// CHECK-DAG:       %[[INIT:.+]] = linalg.init_tensor [%[[RES_DIM0]], %[[RES_DIM1]], %[[RES_DIM2]]] : tensor<?x?x?xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<?x?x?xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[S0]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[C0]], %[[IDX1]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]]] : tensor<*xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           return %[[RES]]

// -----

func.func @torch_index_select(%arg0: tensor<5x1x5xi32>,
                         %arg1: tensor<2xi32>) ->  tensor<2x1x5xi32> {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64,
    someattr
  } : (tensor<5x1x5xi32>, tensor<2xi32>) -> tensor<2x1x5xi32>
  func.return %0 : tensor<2x1x5xi32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @torch_index_select
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK: %[[DUMMY:.+]] = linalg.init_tensor [1, 5] : tensor<1x5xi32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[DUMMY]] : tensor<2xi32>, tensor<1x5xi32>)
// CHECK-SAME: {someattr}
//      CHECK: ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32):
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:   %[[J:.+]] = linalg.index 1
//      CHECK:   %[[K:.+]] = linalg.index 2
//      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
//      CHECK:   linalg.yield %[[VAL2]] : i32

// -----

func.func @rng_uniform_1d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<10xf32>
{
  %shape = arith.constant dense<[10]>  : tensor<1xi32>
  %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func @rng_uniform_1d
// CHECK-DAG:  ^{{.+}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[ADD_SEED:.+]] = arith.addi %[[IDX0_CAST]], %[[C0]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[ADD_SEED]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = arith.uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = arith.mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<10xf32>

// -----

func.func @rng_uniform_2d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<3x3xf32>
{
        %shape = arith.constant dense<[3, 3]>  : tensor<2xi32>
        %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
        func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @rng_uniform_2d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[ADD_SEED:.+]] = arith.addi %[[IDX0_CAST]], %[[C0]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[ADD_SEED]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL5_CAST:.+]] = arith.uitofp %[[VAL5]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL5_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<3x3xf32>

// -----

func.func @rng_uniform_3d(%min: tensor<f32>, %max: tensor<f32>) -> tensor<2x2x2xf32>
{
        %shape = arith.constant dense<[2, 2, 2]>  : tensor<3xi32>
        %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x2x2xf32>
        func.return %0 : tensor<2x2x2xf32>
}
// CHECK-LABEL: func @rng_uniform_3d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:  %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:  %[[IDX2_CAST:.+]] = arith.index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[ADD_SEED:.+]] = arith.addi %[[IDX0_CAST]], %[[C0]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[ADD_SEED]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL3:.+]] = arith.addi %[[IDX1_CAST]], %[[VAL2]] : i32
// CHECK-DAG:    %[[VAL4:.+]] = arith.muli %[[VAL3]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addi %[[VAL4]], %[[CST1]] : i32
// CHECK-DAG:    %[[VAL6:.+]] = arith.addi %[[IDX2_CAST]], %[[VAL5]] : i32
// CHECK-DAG:    %[[VAL7:.+]] = arith.muli %[[VAL6]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL8:.+]] = arith.addi %[[VAL7]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL8_CAST:.+]] = arith.uitofp %[[VAL8]] : i32 to f32
// CHECK-DAG:    %[[VAL6:.+]] = arith.mulf %[[VAL8_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL7:.+]] = arith.addf %[[VAL6]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL7]] : f32
// CHECK-NEXT: -> tensor<2x2x2xf32>

// -----

func.func @rng_uniform_dynamic_1d(%min: tensor<f32>, %max: tensor<f32>, %shape: tensor<1xi32>) -> tensor<?xf32>
{
  %0 = "mhlo.rng_uniform"(%min, %max, %shape) : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @rng_uniform_dynamic_1d
// CHECK-DAG:    %[[CAST:.+]] = arith.index_cast %{{.+}} : tensor<1xi32> to tensor<1xindex>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C0INT:.+]] = arith.constant 0 : i32
// CHECK-DAG:    %[[IND:.+]] = tensor.extract %[[CAST]][%[[C0]]] : tensor<1xindex>
// CHECK-DAG:    %{{.+}} = linalg.init_tensor [%[[IND]]] : tensor<?xf32>
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[ADD_SEED:.+]] = arith.addi %[[IDX0_CAST]], %[[C0INT]] : i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[ADD_SEED]], %[[CST0]] : i32
// CHECK-DAG:    %[[VAL2:.+]] = arith.addi %[[VAL1]], %[[CST1]] : i32
// CHECK-DAG:    %[[DIFF:.+]] = arith.subf %[[MAX]], %[[MIN]] : f32
// CHECK-DAG:    %[[FACT:.+]] = arith.mulf %[[DIFF]], %[[CST2]] : f32
// CHECK-DAG:    %[[VAL2_CAST:.+]] = arith.uitofp %[[VAL2]] : i32 to f32
// CHECK-DAG:    %[[VAL4:.+]] = arith.mulf %[[VAL2_CAST]], %[[FACT]] : f32
// CHECK-DAG:    %[[VAL5:.+]] = arith.addf %[[VAL4]], %[[MIN]] : f32
// CHECK-NEXT:   linalg.yield %[[VAL5]] : f32
// CHECK-NEXT: -> tensor<?xf32>

// -----

func.func @torch_index_select_unsigned(%arg0: tensor<5x1x5xui32>,
                                  %arg1: tensor<2xi32>) ->  tensor<2x1x5xui32> {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xui32>, tensor<2xi32>) -> tensor<2x1x5xui32>
  func.return %0 : tensor<2x1x5xui32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//      CHECK: func @torch_index_select_unsigned
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK:   %[[INPUT_SIGNLESS:.*]] = builtin.unrealized_conversion_cast %[[INPUT]] : tensor<5x1x5xui32> to tensor<5x1x5xi32>
//      CHECK:   %[[DUMMY:.+]] = linalg.init_tensor [1, 5] : tensor<1x5xi32>
//      CHECK:   %[[RES:.+]] = linalg.generic {
// CHECK-SAME:     indexing_maps
// CHECK-SAME:     #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[DUMMY]] : tensor<2xi32>, tensor<1x5xi32>)
//      CHECK:   ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: i32, %{{.+}}: i32):
//      CHECK:     %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:     %[[J:.+]] = linalg.index 1
//      CHECK:     %[[K:.+]] = linalg.index 2
//      CHECK:     %[[VAL2:.+]] = tensor.extract %[[INPUT_SIGNLESS]][%[[CAST]], %[[J]], %[[K]]] : tensor<5x1x5xi32>
//      CHECK:     linalg.yield %[[VAL2]] : i32
//      CHECK:   %[[RES_UNSIGNED:.+]] = builtin.unrealized_conversion_cast %[[RES]] : tensor<2x1x5xi32> to tensor<2x1x5xui32>
//      CHECK:   return %[[RES_UNSIGNED]]

// -----

func.func @torch_index_select_scalar(%arg0: tensor<4x8xf32>,
                                %arg1: tensor<i32>) -> tensor<8xf32> {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
    batch_dims = 0 : i64,
    dim = 0 : i64
  } : (tensor<4x8xf32>, tensor<i32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0) -> ()>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0)>
//      CHECK: func @torch_index_select_scalar
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK: %[[DUMMY:.+]] = linalg.init_tensor [8] : tensor<8xf32>
//      CHECK: %[[T0:.+]] = linalg.init_tensor [8] : tensor<8xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP1]]
// CHECK-SAME:   iterator_types = ["parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[DUMMY]] : tensor<i32>, tensor<8xf32>) outs(%[[T0]] : tensor<8xf32>)
//      CHECK:   ^{{.+}}(%[[VAL:[a-zA-Z0-9_]+]]: i32, %{{.+}}: f32, %{{.+}}: f32):
//      CHECK:     %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:     %[[I:.+]] = linalg.index 0
//      CHECK:     %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[CAST]], %[[I]]] : tensor<4x8xf32>
//      CHECK:     linalg.yield %[[VAL2]] : f32

// -----

func.func @torch_index_select_batch(%arg0: tensor<4x7x8x2xf32>,
                               %arg1: tensor<4x1xi32>) -> tensor<4x7x1x2xf32> {
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
    dim = 2 : i64,
    batch_dims = 1 : i64
  } : (tensor<4x7x8x2xf32>, tensor<4x1xi32>) -> tensor<4x7x1x2xf32>
  func.return %0 : tensor<4x7x1x2xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @torch_index_select_batch
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK: %[[DUMMY:.+]] = linalg.init_tensor [4, 7, 2] : tensor<4x7x2xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[DUMMY]] : tensor<4x1xi32>, tensor<4x7x2xf32>)
// CHECK-NEXT: ^{{.+}}(%[[VAL:.+]]: i32, %{{.+}}: f32, %{{.+}}: f32):
//      CHECK:   %[[CAST:.+]] = arith.index_cast %[[VAL]] : i32 to index
//      CHECK:   %[[I:.+]] = linalg.index 0
//      CHECK:   %[[J:.+]] = linalg.index 1
//      CHECK:   %[[L:.+]] = linalg.index 3
//      CHECK:   %[[VAL2:.+]] = tensor.extract %[[INPUT]][%[[I]], %[[J]], %[[CAST]], %[[L]]] : tensor<4x7x8x2xf32>
//      CHECK:   linalg.yield %[[VAL2]] : f32

// -----

func.func @torch_index_select_dynamic(%input: tensor<?x?x?x?xf32>,
                                 %index: tensor<?x?xi32>) -> tensor<?x?x?x?xf32>{
  %0 = "mhlo.torch_index_select"(%input, %index) {
    batch_dims = 1 : i64,
    dim = 2 : i64
  } : (tensor<?x?x?x?xf32>, tensor<?x?xi32>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}
//      CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
//      CHECK: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
//      CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//      CHECK: func @torch_index_select_dynamic
// CHECK-SAME:   %[[INPUT:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[INDEX:[a-zA-Z0-9_]*]]
//      CHECK:   %[[C0:.+]] = arith.constant 0 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[D1:.+]] = tensor.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[C1:.+]] = arith.constant 1 : index
//      CHECK:   %[[D2:.+]] = tensor.dim %[[INDEX]], %[[C1]]
//      CHECK:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   %[[D3:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[C4:.+]] = arith.constant 0 : index
//      CHECK:   %[[D4:.+]] = tensor.dim %[[INPUT]], %[[C4]]
//      CHECK:   %[[C5:.+]] = arith.constant 1 : index
//      CHECK:   %[[D5:.+]] = tensor.dim %[[INPUT]], %[[C5]]
//      CHECK:   %[[C6:.+]] = arith.constant 3 : index
//      CHECK:   %[[D6:.+]] = tensor.dim %[[INPUT]], %[[C6]]
//      CHECK:   %[[DUMMY:.+]] = linalg.init_tensor [%[[D4]], %[[D5]], %[[D6]]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]], %[[D3]]]
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[INDEX]], %[[DUMMY]] : tensor<?x?xi32>, tensor<?x?x?xf32>)
// CHECK-SAME:     outs(%[[INIT]] : tensor<?x?x?x?xf32>)
//      CHECK:     ^{{.+}}(
// CHECK-SAME:       %[[ARG0:[a-zA-Z0-9_]+]]: i32, %{{[a-zA-Z0-9_]+}}: f32, %{{[a-zA-Z0-9_]+}}: f32)
//      CHECK:       %[[POS:.+]] = arith.index_cast %[[ARG0]]
//      CHECK:       %[[IDX0:.+]] = linalg.index 0
//      CHECK:       %[[IDX1:.+]] = linalg.index 1
//      CHECK:       %[[IDX3:.+]] = linalg.index 3
//      CHECK:       %[[YIELD:.+]] = tensor.extract %[[INPUT]][%[[IDX0]], %[[IDX1]], %[[POS]], %[[IDX3]]]
//      CHECK:       linalg.yield %[[YIELD]]

// -----

// CHECK-LABEL:   func @concatenate(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[VAL_8]] : tensor<?x?xi32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_11:.*]] = tensor.dim %[[VAL_1]], %[[VAL_10]] : tensor<?x?xi32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_13:.*]] = tensor.dim %[[VAL_2]], %[[VAL_12]] : tensor<?x?xi32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_2]], %[[VAL_14]] : tensor<?x?xi32>
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_7]], %[[VAL_11]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK:           %[[VAL_18:.*]] = tensor.from_elements %[[VAL_5]], %[[VAL_17]] : tensor<2xindex>
// CHECK:           %[[VAL_19:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_20:.*]] = tensor.extract %[[VAL_18]][%[[VAL_19]]] : tensor<2xindex>
// CHECK:           %[[VAL_21:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_22:.*]] = tensor.extract %[[VAL_18]][%[[VAL_21]]] : tensor<2xindex>
// CHECK:           %[[VAL_23:.*]] = linalg.init_tensor [%[[VAL_20]], %[[VAL_22]]] : tensor<?x?xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_23]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_27:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_29:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_30:.*]] = tensor.dim %[[VAL_0]], %[[VAL_29]] : tensor<?x?xi32>
// CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_3]], %[[VAL_30]] : index
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_31]] : index
// CHECK:             %[[VAL_33:.*]] = scf.if %[[VAL_32]] -> (i32) {
// CHECK:               %[[VAL_34:.*]] = arith.subi %[[VAL_28]], %[[VAL_3]] : index
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_0]][%[[VAL_26]], %[[VAL_34]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_35]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_36:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_37:.*]] = tensor.dim %[[VAL_1]], %[[VAL_36]] : tensor<?x?xi32>
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_31]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_38]] : index
// CHECK:               %[[VAL_40:.*]] = scf.if %[[VAL_39]] -> (i32) {
// CHECK:                 %[[VAL_41:.*]] = arith.subi %[[VAL_28]], %[[VAL_31]] : index
// CHECK:                 %[[VAL_42:.*]] = tensor.extract %[[VAL_1]][%[[VAL_26]], %[[VAL_41]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_42]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_43:.*]] = arith.subi %[[VAL_28]], %[[VAL_38]] : index
// CHECK:                 %[[VAL_44:.*]] = tensor.extract %[[VAL_2]][%[[VAL_26]], %[[VAL_43]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_44]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_45:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_46:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           return %[[VAL_47:.*]] : tensor<?x?xi32>
// CHECK:         }
func.func @concatenate(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
    %concat = "mhlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
    func.return %concat : tensor<?x?xi32>
}

// -----

// CHECK-LABEL:   func @concatenate_unsigned(
// CHECK-SAME:   %[[VAL_0:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_1:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[VAL_2:[a-zA-Z0-9_]*]]
// CHECK-DAG:       %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_3]], %[[VAL_7]] : tensor<?x?xi32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_3]], %[[VAL_9]] : tensor<?x?xi32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_12:.*]] = tensor.dim %[[VAL_4]], %[[VAL_11]] : tensor<?x?xi32>
// CHECK:           %[[VAL_13:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[VAL_13]] : tensor<?x?xi32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = tensor.dim %[[VAL_5]], %[[VAL_15]] : tensor<?x?xi32>
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_5]], %[[VAL_17]] : tensor<?x?xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:           %[[VAL_21:.*]] = tensor.from_elements %[[VAL_8]], %[[VAL_20]] : tensor<2xindex>
// CHECK:           %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_23:.*]] = tensor.extract %[[VAL_21]][%[[VAL_22]]] : tensor<2xindex>
// CHECK:           %[[VAL_24:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_25:.*]] = tensor.extract %[[VAL_21]][%[[VAL_24]]] : tensor<2xindex>
// CHECK:           %[[VAL_26:.*]] = linalg.init_tensor [%[[VAL_23]], %[[VAL_25]]] : tensor<?x?xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_26]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_31:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_32:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_33:.*]] = tensor.dim %[[VAL_3]], %[[VAL_32]] : tensor<?x?xi32>
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_6]], %[[VAL_33]] : index
// CHECK:             %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_31]], %[[VAL_34]] : index
// CHECK:             %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i32) {
// CHECK:               %[[VAL_37:.*]] = arith.subi %[[VAL_31]], %[[VAL_6]] : index
// CHECK:               %[[VAL_38:.*]] = tensor.extract %[[VAL_3]][%[[VAL_29]], %[[VAL_37]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_38]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_39:.*]] = arith.constant 1 : index
// CHECK:               %[[VAL_40:.*]] = tensor.dim %[[VAL_4]], %[[VAL_39]] : tensor<?x?xi32>
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_34]], %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_31]], %[[VAL_41]] : index
// CHECK:               %[[VAL_43:.*]] = scf.if %[[VAL_42]] -> (i32) {
// CHECK:                 %[[VAL_44:.*]] = arith.subi %[[VAL_31]], %[[VAL_34]] : index
// CHECK:                 %[[VAL_45:.*]] = tensor.extract %[[VAL_4]][%[[VAL_29]], %[[VAL_44]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_45]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_46:.*]] = arith.subi %[[VAL_31]], %[[VAL_41]] : index
// CHECK:                 %[[VAL_47:.*]] = tensor.extract %[[VAL_5]][%[[VAL_29]], %[[VAL_46]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_47]] : i32
// CHECK:               }
// CHECK:               scf.yield %[[VAL_48:.*]] : i32
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_49:.*]] : i32
// CHECK:           } -> tensor<?x?xi32>
// CHECK:           %[[VAL_50:.*]] = builtin.unrealized_conversion_cast %[[VAL_51:.*]] : tensor<?x?xi32> to tensor<?x?xui32>
// CHECK:           return %[[VAL_50]] : tensor<?x?xui32>
// CHECK:         }
func.func @concatenate_unsigned(%a: tensor<?x?xui32>, %b: tensor<?x?xui32>, %c: tensor<?x?xui32>) -> tensor<?x?xui32> {
    %concat = "mhlo.concatenate"(%a, %b, %c) {
      dimension = 1
    } : (tensor<?x?xui32>, tensor<?x?xui32>, tensor<?x?xui32>) -> tensor<?x?xui32>
    func.return %concat : tensor<?x?xui32>
}

// -----

// CHECK-LABEL: signed_divide
func.func @signed_divide(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
  // CHECK-DAG:   %[[VAL_7:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_8:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_10:.*]] = arith.constant 1 : i32
  // CHECK:   %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_9]] : i32
  // CHECK-DAG:   %[[VAL_12:.*]] = arith.constant -2147483648 : i32
  // CHECK:   %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_12]] : i32
  // CHECK-DAG:   %[[VAL_14:.*]] = arith.constant -1 : i32
  // CHECK:   %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_14]] : i32
  // CHECK:   %[[VAL_16:.*]] = arith.andi %[[VAL_13]], %[[VAL_15]] : i1
  // CHECK:   %[[VAL_17:.*]] = arith.ori %[[VAL_11]], %[[VAL_16]] : i1
  // CHECK:   %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_10]], %[[VAL_5]] : i32
  // CHECK:   %[[VAL_19:.*]] = arith.divsi %[[VAL_4]], %[[VAL_18]] : i32
  // CHECK:   %[[VAL_20:.*]] = arith.select %[[VAL_16]], %[[VAL_8]], %[[VAL_19]] : i32
  // CHECK:   %[[VAL_21:.*]] = arith.select %[[VAL_11]], %[[VAL_7]], %[[VAL_20]] : i32
  // CHECK:   linalg.yield %[[VAL_21]] : i32
  %0 = "mhlo.divide"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: unsigned_divide
func.func @unsigned_divide(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_10:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_12:.*]] = arith.constant 1 : i32
  // CHECK:   %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_11]] : i32
  // CHECK:   %[[VAL_14:.*]] = arith.select %[[VAL_13]], %[[VAL_12]], %[[VAL_7]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.divui %[[VAL_6]], %[[VAL_14]] : i32
  // CHECK:   %[[VAL_16:.*]] = arith.select %[[VAL_13]], %[[VAL_9]], %[[VAL_15]] : i32
  // CHECK:   linalg.yield %[[VAL_16]] : i32
  %0 = "mhlo.divide"(%lhs, %rhs) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}

// -----

// CHECK-LABEL: complex_divide
func.func @complex_divide(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.div
  %0 = "mhlo.divide"(%lhs, %rhs) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
  func.return %0 : tensor<2xcomplex<f32>>
}

// -----

// CHECK-LABEL: signed_remainder
func.func @signed_remainder(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
  // CHECK-DAG:   %[[VAL_7:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_8:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 1 : i32
  // CHECK:   %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_8]] : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant -2147483648 : i32
  // CHECK:   %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_11]] : i32
  // CHECK-DAG:   %[[VAL_13:.*]] = arith.constant -1 : i32
  // CHECK:   %[[VAL_14:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_13]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.andi %[[VAL_12]], %[[VAL_14]] : i1
  // CHECK:   %[[VAL_16:.*]] = arith.ori %[[VAL_10]], %[[VAL_15]] : i1
  // CHECK:   %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_9]], %[[VAL_5]] : i32
  // CHECK:   %[[VAL_18:.*]] = arith.remsi %[[VAL_4]], %[[VAL_17]] : i32
  // CHECK:   %[[VAL_19:.*]] = arith.select %[[VAL_15]], %[[VAL_7]], %[[VAL_18]] : i32
  // CHECK:   %[[VAL_20:.*]] = arith.select %[[VAL_10]], %[[VAL_4]], %[[VAL_19]] : i32
  // CHECK:   linalg.yield %[[VAL_20]] : i32
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: unsigned_remainder
func.func @unsigned_remainder(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_10:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant 1 : i32
  // CHECK:   %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_7]], %[[VAL_10]] : i32
  // CHECK:   %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[VAL_11]], %[[VAL_7]] : i32
  // CHECK:   %[[VAL_14:.*]] = arith.remui %[[VAL_6]], %[[VAL_13]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.select %[[VAL_12]], %[[VAL_6]], %[[VAL_14]] : i32
  // CHECK:   linalg.yield %[[VAL_15]] : i32
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}

// -----

// CHECK-LABEL: unsigned_convert
func.func @unsigned_convert(%in: tensor<2x2xui32>) -> tensor<2x2xui64> {
  // CHECK: linalg.generic
  // CHECK: arith.extui
  %0 = "mhlo.convert"(%in) : (tensor<2x2xui32>) -> tensor<2x2xui64>
  func.return %0 : tensor<2x2xui64>
}

// -----

// CHECK-LABEL: unsigned_compare
func.func @unsigned_compare(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xi1> {
  // CHECK: linalg.generic
  // CHECK: cmpi ugt
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<"comparison_direction GT">} : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}

// -----

func.func @scatter_update_scalar(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}
// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1) -> (d1, 0)>
// CHECK-DAG:   #[[MAP2:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       func @scatter_update_scalar
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] : tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>)
// CHECK-SAME:     outs(%[[ARG0]] : tensor<3xi32>) {
// CHECK:         ^bb0(%{{.*}}: i32, %[[IDX_I32:.*]]: i32, %[[UPDATE:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:           %[[CMP_IDX:.*]] = linalg.index 0 : index
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
// CHECK:           %[[PRED:.*]] = arith.cmpi eq, %[[CMP_IDX]], %[[IDX]] : index
// CHECK:           %[[SELECT:.*]] = arith.select %[[PRED]], %[[UPDATE]], %[[OUT]] : i32
// CHECK:           linalg.yield %[[SELECT]] : i32
// CHECK:         } -> tensor<3xi32>
// CHECK:         return %[[RES]] : tensor<3xi32>

// -----

func.func @scatter_update_slice(%arg0: tensor<6x3xi32>, %arg1: tensor<2x1xi32>,
                           %arg2: tensor<2x3xi32>) -> tensor<6x3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  func.return %0 : tensor<6x3xi32>
}
// CHECK-DAG:   #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:   #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2, 0)>
// CHECK-DAG:   #[[MAP2:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK:       func @scatter_update_slice
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK:         %[[RES:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP0]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:      ins(%[[ARG0]], %[[ARG1]], %[[ARG2]] : tensor<6x3xi32>, tensor<2x1xi32>, tensor<2x3xi32>)
// CHECK-SAME:     outs(%[[ARG0]] : tensor<6x3xi32>) {
// CHECK:         ^bb0(%{{.*}}: i32, %[[IDX_I32:.*]]: i32, %[[UPDATE:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK:           %[[CMP_IDX:.*]] = linalg.index 0 : index
// CHECK:           %[[IDX:.*]] = arith.index_cast %[[IDX_I32]] : i32 to index
// CHECK:           %[[PRED:.*]] = arith.cmpi eq, %[[CMP_IDX]], %[[IDX]] : index
// CHECK:           %[[SELECT:.*]] = arith.select %[[PRED]], %[[UPDATE]], %[[OUT]] : i32
// CHECK:           linalg.yield %[[SELECT]] : i32
// CHECK:         } -> tensor<6x3xi32>
// CHECK:         return %[[RES]] : tensor<6x3xi32>

// -----

func.func @const() -> tensor<3xi32> {
  // CHECK: = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst = mhlo.constant dense<[1, 2, 3]> : tensor<3xi32>
  func.return %cst : tensor<3xi32>
}
// -----

func.func @const_unsigned() -> tensor<3xui32> {
  // CHECK: = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
  %cst = mhlo.constant dense<[1, 2, 3]> : tensor<3xui32>
  func.return %cst : tensor<3xui32>
}

// -----

func.func @const_splat() -> tensor<3xi16> {
  // CHECK: = arith.constant dense<1> : tensor<3xi16>
  %cst = mhlo.constant dense<1> : tensor<3xi16>
  func.return %cst : tensor<3xi16>
}

// -----

// CHECK-LABEL: @real_real
func.func @real_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.real"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.+]]: f32,
  // CHECK: yield %[[ARG]]
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @imag_real
func.func @imag_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.imag"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: linalg.generic
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: yield %[[CST]]
  func.return %1 : tensor<?xf32>
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>,
                  %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// The iterations are (Batch Dim, LHS Other Dim, RHS Other dim, Contracting Dim)
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d2, d3, d0)>
// Output is the iterators excluding contracting
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK: func @dot_general(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x?xf32>, %[[ARG1:.*]]: tensor<?x?x?xf32>)
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements %[[D0]], %[[D1]], %[[D2]]
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK: %[[INIT:.*]] = linalg.init_tensor [%[[D0]], %[[D1]], %[[D2]]]
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// Only contracting dims are reductions
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<?x?x?xf32>, tensor<?x?x?xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<?x?x?xf32>)
// CHECK-SAME: {someattr}
// CHECK:   ^bb0(%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: f32):
// CHECK:     %[[MUL:.*]] = arith.mulf %[[ARG2]], %[[ARG3]] : f32
// CHECK:     %[[SUM:.*]] = arith.addf %[[ARG4]], %[[MUL]] : f32
// CHECK:     linalg.yield %[[SUM]] : f32
// CHECK: } -> tensor<?x?x?xf32>

// -----

func.func @dot_general_unsigned(%arg0: tensor<?x?x?xui32>,
                  %arg1: tensor<?x?x?xui32>) -> tensor<?x?x?xui32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<?x?x?xui32>, tensor<?x?x?xui32>) -> tensor<?x?x?xui32>
  func.return %0 : tensor<?x?x?xui32>
}

// CHECK-LABEL: func @dot_general_unsigned(
// CHECK: linalg.generic
// CHECK-SAME: ins({{.*}} : tensor<?x?x?xi32>, tensor<?x?x?xi32>)
// CHECK-SAME: outs({{.*}} : tensor<?x?x?xi32>)

// -----

func.func @dot_general_complex(%arg0: tensor<?x?x?xcomplex<f32>>,
                  %arg1: tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [2],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<?x?x?xcomplex<f32>>, tensor<?x?x?xcomplex<f32>>) -> tensor<?x?x?xcomplex<f32>>
  func.return %0 : tensor<?x?x?xcomplex<f32>>
}

// CHECK-LABEL: func @dot_general_complex(
// CHECK: linalg.generic
// CHECK: complex.mul
// CHECK: complex.add

// -----

func.func @dot_general_multiple_batch_dimensions(%arg0: tensor<3x4x2x4xi32>,
             %arg1: tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3]>,
    precision_config = [#mhlo<"precision DEFAULT">, #mhlo<"precision DEFAULT">],
    someattr
  } : (tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>) -> tensor<3x4x4x3xi32>
  return %0 : tensor<3x4x4x3xi32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
// CHECK: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @dot_general_multiple_batch_dimensions(
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<3x4x2x4xi32>, tensor<3x4x3x2xi32>)
// CHECK-SAME: outs({{.*}} : tensor<3x4x4x3xi32>)
// CHECK-SAME: {someattr}
