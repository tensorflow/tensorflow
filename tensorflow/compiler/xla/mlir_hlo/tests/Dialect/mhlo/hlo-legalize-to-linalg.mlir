// RUN: mlir-hlo-opt %s --hlo-legalize-to-linalg --split-input-file \
// RUN:   --canonicalize | \
// RUN: FILECHECK_OPTS="" FileCheck %s

// RUN: mlir-hlo-opt %s --hlo-legalize-to-linalg="enable-primitive-ops=true" \
// RUN:   --split-input-file --canonicalize | \
// RUN: FILECHECK_OPTS="" FileCheck %s --check-prefix=CHECK-PRIMITIVE

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
// CHECK-PRIMITIVE-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = arith.addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: linalg.yield
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_add_dynamic_encoding
// CHECK-PRIMITIVE-LABEL: func @float_add_dynamic_encoding
func.func @float_add_dynamic_encoding(
  %lhs: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>,
  %rhs: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>)
    -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.yield

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: linalg.yield
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>,
         tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>)
      -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
}

// -----

// CHECK-LABEL: integer_add
// CHECK-PRIMITIVE-LABEL: integer_add
func.func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: addi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: addi
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_add
// CHECK-PRIMITIVE-LABEL: complex_add
func.func @complex_add(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.add
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.add
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_atan2
// CHECK-PRIMITIVE-LABEL: func @complex_atan2
func.func @complex_atan2(%lhs: tensor<2x2xcomplex<f32>>,
    %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.atan2"(%lhs, %rhs)
      : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
      -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.atan2
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.atan2
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}


// -----

// CHECK-LABEL: func @float_mul
// CHECK-PRIMITIVE-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: mulf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: mulf
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
// CHECK-PRIMITIVE-LABEL: func @integer_mul
func.func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: muli
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: muli
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_mul
// CHECK-PRIMITIVE-LABEL: func @complex_mul
func.func @complex_mul(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.mul
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.mul
  %0 = "mhlo.multiply"(%lhs, %rhs)
          : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_remainder
// CHECK-PRIMITIVE-LABEL: func @float_remainder
func.func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: remf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: remf
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
// CHECK-PRIMITIVE-LABEL: func @integer_remainder
func.func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: arith.remsi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.remsi
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @population_count_integer
// CHECK-PRIMITIVE-LABEL: func @population_count_integer
func.func @population_count_integer(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctpop
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctpop
  %0 = "mhlo.popcnt"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_sqrt
// CHECK-PRIMITIVE-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.sqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_rsqrt
// CHECK-PRIMITIVE-LABEL: func @float_rsqrt
func.func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: rsqrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_rsqrt
// CHECK-PRIMITIVE-LABEL: func @complex_rsqrt
func.func @complex_rsqrt(%operand: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.rsqrt
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.rsqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_cbrt
// CHECK-PRIMITIVE-LABEL: func @float_cbrt
func.func @float_cbrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.cbrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: %[[THIRD:.+]] = arith.constant 0.333333343
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ABS:.+]] = math.absf %[[IN]]
  // CHECK: %[[POW:.+]] = math.powf %[[ABS]], %[[THIRD]]
  // CHECK: %[[RESULT:.+]] = math.copysign %[[POW]], %[[IN]]
  // CHECK: linalg.yield %[[RESULT]]

  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.absf
  // CHECK-PRIMITIVE: math.powf
  // CHECK-PRIMITIVE: math.copysign
  func.return %tensor_result : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func @float_sub
// CHECK-PRIMITIVE-LABEL: func @float_sub
func.func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: subf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subf
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
// CHECK-PRIMITIVE-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: subi
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_sub
// CHECK-PRIMITIVE-LABEL: complex_sub
func.func @complex_sub(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sub
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sub
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_abs
// CHECK-PRIMITIVE-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK-SAME: {someattr}
  // CHECK: math.absf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE-NEXT: ins(
  // CHECK-PRIMITIVE-NEXT: outs(
  // CHECK-PRIMITIVE-SAME: {someattr}
  // CHECK-PRIMITIVE: math.absf
  %0 = "mhlo.abs"(%arg0) {someattr} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
// CHECK-PRIMITIVE-LABEL: func @float_exp
func.func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_exp
func.func @complex_exp(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.exp
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                 -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: expm1
  %0 = "mhlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_expm1
func.func @complex_expm1(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.expm1
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.expm1
  %0 = "mhlo.exponential_minus_one"(%arg0)
    : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log
func.func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log
func.func @complex_log(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xcomplex<f32>>)
                         -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log1p
// CHECK-PRIMITIVE-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log1p
// CHECK-PRIMITIVE-LABEL: func @complex_log1p
func.func @complex_log1p(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.log1p
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                  -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_logistic
// CHECK-PRIMITIVE-LABEL: func @float_logistic
func.func @float_logistic(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[C1:.*]] = arith.constant 1.{{.*}}e+00
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: f32, %{{.*}}: f32):
  // CHECK: %[[NEG_ARG:.*]] = arith.negf %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = math.exp %[[NEG_ARG]]
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = arith.addf %[[EXP_NEG_ARG]], %[[C1]]
  // CHECK: %[[RESULT:.*]] = arith.divf %[[C1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: arith.negf
  // CHECK-PRIMITIVE: math.exp
  // CHECK-PRIMITIVE: arith.addf
  // CHECK-PRIMITIVE: arith.divf
  %0 = "mhlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_logistic
func.func @complex_logistic(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[ARG:.*]]: complex<f32>, %{{.*}}: complex<f32>):
  // CHECK: %[[NEG_ARG:.*]] = complex.neg %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = complex.exp %[[NEG_ARG]]
  // CHECK: %[[CC1:.*]] = complex.create %[[C1]], %[[C0]] : complex<f32>
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = complex.add %[[EXP_NEG_ARG]], %[[CC1]]
  // CHECK: %[[RESULT:.*]] = complex.div %[[CC1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.logistic"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_ceil
// CHECK-PRIMITIVE-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.ceil
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ceil
  %0 = "mhlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
// CHECK-PRIMITIVE-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.floor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.floor
  %0 = "mhlo.floor"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
// CHECK-PRIMITIVE-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: negf
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: negf
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_neg
// CHECK-PRIMITIVE-LABEL: func @complex_neg
func.func @complex_neg(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.neg
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.neg
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xcomplex<f32>>)
                            -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_sign
// CHECK-PRIMITIVE-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sign
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sign
  %0 = "mhlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>>)
                          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_tanh
// CHECK-PRIMITIVE-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: tanh
  %0 = "mhlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_tanh
// CHECK-PRIMITIVE-LABEL: func @complex_tanh
func.func @complex_tanh(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.tanh"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: linalg.generic
  // CHECK: complex.tanh
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.tanh
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @integer_and
// CHECK-PRIMITIVE-LABEL: func @integer_and
func.func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: and
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: and
  %0 = "mhlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_or
// CHECK-PRIMITIVE-LABEL: func @integer_or
func.func @integer_or(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: or
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: or
  %0 = "mhlo.or"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_xor
// CHECK-PRIMITIVE-LABEL: func @integer_xor
func.func @integer_xor(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: xor
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: xor
  %0 = "mhlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @count_leading_zeros
// CHECK-PRIMITIVE-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: math.ctlz
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.ctlz
  %0 = "mhlo.count_leading_zeros"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_cmp
// CHECK-PRIMITIVE-LABEL: func @float_cmp
func.func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction EQ>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_ne
// CHECK-PRIMITIVE-LABEL: func @float_cmp_ne
func.func @float_cmp_ne(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction NE>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @float_cmp_totalorder
// CHECK-PRIMITIVE-LABEL: func @float_cmp_totalorder
func.func @float_cmp_totalorder(%lhs: tensor<2x2xbf16>,
                %rhs: tensor<2x2xbf16>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {
    comparison_direction = #mhlo<comparison_direction LT>,
    compare_type = #mhlo<comparison_type TOTALORDER>
  } : (tensor<2x2xbf16>, tensor<2x2xbf16>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: bf16, %[[RHS_IN:.*]]: bf16, %{{.*}}: i1):
// CHECK-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : i16
// CHECK-PRIMITIVE-DAG: %[[C32767:.*]] = arith.constant 32767 : i16
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE-NEXT: ins(
// CHECK-PRIMITIVE-NEXT: outs(
// CHECK-PRIMITIVE-NEXT: (%[[LHS_IN:[a-zA-Z0-9]*]]: bf16, %[[RHS_IN:.*]]: bf16) {
// CHECK-PRIMITIVE-NEXT:   %[[LHS_INT:.*]] = arith.bitcast %[[LHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_CMP:.*]] = arith.cmpi slt, %[[LHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SUB:.*]] = arith.subi %[[C32767]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[LHS_SELECT:.*]] = arith.select %[[LHS_CMP]], %[[LHS_SUB]], %[[LHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_INT:.*]] = arith.bitcast %[[RHS_IN]] : bf16 to i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_CMP:.*]] = arith.cmpi slt, %[[RHS_INT]], %[[C0]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SUB:.*]] = arith.subi %[[C32767]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RHS_SELECT:.*]] = arith.select %[[RHS_CMP]], %[[RHS_SUB]], %[[RHS_INT]] : i16
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_SELECT]], %[[RHS_SELECT]] : i16
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
// CHECK-PRIMITIVE-LABEL: func @int_cmp
func.func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction LT>}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.cmpi

// -----

// CHECK-LABEL: func @complex_cmp_eq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_eq
func.func @complex_cmp_eq(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction EQ>}
          : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.eq

// -----

// CHECK-LABEL: func @complex_cmp_neq
// CHECK-PRIMITIVE-LABEL: func @complex_cmp_neq
func.func @complex_cmp_neq(%lhs: tensor<2xcomplex<f64>>,
                      %rhs: tensor<2xcomplex<f64>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction NE>}
          : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>, %[[RESULT_OUT:.*]]: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: complex.neq

// -----

// CHECK-LABEL: func @float_cos
// CHECK-PRIMITIVE-LABEL: func @float_cos
func.func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_cos
// CHECK-PRIMITIVE-LABEL: func @complex_cos
func.func @complex_cos(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.cos
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_sin
// CHECK-PRIMITIVE-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: math.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_sin
// CHECK-PRIMITIVE-LABEL: func @complex_sin
func.func @complex_sin(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: linalg.generic
  // CHECK: complex.sin
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: complex.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
// CHECK-PRIMITIVE-LABEL: func @copy
// CHECK-PRIMITIVE-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @copy(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "mhlo.copy"(%input) : (tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>)
  func.return %0 : tensor<2x4x8xf32>
}
// CHECK: return [[ARG]] : tensor<2x4x8xf32>
// CHECK-PRIMITIVE: return [[ARG]] : tensor<2x4x8xf32>

// -----

// CHECK-LABEL: func @is_finte
// CHECK-PRIMITIVE-LABEL: func @is_finte
func.func @is_finte(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %0 = "mhlo.is_finite"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32
// CHECK-NEXT:   %[[ABS_X:.+]] = math.absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = arith.cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: math.absf
// CHECK-PRIMITIVE: arith.cmpf

// -----

// CHECK-LABEL: func @round_nearest_even
// CHECK-PRIMITIVE-LABEL: func @round_nearest_even
func.func @round_nearest_even(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.roundeven %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.roundeven
  %0 = "mhlo.round_nearest_even"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @round
// CHECK-PRIMITIVE-LABEL: func @round
func.func @round(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[ROUND:.+]] = math.round %[[IN]]
  // CHECK: linalg.yield %[[ROUND]]
  // CHECK-PRIMITIVE: linalg.map
  // CHECK-PRIMITIVE: math.round
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
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE-LABEL: func @select
// CHECK-PRIMITIVE: tensor.empty() : tensor<2x2xf32>
// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE-NEXT: ins(
// CHECK-PRIMITIVE-NEXT: outs(
// CHECK-PRIMITIVE-NEXT: (%[[PRED_IN:[a-zA-Z0-9]*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32) {
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : f32

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
// CHECK-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK:      linalg.generic
// CHECK-SAME:   indexing_maps = [#[[SCALAR_MAP]], #[[ID_MAP]], #[[ID_MAP]], #[[ID_MAP]]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel"]
// CHECK-SAME:   ins(%[[PRED]], %[[LHS]], %[[RHS]] : tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-SAME:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-SAME:   {someattr}
// CHECK:      ^bb0(%[[PRED_:.*]]: i1, %[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32, %{{.*}}: f32):
// CHECK:        %[[RES:.*]] = arith.select %[[PRED_]], %[[LHS_]], %[[RHS_]] : f32
// CHECK:        linalg.yield %[[RES]]

// CHECK-PRIMITIVE-LABEL: func @select_scalar_pred_dyn
// CHECK-PRIMITIVE-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
// CHECK-PRIMITIVE-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-PRIMITIVE-DAG:  %[[DIM:.*]] = tensor.dim %[[LHS]], %[[C1]]
// CHECK-PRIMITIVE-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK-PRIMITIVE-DAG:  %[[PRED_ELEM:.*]] = tensor.extract %[[PRED]]
// CHECK-PRIMITIVE:      linalg.map
// CHECK-PRIMITIVE-NEXT:   ins(%[[LHS]], %[[RHS]] : tensor<2x?xf32>, tensor<2x?xf32>)
// CHECK-PRIMITIVE-NEXT:   outs(%[[DST]] : tensor<2x?xf32>)
// CHECK-PRIMITIVE-SAME:   {someattr}
// CHECK-PRIMITIVE:      (%[[LHS_:.*]]: f32, %[[RHS_:.*]]: f32) {
// CHECK-PRIMITIVE:        %[[RES:.*]] = arith.select %[[PRED_ELEM]], %[[LHS_]], %[[RHS_]] : f32
// CHECK-PRIMITIVE:        linalg.yield %[[RES]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func.func @broadcast_scalar(%arg: tensor<f32>) -> tensor<4x2x1xf32> {
  %0 = "mhlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<f32>) -> tensor<4x2x1xf32>
  func.return %0: tensor<4x2x1xf32>
}
// CHECK: tensor.empty() : tensor<4x2x1xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<4x2x1xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE-NEXT: ins(
// CHECK-PRIMITIVE-NEXT: outs(
// CHECK-PRIMITIVE-NEXT: dimensions = [0, 1, 2]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg: tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32> {
  %0 = "mhlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32>
  func.return %0: tensor<4x2x1x4x?x16xf32>
}
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE: %[[DIM:.*]] = tensor.dim %{{.*}}, %[[C1]] : tensor<4x?x16xf32>
// CHECK-PRIMITIVE: %{{.*}} = tensor.empty(%[[DIM]]) : tensor<4x2x1x4x?x16xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

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
// CHECK: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]

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
// CHECK: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK: %[[RES:.*]] = linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_ui32
// CHECK-PRIMITIVE: tensor.collapse_shape
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6x4x5xi32>
// CHECK-PRIMITIVE: %[[RES:.*]] = linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [1, 2, 3]
// CHECK-PRIMITIVE: builtin.unrealized_conversion_cast %[[RES]] : tensor<7x10x6x4x5xi32> to tensor<7x10x6x4x5xui32>

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
// CHECK: tensor.empty() : tensor<1x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_one_to_one
// CHECK-PRIMITIVE-NOT: tensor.collapse_shape
// CHECK-PRIMITIVE-NOT: linalg.transpose
// CHECK-PRIMITIVE:     linalg.broadcast
// CHECK-PRIMITIVE:       dimensions = [1]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d2, d0, d1)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @broadcast_in_dim_with_transpose
func.func @broadcast_in_dim_with_transpose(
         %operand: tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[2, 0, 1]> : tensor<3xi64>}
         : (tensor<2x3x4xf32>) -> tensor<3x4x2x5xf32>
  func.return %0 : tensor<3x4x2x5xf32>
}
// CHECK: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_with_transpose
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2xf32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE:   permutation = [1, 2, 0]
// CHECK-PRIMITIVE: tensor.empty() : tensor<3x4x2x5xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [3]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_in_dim_scalar
func.func @broadcast_in_dim_scalar(%operand: tensor<f32>) -> tensor<7x10x6xf32> {
  %0 = "mhlo.broadcast_in_dim"(%operand)
        {broadcast_dimensions = dense<[]> : tensor<0xi64>}
        : (tensor<f32>) -> tensor<7x10x6xf32>
  func.return %0 : tensor<7x10x6xf32>
}
// CHECK: tensor.empty() : tensor<7x10x6xf32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @broadcast_in_dim_scalar
// CHECK-PRIMITIVE: tensor.empty() : tensor<7x10x6xf32>
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE:   dimensions = [0, 1, 2]

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

// CHECK-PRIMITIVE-LABEL: func @transpose
// CHECK-PRIMITIVE: linalg.transpose

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose_dynamic
func.func @transpose_dynamic(%arg0: tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>, someattr}
        : (tensor<?x?x9x?xi32>) -> tensor<?x?x?x9xi32>
  func.return %0 : tensor<?x?x?x9xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins(%arg0 : tensor<?x?x9x?xi32>) outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-SAME: {someattr}

// CHECK-PRIMITIVE-LABEL: func @transpose_dynamic
// CHECK-PRIMITIVE-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-PRIMITIVE-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-PRIMITIVE-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-PRIMITIVE: %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
// CHECK-PRIMITIVE: %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
// CHECK-PRIMITIVE: %[[D3:.*]] = tensor.dim %arg0, %[[C3]]
// CHECK-PRIMITIVE: %[[INIT:.*]] = tensor.empty(%[[D1]], %[[D0]], %[[D3]]) : tensor<?x?x?x9xi32>
// CHECK-PRIMITIVE: linalg.transpose
// CHECK-PRIMITIVE-NEXT: ins(%arg0 : tensor<?x?x9x?xi32>)
// CHECK-PRIMITIVE-NEXT: outs(%[[INIT]] : tensor<?x?x?x9xi32>)
// CHECK-PRIMITIVE-NEXT: permutation = [1, 0, 3, 2]
// CHECK-PRIMITIVE-SAME: {someattr}

// -----

// CHECK-LABEL: func @real_dynamic_slice
// CHECK-SAME: (%[[OPERAND:.*]]: tensor<256x?xf32>, %[[START_INDICES:.*]]: tensor<2xindex>, %[[LIMIT_INDICES:.*]]: tensor<2xindex>, %[[STRIDES:.*]]: tensor<2xindex>)
func.func @real_dynamic_slice(%input: tensor<256x?xf32>, %start_indices: tensor<2xindex>, %limit_indices: tensor<2xindex>, %strides: tensor<2xindex>) -> tensor<256x?xf32> {
  %0 = "mhlo.real_dynamic_slice"(%input, %start_indices, %limit_indices, %strides) : (tensor<256x?xf32>, tensor<2xindex>, tensor<2xindex>, tensor<2xindex>) -> tensor<256x?xf32>
  func.return %0 : tensor<256x?xf32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1

// Fetch start index, limit index and stride.
// CHECK-DAG: %[[START0:.*]] = tensor.extract %[[START_INDICES]][%[[C0]]]
// CHECK-DAG: %[[STRIDE0:.*]] = tensor.extract %[[STRIDES]][%[[C0]]]

// Clamp starting index : 0 <= start <= ub
// CHECK-DAG: %[[MAX0:.*]] = arith.maxsi %[[START0]], %[[C0]] : index
// CHECK-DAG: %[[MIN0:.*]] = arith.minsi %[[MAX0]], %[[C0]] : index

// CHECK-DAG: %[[START1:.*]] = tensor.extract %[[START_INDICES]][%[[C1]]]
// CHECK-DAG: %[[LIMIT1:.*]] = tensor.extract %[[LIMIT_INDICES]][%[[C1]]]
// CHECK-DAG: %[[STRIDE1:.*]] = tensor.extract %[[STRIDES]][%[[C1]]]

// 2.2. Since 1-th dimension of result is unknown we compute result size at 1-th
//      dimension as size[1] = (limit - start)/stride
// CHECK-DAG: %[[DELTA1:.*]] = arith.subi %[[LIMIT1]], %[[START1]] : index
// CHECK-DAG: %[[SIZE1:.*]] = arith.ceildivui %[[DELTA1]], %[[STRIDE1]] : index

// 2.3. Compute upper bound for starting index = operand_dim[1] - size[1].
//      where, size[1] is computed at step 2.2
// CHECK-DAG: %[[OPERAND_DIM1:.*]] = tensor.dim %[[OPERAND]], %[[C1]] : tensor<256x?xf32>
// CHECK-DAG: %[[UB:.*]] = arith.subi %[[OPERAND_DIM1]], %[[SIZE1]] : index

// 2.4. Clamp starting index : 0 <= start <= ub
//      where upper bound (ub) is computed at step 2.3
// CHECK-DAG: %[[MAX1:.*]] = arith.maxsi %[[START1]], %[[C0]] : index
// CHECK-DAG: %[[MIN1:.*]] = arith.minsi %[[MAX1]], %[[UB]] : index

// CHECK-DAG: %[[SLICE:.*]] = tensor.extract_slice %[[OPERAND]][%[[MIN0]], %[[MIN1]]] [256, %[[SIZE1]]] [%[[STRIDE0]], %[[STRIDE1]]] : tensor<256x?xf32> to tensor<256x?xf32>
// CHECK: return %[[SLICE]] : tensor<256x?xf32>

// -----

// Verify that legalization of real_dynamic_slice legalization with integer
// dims work & passes verification.
// CHECK-LABEL: real_dynamic_slice_with_int
func.func public @real_dynamic_slice_with_int(%arg0: tensor<10xi32> , %arg1: tensor<1xi32> ) -> tensor<?xi32> {
  %0 = mhlo.constant dense<0> : tensor<1xi32>
  %1 = mhlo.constant dense<1> : tensor<1xi32>
  %2 = mhlo.constant dense<0> : tensor<i32>
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

// CHECK-LABEL: func @reshape_empty
func.func @reshape_empty(%arg0: tensor<7x0xf64>) -> tensor<0x42x101xf64> {
  %0 = mhlo.reshape %arg0 : (tensor<7x0xf64>) -> tensor<0x42x101xf64>
  return %0 : tensor<0x42x101xf64>
}

// CHECK: %[[INIT:.*]] = tensor.empty
// CHECK: return %[[INIT]]

// -----

// CHECK-LABEL: func @minf
func.func @minf(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.minimum"(%lhs, %rhs) {someattr}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: linalg.generic
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.minf %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.minf

// -----

// CHECK-LABEL: func @maxi
func.func @maxi(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}
// CHECK: tensor.empty() : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxsi

// -----

// CHECK-LABEL: func @maxu
func.func @maxu(%lhs: tensor<2x2xui32>, %rhs: tensor<2x2xui32>) -> tensor<2x2xui32> {
  %0 = "mhlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xui32>
  func.return %0 : tensor<2x2xui32>
}
// CHECK: tensor.empty() : tensor<2x2xi32>
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxui %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxui

// -----

// CHECK-LABEL: func @maxi1
func.func @maxi1(%lhs: tensor<?x?xi1>, %rhs: tensor<?x?xi1>) -> tensor<?x?xi1> {
  %0 = "mhlo.maximum"(%lhs, %rhs)
          : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i1, %[[RHS_IN:.*]]: i1, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxui %[[LHS_IN]], %[[RHS_IN]] : i1
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.maxui

// -----

// CHECK-DAG: #[[MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @add_scalar
func.func @add_scalar(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32, %{{.*}}: f32):
// CHECK: %[[RESULT:.*]] = arith.addf %[[LHS]], %[[RHS]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.addf

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
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK-LABEL: func @bitcast_convert_dynamic
func.func @bitcast_convert_dynamic(%input: tensor<?x?xi32>) -> tensor<?x?xf32> {
  %result = "mhlo.bitcast_convert"(%input) : (tensor<?x?xi32>) -> tensor<?x?xf32>
  func.return %result : tensor<?x?xf32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.bitcast %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// CHECK-PRIMITIVE: linalg.map
// CHECK-PRIMITIVE: arith.bitcast

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @bitcast_convert_expand
func.func @bitcast_convert_expand(%input: tensor<6xi32>) -> tensor<6x4xi8> {
  %result = "mhlo.bitcast_convert"(%input) : (tensor<6xi32>) -> tensor<6x4xi8>
  func.return %result : tensor<6x4xi8>
}

// CHECK: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: tensor.empty() : tensor<6x4xi8>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "parallel"]}
// CHECK:    ^bb0(%[[IN:.*]]: i32, %[[OUT:.*]]: i8):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i32
// CHECK:      %[[SHIFT:.*]] = arith.shrui %[[IN]], %[[AMT]] : i32
// CHECK:      %[[TRUNC:.*]] = arith.trunci %[[SHIFT]] : i32 to i8
// CHECK:      linalg.yield %[[TRUNC]] : i8
// CHECK:    } -> tensor<6x4xi8>
// CHECK:    return %[[RESULT]] : tensor<6x4xi8>

// -----

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-LABEL: func @bitcast_convert_contract
func.func @bitcast_convert_contract(%input: tensor<7x4xi8>) -> tensor<7xi32> {
  %result = "mhlo.bitcast_convert"(%input) : (tensor<7x4xi8>) -> tensor<7xi32>
  func.return %result : tensor<7xi32>
}
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant 8 : i32
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<7xi32>
// CHECK: linalg.fill ins(%[[C0]] : i32) outs(%[[EMPTY]] : tensor<7xi32>) -> tensor<7xi32>
// CHECK: %[[RESULT:.*]] = linalg.generic {
// CHECK:    indexing_maps = [#[[MAP0]], #[[MAP1]]],
// CHECK:    iterator_types = ["parallel", "reduction"]}
// CHECK:    ^bb0(%[[IN:.*]]: i8, %[[OUT:.*]]: i32):
// CHECK:      %[[IOTA:.*]] = linalg.index 1 : index
// CHECK:      %[[IOTA_CASTED:.*]] = arith.index_cast %[[IOTA]] : index to i32
// CHECK:      %[[AMT:.*]] = arith.muli %[[IOTA_CASTED]], %[[C8]] : i3
// CHECK:      %[[EXT:.*]] = arith.extui %[[IN]] : i8 to i32
// CHECK:      %[[SHIFT:.*]] = arith.shli %[[EXT]], %[[AMT]] : i32
// CHECK:      %[[OR:.*]] = arith.ori %[[SHIFT]], %[[OUT]] : i32
// CHECK:      linalg.yield %[[OR]] : i32
// CHECK: } -> tensor<7xi32>
// CHECK: return %[[RESULT]] : tensor<7xi32>

// -----

// CHECK-LABEL: func @convert_i1_to_f32
func.func @convert_i1_to_f32(%input: tensor<2x2xi1>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi1>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi ne, %[[OPERAND_IN]], %[[ZERO]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_ui32_to_i1
func.func @convert_ui32_to_i1(%input: tensor<2x2xui32>) -> tensor<2x2xi1> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xui32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: builtin.unrealized_conversion_cast %arg0 : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi ne, %[[OPERAND_IN]], %[[ZERO]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i1
func.func @convert_f32_to_i1(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %result : tensor<2x2xi1>
}
// CHECK-DAG:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: i1):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[OPERAND_IN]], %[[ZERO]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @convert_f32_to_i32
func.func @convert_f32_to_i32(%input: tensor<2x2xf32>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.fptosi %[[OPERAND_IN]] : f32 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_f32_to_ui16
func.func @convert_f32_to_ui16(%input: tensor<2x2xf32>) -> tensor<2x2xui16> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xui16>
  func.return %result : tensor<2x2xui16>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: i16):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.fptoui %[[OPERAND_IN]] : f32 to i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x2xi16> to tensor<2x2xui16>

// -----

// CHECK-LABEL: func @convert_bf16_to_f16
func.func @convert_bf16_to_f16(%input: tensor<2x2xbf16>) -> tensor<2x2xf16> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xbf16>) -> tensor<2x2xf16>
  func.return %result : tensor<2x2xf16>
}
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: bf16, %{{.*}}: f16):
// CHECK-NEXT:   %[[EXT:.*]] = arith.extf %[[OPERAND_IN]] : bf16 to f32
// CHECK-NEXT:   %[[TRUNC:.*]] = arith.truncf %[[EXT]] : f32 to f16
// CHECK-NEXT:   linalg.yield %[[TRUNC]] : f16

// -----

// CHECK-LABEL: func @convert_c64_to_c128
func.func @convert_c64_to_c128(%input: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f64>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f64>>
  func.return %result : tensor<2x2xcomplex<f64>>
}
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %{{.*}}: complex<f64>):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK-DAG:  %[[IMAG:.*]] = complex.im %[[OPERAND_IN]]
// CHECK-DAG:  %[[REAL_RESULT:.*]] = arith.extf %[[REAL]] : f32 to f64
// CHECK-DAG:  %[[IMAG_RESULT:.*]] = arith.extf %[[IMAG]] : f32 to f64
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[REAL_RESULT]], %[[IMAG_RESULT]]
// CHECK:      linalg.yield %[[RESULT]] : complex<f64>

// -----

// CHECK-LABEL: func @convert_c128_to_c64
func.func @convert_c128_to_c64(%input: tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xcomplex<f32>>
  func.return %result : tensor<2x2xcomplex<f32>>
}
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f64>, %{{.*}}: complex<f32>):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK-DAG:  %[[IMAG:.*]] = complex.im %[[OPERAND_IN]]
// CHECK-DAG:  %[[REAL_RESULT:.*]] = arith.truncf %[[REAL]] : f64 to f32
// CHECK-DAG:  %[[IMAG_RESULT:.*]] = arith.truncf %[[IMAG]] : f64 to f32
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[REAL_RESULT]], %[[IMAG_RESULT]]
// CHECK:      linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @convert_c64_to_f32
func.func @convert_c64_to_f32(%input: tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xf32>
  func.return %result : tensor<2x2xf32>
}
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f32>, %{{.*}}: f32):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK:      linalg.yield %[[REAL]] : f32

// -----

// CHECK-LABEL: func @convert_c128_to_i32
func.func @convert_c128_to_i32(%input: tensor<2x2xcomplex<f64>>) -> tensor<2x2xi32> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xcomplex<f64>>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: complex<f64>, %{{.*}}: i32):
// CHECK-DAG:  %[[REAL:.*]] = complex.re %[[OPERAND_IN]]
// CHECK-NEXT:  %[[RESULT:.*]] = arith.fptosi %[[REAL]] : f64 to i32
// CHECK-NEXT:  linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_f32_to_c64
func.func @convert_f32_to_c64(%input: tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xcomplex<f32>>
  func.return %result : tensor<2x2xcomplex<f32>>
}
// CHECK-DAG:  %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32, %{{.*}}: complex<f32>):
// CHECK-DAG:  %[[RESULT:.*]] = complex.create %[[OPERAND_IN]], %[[IMAG]]
// CHECK:      linalg.yield %[[RESULT]] : complex<f32>

// -----

// CHECK-LABEL: func @convert_i32_to_c128
func.func @convert_i32_to_c128(%input: tensor<2x2xi32>) -> tensor<2x2xcomplex<f64>> {
  %result = "mhlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xcomplex<f64>>
  func.return %result : tensor<2x2xcomplex<f64>>
}
// CHECK-DAG:  %[[IMAG:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:      tensor.empty
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32, %{{.*}}: complex<f64>):
// CHECK-DAG:  %[[REAL:.*]] = arith.sitofp %[[OPERAND_IN]] : i32 to f64
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK: tensor.empty
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
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%{{.*}}: complex<f32>):
// CHECK-NEXT:   %[[INDEX:.*]] = linalg.index 1
// CHECK-NEXT:   %[[INT_CAST:.*]] = arith.index_cast %[[INDEX]] : index to i32
// CHECK-NEXT:   %[[FLOAT_CAST:.*]] = arith.sitofp %[[INT_CAST]] : i32 to f32
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
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xf32>
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
// CHECK: %[[V1:.*]] = tensor.extract %[[SHAPE]][%c0]
// CHECK: %[[I1:.*]] = arith.index_cast %[[V1]] : i32 to index
// CHECK: %[[V2:.*]] = tensor.extract %[[SHAPE]][%c1]
// CHECK: %[[I2:.*]] = arith.index_cast %[[V2]] : i32 to index
// CHECK: tensor.empty(%[[I1]], %[[I2]]) : tensor<?x?x8xi32>
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
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shli %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[ZERO]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_arithmetic(%lhs: tensor<2x2xi32>,
                             %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_arithmetic"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_arithmetic
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK-DAG:    %[[MAX_SHIFT:.*]] = arith.constant 31
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shrsi %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[MAX_SHIFTED:.*]] = arith.shrsi %[[LHS]], %[[MAX_SHIFT]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[MAX_SHIFTED]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @shift_right_logical(%lhs: tensor<2x2xi32>,
                          %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %result = "mhlo.shift_right_logical"(%lhs, %rhs)
      : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %result : tensor<2x2xi32>
}
// CHECK-LABEL: func @shift_right_logical
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0
// CHECK-DAG:    %[[BITS:.*]] = arith.constant 32
// CHECK: tensor.empty
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %{{.*}}: i32):
// CHECK-DAG:    %[[SHIFT:.*]] = arith.shrui %[[LHS]], %[[RHS]] : i32
// CHECK-DAG:    %[[NOT_SATURATING:.*]] = arith.cmpi ult, %[[RHS]], %[[BITS]]
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[NOT_SATURATING]], %[[SHIFT]], %[[ZERO]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @constant
func.func @constant() -> tensor<i32> {
  %result = "mhlo.constant"() {
    value = dense<10> : tensor<i32>
  } : () -> (tensor<i32>)
  func.return %result : tensor<i32>
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
  // CHECK-SAME:   %[[ITER1:.*]] = %[[ARG0]],
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
  // CHECK: %[[VAL5:.*]] = arith.extui %[[LHS_ONE]] : i1 to i32
  // CHECK: %[[VAL6:.*]] = arith.select %[[RHS_EVEN]], %c1{{.*}}, %c-1
  // CHECK: %[[VAL7:.*]] = arith.select %[[LHS_NEG_ONE]], %[[VAL6]], %[[VAL5]]
  // CHECK: %[[RESULT:.*]] = arith.select %[[RHS_NEG]], %[[VAL7]], %[[FOR_RESULT]]#0
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "mhlo.power"(%lhs, %rhs) : (tensor<2x2xi32>,
                                   tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK: #[[OPERAND_MAP:.*]] = affine_map<(d0) -> ()>
// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-SAME: [[SHAPE:%.*]]: tensor<1xindex>
func.func @dynamic_broadcast_in_dim(%shape: tensor<1xindex>) -> tensor<?xf32> {
  %cst = mhlo.constant dense<0x7F800000> : tensor<f32>
  %result = "mhlo.dynamic_broadcast_in_dim"(%cst, %shape) {
     broadcast_dimensions = dense<> : tensor<0xi64>, someattr
  } : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
  func.return %result : tensor<?xf32>
}
// CHECK: [[CST:%.*]] = arith.constant dense
// CHECK: [[INIT:%.*]] = tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[CST]] : tensor<f32>) outs([[INIT]] : tensor<?xf32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// CHECK-PRIMITIVE: [[CST:%.*]] = arith.constant dense
// CHECK-PRIMITIVE: [[INIT:%.*]] = tensor.empty
// CHECK-PRIMITIVE: linalg.broadcast
// CHECK-PRIMITIVE-NEXT: ins([[CST]]
// CHECK-PRIMITIVE-NEXT: outs([[INIT]]

// -----

// CHECK: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

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
// CHECK: [[INIT:%.*]] = tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[SCALAR]] : tensor<f32>) outs([[INIT]] : tensor<?x32xf32>)
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// CHECK-PRIMITIVE: tensor.empty
// CHECK-PRIMITIVE: linalg.broadcast

// -----

// CHECK: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

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
// CHECK: [[INIT:%.*]] = tensor.empty
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[VECTOR]] : tensor<42xf32>) outs([[INIT]] :
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32, %[[RESULT:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// CHECK-PRIMITIVE: tensor.empty
// CHECK-PRIMITIVE: %[[RESULT:.*]] = linalg.broadcast
// CHECK-PRIMITIVE: tensor.cast %[[RESULT]] : tensor<?x42x?xf32> to tensor<?x?x?xf32>

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim(
// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// Note: this test requires no checks. The tensor.empty verifier will
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
// CHECK: [[INIT:%.*]] = tensor.empty
// CHECK: [[GENERIC:%.*]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-SAME: ins([[CST]] : tensor<i32>) outs([[INIT]] : tensor<?xi32>)
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: i32, %[[RESULT:.*]]: i32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : i32
// CHECK: [[RES:%.*]] = builtin.unrealized_conversion_cast [[GENERIC]] : tensor<?xi32> to tensor<?xui32>
// CHECK: return [[RES]] : tensor<?xui32>

// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// CHECK-PRIMITIVE: tensor.empty
// CHECK-PRIMITIVE: %[[BROADCASTED:.*]] = linalg.broadcast
// CHECK-PRIMITIVE: %[[RES:.*]] = builtin.unrealized_conversion_cast %[[BROADCASTED]]
// CHECK-PRIMITIVE-SAME:  tensor<?xi32> to tensor<?xui32>
// CHECK-PRIMITIVE: return %[[RES]] : tensor<?xui32>

// -----

// CHECK: #[[ARG_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (0, 0, d3, d4, 0, d6)>
// CHECK: #[[RES_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?x?x1x42xf32>, %[[SHAPE:.*]]: tensor<7xindex>
// CHECK-PRIMITIVE-LABEL: func @dynamic_broadcast_in_dim
// CHECK-PRIMITIVE-SAME:  %[[ARG:.*]]: tensor<?x?x?x?x1x42xf32>, %[[SHAPE:.*]]: tensor<7xindex>
func.func @dynamic_broadcast_in_dim(%arg: tensor<?x?x?x?x1x42xf32>,
    %shape: tensor<7xindex>) -> tensor<?x?x?x?x?x?x?xf32> {
  %result = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape) {
      broadcast_dimensions = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>,
      known_expanding_dimensions = dense<[0, 1]> : tensor<2xi64>,
      known_nonexpanding_dimensions = dense<[2, 3]> : tensor<2xi64> }
      : (tensor<?x?x?x?x1x42xf32>, tensor<7xindex>) -> tensor<?x?x?x?x?x?x?xf32>
  func.return %result : tensor<?x?x?x?x?x?x?xf32>
}

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
// CHECK-DAG:  %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[DIM4]], %[[DIM5]], %[[DIM6]]) : tensor<?x?x?x?x?x?x?xf32>
// CHECK:      %[[RES:.*]] = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[ARG_MAP]], #[[RES_MAP]]],
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]}
// CHECK-SAME:     ins(%[[ARG]] : tensor<?x?x?x?x1x42xf32>) outs(%[[INIT]] : tensor<?x?x?x?x?x?x?xf32>) {
// CHECK:      ^bb0(%[[ARG_:.*]]: f32, %{{.*}}: f32):
// CHECK:        linalg.yield %[[ARG_]]
// CHECK:      return %[[RES]]

// CHECK-PRIMITIVE-DAG:  %[[C0:.*]] = arith.constant 0
// CHECK-PRIMITIVE-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-PRIMITIVE-DAG:  %[[C2:.*]] = arith.constant 2
// CHECK-PRIMITIVE-DAG:  %[[C3:.*]] = arith.constant 3
// CHECK-PRIMITIVE-DAG:  %[[C4:.*]] = arith.constant 4
// CHECK-PRIMITIVE-DAG:  %[[C5:.*]] = arith.constant 5
// CHECK-PRIMITIVE:      tensor.cast %[[ARG]]
// CHECK-PRIMITIVE-SAME:   tensor<?x?x?x?x1x42xf32> to tensor<1x1x?x?x1x42xf32>
// CHECK-PRIMITIVE:      %[[COLLAPSED:.*]] = tensor.collapse_shape
// CHECK-PRIMITIVE-SAME{literal}:   [[0, 1, 2], [3], [4, 5]]
// CHECK-PRIMITIVE-SAME:   tensor<1x1x?x?x1x42xf32> into tensor<?x?x42xf32>
// CHECK-PRIMITIVE-DAG:  %[[DIM0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
// CHECK-PRIMITIVE-DAG:  %[[DIM1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
// CHECK-PRIMITIVE-DAG:  %[[DIM2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
// CHECK-PRIMITIVE-DAG:  %[[DIM3:.*]] = tensor.extract %[[SHAPE]][%[[C3]]]
// CHECK-PRIMITIVE-DAG:  %[[DIM4:.*]] = tensor.extract %[[SHAPE]][%[[C4]]]
// CHECK-PRIMITIVE-DAG:  %[[DIM5:.*]] = tensor.extract %[[SHAPE]][%[[C5]]]
// CHECK-PRIMITIVE:      %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]], %[[DIM3]], %[[DIM4]], %[[DIM5]]) : tensor<?x?x?x?x?x?x42xf32>
// CHECK-PRIMITIVE:      %[[BROADCASTED:.*]] = linalg.broadcast
// CHECK-PRIMITIVE-NEXT:   ins(%[[COLLAPSED]] : tensor<?x?x42xf32>)
// CHECK-PRIMITIVE-NEXT:   outs(%[[INIT]] : tensor<?x?x?x?x?x?x42xf32>)
// CHECK-PRIMITIVE-NEXT:   dimensions = [0, 1, 2, 5]
// CHECK-PRIMITIVE:      %[[RES:.*]] = tensor.cast %[[BROADCASTED]]
// CHECK-PRIMITIVE-SAME:    tensor<?x?x?x?x?x?x42xf32> to tensor<?x?x?x?x?x?x?xf32>
// CHECK-PRIMITIVE:      return %[[RES]]

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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]]) : tensor<2x?x
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]])
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
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D1]])
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
// CHECK: %[[INIT:.*]] = tensor.empty()
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
// CHECK: %[[INIT:.*]] = tensor.empty()
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x3xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}
// CHECK-LABEL: func @dot_general_batch_matmul(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xf32>, %[[ARG1:.*]]: tensor<?x3x?xf32>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<?x?x3xi8>, tensor<?x3x?xi8>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_i8_i8_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xi8>, %[[ARG1:.*]]: tensor<?x3x?xi8>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<?x?x3xi16>, tensor<?x3x?xi16>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_i16_i16_i32(
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x?x3xi16>, %[[ARG1:.*]]: tensor<?x3x?xi16>)
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C2]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
    : (tensor<2x16x32xf32>, tensor<2x32x32xf32>) -> tensor<2x16x32xf32>
  func.return %0 : tensor<2x16x32xf32>
}
// CHECK-LABEL: func @dot_general_batch_matmul_large(
// CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: tensor<2x16x32xf32>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: tensor<2x32x32xf32>)
// CHECK: %[[INIT:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT]]
// CHECK: %[[DOT:.*]] = linalg.batch_matmul
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x16x32xf32>, tensor<2x32x32xf32>)
// CHECK-SAME: outs(%[[FILL]] : tensor<2x16x32xf32>)

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-LABEL: func @einsum_basic
func.func @einsum_basic(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x5x6xf32>) -> tensor<3x4x6xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ijk,ikm->ijm", someattr}: (tensor<3x4x5xf32>, tensor<3x5x6xf32>) -> tensor<3x4x6xf32>
  func.return %0 : tensor<3x4x6xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x5x6xf32>)
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x6xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
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

func.func @dot_general_batch_matvec(%arg0: tensor<?x?x3xf32>,
                                    %arg1: tensor<?x3xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
    someattr
  } : (tensor<?x?x3xf32>, tensor<?x3xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: func @dot_general_batch_matvec(
// CHECK: linalg.generic

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func @einsum_pointwisemul
func.func @einsum_pointwisemul(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "abc,abc->abc"} : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
  func.return %0 : tensor<3x4x5xf32>
}
// CHECK-SAME:  (%[[LHS:.*]]: tensor<3x4x5xf32>, %[[RHS:.*]]: tensor<3x4x5xf32>)
// CHECK: tensor.empty() : tensor<3x4x5xf32>
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
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<7x5xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
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
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x4x5x6x8xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
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
// CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<1x512x256xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
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
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[DIM0:.+]] = tensor.dim %[[LHS]], %[[C0]] : tensor<?x?x4xf32>
// CHECK: %[[DIM1:.+]] = tensor.dim %[[LHS]], %[[C1]] : tensor<?x?x4xf32>
// CHECK: %[[DIM2:.+]] = tensor.dim %[[RHS]], %[[C1:.+]] : tensor<4x?xf32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM0]], %[[DIM1]], %[[DIM2]]) : tensor<?x?x?xf32>
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
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) outs(%[[INIT]] : tensor<4xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
  // CHECK:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
  // CHECK:   linalg.yield %[[MIN]]
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
  // CHECK: %[[INIT:.*]] = tensor.empty
  // CHECK: %[[RESULT:.*]] = linalg.generic {{.*}} ins(%[[LB]], %[[X]], %[[UB]] : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%[[INIT]] : tensor<?xf32>)
  // CHECK: ^bb0(%[[SCALAR_LB:.*]]: f32, %[[SCALAR_X:.*]]: f32, %[[SCALAR_UB:.*]]: f32, %{{.*}}: f32):
  // CHECK:   %[[MAX:.*]] = arith.maxf %[[SCALAR_LB]], %[[SCALAR_X]] : f32
  // CHECK:   %[[MIN:.*]] = arith.minf %[[MAX]], %[[SCALAR_UB]] : f32
  // CHECK:   linalg.yield %[[MIN]]
  // CHECK: } -> tensor<?xf32>
  // CHECK: return %[[RESULT]] : tensor<?xf32>
  %0 = "mhlo.clamp"(%lb, %x, %ub) : (tensor<?xf32>, tensor<?xf32>,
      tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @map_compare(%arg0: tensor<?xcomplex<f32>>,
                       %arg1: tensor<?xcomplex<f32>>) -> tensor<?xi1> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<complex<f32>>, %arg3: tensor<complex<f32>>):
    %1 = mhlo.real %arg2 : (tensor<complex<f32>>) -> tensor<f32>
    %2 = mhlo.real %arg3 : (tensor<complex<f32>>) -> tensor<f32>
    %3 = "mhlo.compare"(%1, %2)
       {comparison_direction = #mhlo<comparison_direction EQ>}
       : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%3) : (tensor<i1>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>}
  : (tensor<?xcomplex<f32>>, tensor<?xcomplex<f32>>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// CHECK-LABEL: @map_compare
// CHECK-SAME: %[[ARG0:.*]]: tensor<?xcomplex<f32>>,
// CHECK-SAME: %[[ARG1:.*]]: tensor<?xcomplex<f32>>)

// CHECK: %[[INIT:.+]] = tensor.empty
// CHECK: %[[MAP:.+]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]}
// CHECK-SAME: ins(%[[ARG0]], %[[ARG1]]
// CHECK-SAME: outs(%[[INIT]] : tensor<?xi1>) {
// CHECK:  ^bb0(%[[A:.+]]: complex<f32>, %[[B:.+]]: complex<f32>, %{{.+}}: i1):
// CHECK: %[[RE1:.+]] = complex.re %[[A]] : complex<f32>
// CHECK: %[[RE2:.+]] = complex.re %[[B]] : complex<f32>
// CHECK: %[[CMP:.+]] = arith.cmpf oeq, %[[RE1]], %[[RE2]] : f32
// CHECK: linalg.yield %[[CMP]] : i1
// CHECK: }
// CHECK: return %[[MAP]] : tensor<?xi1>

// CHECK-PRIMITIVE-LABEL: @map_compare
// CHECK-PRIMITIVE-SAME: %[[ARG0:.*]]: tensor<?xcomplex<f32>>,
// CHECK-PRIMITIVE-SAME: %[[ARG1:.*]]: tensor<?xcomplex<f32>>)

// CHECK-PRIMITIVE: %[[INIT:.+]] = tensor.empty
// CHECK-PRIMITIVE: %[[MAP:.+]] = linalg.map
// CHECK-PRIMITIVE-NEXT: ins(%[[ARG0]], %[[ARG1]]
// CHECK-PRIMITIVE-NEXT: outs(%[[INIT]] : tensor<?xi1>)
// CHECK-PRIMITIVE-NEXT: (%[[A:.+]]: complex<f32>, %[[B:.+]]: complex<f32>) {
// CHECK-PRIMITIVE: %[[RE1:.+]] = complex.re %[[A]] : complex<f32>
// CHECK-PRIMITIVE: %[[RE2:.+]] = complex.re %[[B]] : complex<f32>
// CHECK-PRIMITIVE: %[[CMP:.+]] = arith.cmpf oeq, %[[RE1]], %[[RE2]] : f32
// CHECK-PRIMITIVE: linalg.yield %[[CMP]] : i1
// CHECK-PRIMITIVE: }
// CHECK-PRIMITIVE: return %[[MAP]] : tensor<?xi1>
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
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-SAME: {someattr}
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE-LABEL: @reduce_add
// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-PRIMITIVE-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-PRIMITIVE-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK-PRIMITIVE: linalg.reduce
// CHECK-PRIMITIVE-NEXT: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-PRIMITIVE-NEXT: outs(%[[FILL_TENSOR]] : tensor<5xi32>)
// CHECK-PRIMITIVE-NEXT: dimensions = [1]  {someattr}
// CHECK-PRIMITIVE-NEXT: (%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: @reduce_add_unranked
// CHECK-PRIMITIVE-LABEL: @reduce_add_unranked
func.func @reduce_add_unranked(%arg0: tensor<*xi32>, %arg1: tensor<i32>) -> tensor<*xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}
// CHECK: mhlo.reduce
// CHECK-PRIMITIVE: mhlo.reduce

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
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// CHECK-PRIMITIVE-LABEL: @reduce_dim0
// CHECK-PRIMITIVE-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-PRIMITIVE-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-PRIMITIVE-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK-PRIMITIVE: linalg.reduce
// CHECK-PRIMITIVE-NEXT: ins(%{{.*}}tensor<5x4xi32>)
// CHECK-PRIMITIVE-NEXT: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-PRIMITIVE-NEXT: dimensions = [0]
// CHECK-PRIMITIVE-NEXT: (%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:   %[[RESULT:.*]] = arith.maxsi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-PRIMITIVE-NEXT:   linalg.yield %[[RESULT]] : i32

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
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%{{.*}}{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<1x10xf32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<1xf32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addf %[[RHS_IN]], %[[LHS_IN]] : f32
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
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty()
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<5x4x3xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<4xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @reduce_lexicographic_min_complex(%arg0: tensor<?x3x4xcomplex<f64>>,
                                            %arg1: tensor<complex<f64>>)
  -> tensor<complex<f64>> {
  %0 = mhlo.reduce(%arg0 init: %arg1)
   across dimensions = [0, 1, 2]
   : (tensor<?x3x4xcomplex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
   reducer(%arg3: tensor<complex<f64>>, %arg4: tensor<complex<f64>>)  {
    %1 = mhlo.real %arg3 : (tensor<complex<f64>>) -> tensor<f64>
    %2 = mhlo.convert %arg4 : (tensor<complex<f64>>) -> tensor<f64>
    %3 = "mhlo.compare"(%1, %2)
      {comparison_direction = #mhlo<comparison_direction EQ>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %4 = mhlo.imag %arg3 : (tensor<complex<f64>>) -> tensor<f64>
    %5 = mhlo.imag %arg4 : (tensor<complex<f64>>) -> tensor<f64>
    %6 = "mhlo.compare"(%4, %5)
      {comparison_direction = #mhlo<comparison_direction LT>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %7 = "mhlo.compare"(%1, %2)
      {comparison_direction = #mhlo<comparison_direction LT>}
      : (tensor<f64>, tensor<f64>) -> tensor<i1>
    %8 = "mhlo.select"(%3, %6, %7)
      : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<i1>
    %9 = "mhlo.select"(%8, %arg3, %arg4)
      : (tensor<i1>, tensor<complex<f64>>, tensor<complex<f64>>)
      -> tensor<complex<f64>>
    "mhlo.return"(%9) : (tensor<complex<f64>>) -> ()
  }
  return %0 : tensor<complex<f64>>
}

// CHECK-LABEL: @reduce_lexicographic_min_complex
// CHECK: linalg.generic
// CHECK: complex.re
// CHECK: complex.re
// CHECK: arith.cmpf
// CHECK: complex.im
// CHECK: complex.im
// CHECK: arith.cmpf
// CHECK: arith.cmpf
// CHECK: arith.select

// CHECK-PRIMITIVE-LABEL: @reduce_lexicographic_min_complex
// CHECK-PRIMITIVE: linalg.reduce
// CHECK-PRIMITIVE: complex.re
// CHECK-PRIMITIVE: complex.re
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: complex.im
// CHECK-PRIMITIVE: complex.im
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: arith.cmpf
// CHECK-PRIMITIVE: arith.select

// -----

func.func @reduce_dynamic(%arg0: tensor<?x?xi32>, %arg1: tensor<i32>) -> tensor<?xi32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i32>, %arg4 : tensor<i32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x?xi32>, tensor<i32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}
// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK: func @reduce_dynamic(%[[ARG0:.*]]: tensor<?x?xi32>
// CHECK-DAG: %[[INIT:.*]] = tensor.extract %{{.*}} : tensor<i32>
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[DIM1:.*]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?xi32>
// CHECK-DAG: %[[INIT_TENSOR:.*]] = tensor.empty(%[[DIM1]])
// CHECK-DAG: %[[FILL_TENSOR:.*]] = linalg.fill ins(%[[INIT]]{{.*}}outs(%[[INIT_TENSOR]]
// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]], #[[MAP1]]]
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
// CHECK-SAME: ins(%{{.*}}tensor<?x?xi32>)
// CHECK-SAME: outs(%[[FILL_TENSOR]] : tensor<?xi32>)
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = arith.addi %[[RHS_IN]], %[[LHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

func.func @variadic_reduce(%arg0: tensor<9x2xi32>, %arg1: tensor<9x2xi32>) -> (tensor<2xi32>, tensor<2xi32>) {
  %cst0 = mhlo.constant dense<-2147483648> : tensor<i32>
  %cst1 = mhlo.constant dense<0> : tensor<i32>
  %res0, %res1 = "mhlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>, %arg15: tensor<i32>, %arg16: tensor<i32>):
    %669 = "mhlo.compare"(%arg2, %arg15) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %670 = "mhlo.select"(%669, %arg2, %arg15) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %671 = "mhlo.compare"(%arg2, %arg15) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
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
// CHECK-DAG:    %[[CST0:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG:    %[[CST1:.*]] = arith.constant 0 : i32
// CHECK:        %[[INIT0:.*]] = tensor.empty() : tensor<2xi32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[INIT1:.*]] = tensor.empty() : tensor<2xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<9x2xi32>, tensor<9x2xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<2xi32>, tensor<2xi32>)
// CHECK-NEXT:   ^bb0(%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %[[OUT0:.*]]: i32, %[[OUT1:.*]]: i32):
// CHECK-NEXT:     %[[T1:.*]] = arith.cmpi sge, %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T2:.*]] = arith.select %[[T1]], %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T3:.*]] = arith.cmpi eq, %[[OUT0]], %[[IN0]] : i32
// CHECK-NEXT:     %[[T4:.*]] = arith.minsi %[[OUT1:.*]], %[[IN1]] : i32
// CHECK-NEXT:     %[[T5:.*]] = arith.select %[[T1]], %[[OUT1]], %[[IN1]] : i32
// CHECK-NEXT:     %[[T6:.*]] = arith.select %[[T3]], %[[T4]], %[[T5]] : i32
// CHECK-NEXT:     linalg.yield %[[T2]], %[[T6]]

// CHECK-PRIMITIVE:      func @variadic_reduce
// CHECK-PRIMITIVE-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-DAG:    %[[CST0:.*]] = arith.constant -2147483648 : i32
// CHECK-PRIMITIVE-DAG:    %[[CST1:.*]] = arith.constant 0 : i32
// CHECK-PRIMITIVE:        %[[INIT0:.*]] = tensor.empty() : tensor<2xi32>
// CHECK-PRIMITIVE:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK-PRIMITIVE:        %[[INIT1:.*]] = tensor.empty() : tensor<2xi32>
// CHECK-PRIMITIVE:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK-PRIMITIVE:        %[[RES:.+]]:2 = linalg.reduce
// CHECK-PRIMITIVE-NEXT:     ins(%[[ARG0]], %[[ARG1]] : tensor<9x2xi32>, tensor<9x2xi32>)
// CHECK-PRIMITIVE-NEXT:    outs(%[[FILL0]], %[[FILL1]] : tensor<2xi32>, tensor<2xi32>)
// CHECK-PRIMITIVE-NEXT:    dimensions = [0]
// CHECK-PRIMITIVE-NEXT:   (%[[IN0:.*]]: i32, %[[IN1:.*]]: i32, %[[OUT0:.*]]: i32, %[[OUT1:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:     %[[T1:.*]] = arith.cmpi sge, %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T2:.*]] = arith.select %[[T1]], %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T3:.*]] = arith.cmpi eq, %[[OUT0]], %[[IN0]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T4:.*]] = arith.minsi %[[OUT1:.*]], %[[IN1]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T5:.*]] = arith.select %[[T1]], %[[OUT1]], %[[IN1]] : i32
// CHECK-PRIMITIVE-NEXT:     %[[T6:.*]] = arith.select %[[T3]], %[[T4]], %[[T5]] : i32
// CHECK-PRIMITIVE-NEXT:     linalg.yield %[[T2]], %[[T6]]

// -----

func.func @variadic_diff_type_reduce(%arg0: tensor<128x10xf32>, %arg1: tensor<128x10xi32>) -> (tensor<128xf32>, tensor<128xi32>) {
  %cst0 = mhlo.constant dense<1.0> : tensor<f32>
  %cst1 = mhlo.constant dense<1> : tensor<i32>
  %res0, %res1 = "mhlo.reduce"(%arg0, %arg1, %cst0, %cst1) ({
  ^bb0(%arg7: tensor<f32>, %arg8: tensor<i32>, %arg9: tensor<f32>, %arg10: tensor<i32>):
    %0 = "mhlo.compare"(%arg7, %arg9) {comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
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
// CHECK-DAG:        %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:        %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:        %[[INIT0:.*]] = tensor.empty() : tensor<128xf32>
// CHECK:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK:        %[[INIT1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK:        %[[RES:.+]]:2 = linalg.generic {
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP0]], #[[MAP1]], #[[MAP1]]]
// CHECK-SAME:     iterator_types = ["parallel", "reduction"]
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<128x10xf32>, tensor<128x10xi32>)
// CHECK-SAME:    outs(%[[FILL0]], %[[FILL1]] : tensor<128xf32>, tensor<128xi32>)
// CHECK-NEXT:   ^bb0(%[[LHS0:.*]]: f32, %[[LHS1:.*]]: i32, %[[RHS0:.*]]: f32, %[[RHS1:.*]]: i32):
// CHECK-NEXT:      %[[B0:.*]] = arith.cmpf oge, %[[RHS0]], %[[LHS0]] : f32
// CHECK-NEXT:      %[[RES0:.*]] = arith.select %[[B0]], %[[RHS0]], %[[LHS0]] : f32
// CHECK-NEXT:      %[[RES1:.*]] = arith.select %[[B0]], %[[RHS1]], %[[LHS1]] : i32
// CHECK-NEXT:      linalg.yield %[[RES0]], %[[RES1]] : f32, i32

// CHECK-PRIMITIVE:      func @variadic_diff_type_reduce
// CHECK-PRIMITIVE-SAME:   %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-PRIMITIVE-DAG:        %[[CST0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-PRIMITIVE-DAG:        %[[CST1:.*]] = arith.constant 1 : i32
// CHECK-PRIMITIVE:        %[[INIT0:.*]] = tensor.empty() : tensor<128xf32>
// CHECK-PRIMITIVE:        %[[FILL0:.*]] = linalg.fill ins(%[[CST0]]{{.*}}outs(%[[INIT0]]
// CHECK-PRIMITIVE:        %[[INIT1:.*]] = tensor.empty() : tensor<128xi32>
// CHECK-PRIMITIVE:        %[[FILL1:.*]] = linalg.fill ins(%[[CST1]]{{.*}}outs(%[[INIT1]]
// CHECK-PRIMITIVE:        %[[RES:.+]]:2 = linalg.reduce
// CHECK-PRIMITIVE-NEXT:     ins(%[[ARG0]], %[[ARG1]] : tensor<128x10xf32>, tensor<128x10xi32>)
// CHECK-PRIMITIVE-NEXT:     outs(%[[FILL0]], %[[FILL1]] : tensor<128xf32>, tensor<128xi32>)
// CHECK-PRIMITIVE-NEXT:     dimensions = [1]
// CHECK-PRIMITIVE-NEXT:   (%[[LHS0:.*]]: f32, %[[LHS1:.*]]: i32, %[[RHS0:.*]]: f32, %[[RHS1:.*]]: i32) {
// CHECK-PRIMITIVE-NEXT:      %[[B0:.*]] = arith.cmpf oge, %[[RHS0]], %[[LHS0]] : f32
// CHECK-PRIMITIVE-NEXT:      %[[RES0:.*]] = arith.select %[[B0]], %[[RHS0]], %[[LHS0]] : f32
// CHECK-PRIMITIVE-NEXT:      %[[RES1:.*]] = arith.select %[[B0]], %[[RHS1]], %[[LHS1]] : i32
// CHECK-PRIMITIVE-NEXT:      linalg.yield %[[RES0]], %[[RES1]] : f32, i32

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

func.func @slice_with_empty_result(%arg0: tensor<3x3x5xf64>) -> tensor<3x0x5xf64> {
  %0 = "mhlo.slice"(%arg0) {
    limit_indices = dense<[3, 2, 5]> : tensor<3xi64>,
    start_indices = dense<[0, 2, 0]> : tensor<3xi64>,
    strides = dense<[1, 2, 1]> : tensor<3xi64>
  } : (tensor<3x3x5xf64>) -> tensor<3x0x5xf64>
  func.return %0 : tensor<3x0x5xf64>
}
// CHECK-LABEL: func @slice_with_empty_result
//       CHECK:   tensor.extract_slice %{{.*}}[0, 2, 0] [3, 0, 5] [1, 2, 1] : tensor<3x3x5xf64> to tensor<3x0x5xf64>

// -----

func.func @dynamic_slice(%arg: tensor<3x4xf32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xf32> {
  %0 = "mhlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xf32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func @dynamic_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C2]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C0]] : index
// CHECK:           tensor.extract_slice %[[ARG0]][%[[CLAMPED1]], %[[CLAMPED2]]] [1, 4] [1, 1]

// -----

func.func @dynamic_slice_unsigned_index(
    %arg: tensor<3x4xui32>, %start1: tensor<ui64>, %start2: tensor<ui64>)
    -> tensor<1x4xui32> {
  %0 = "mhlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xui32>, tensor<ui64>, tensor<ui64>) -> tensor<1x4xui32>
  func.return %0 : tensor<1x4xui32>
}

// CHECK-LABEL: func @dynamic_slice_unsigned_index(
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract
// CHECK:         arith.index_castui %[[EXTRACT1]]

// -----

func.func @dynamic_slice_unsigned(%arg: tensor<3x4xui32>, %start1: tensor<i64>, %start2: tensor<i64>) -> tensor<1x4xui32> {
  %0 = "mhlo.dynamic_slice"(%arg, %start1, %start2) {
    slice_sizes = dense<[1, 4]> : tensor<2xi64>
  } : (tensor<3x4xui32>, tensor<i64>, tensor<i64>) -> tensor<1x4xui32>
  func.return %0 : tensor<1x4xui32>
}

// CHECK-LABEL: func @dynamic_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[SIGNLESS_ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x4xui32> to tensor<3x4xi32>
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG1]][] : tensor<i64>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C2]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i64>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C0]] : index
// CHECK:           tensor.extract_slice %[[SIGNLESS_ARG0]][%[[CLAMPED1]], %[[CLAMPED2]]] [1, 4] [1, 1]

// -----

func.func @dynamic_update_slice(%target: tensor<3x3xi32>, %update: tensor<2x2xi32>, %c0: tensor<i32>) -> tensor<3x3xi32> {
  %0 = "mhlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<i32>, tensor<i32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}
// CHECK-LABEL: func @dynamic_update_slice(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         return %[[RES]] : tensor<3x3xi32>

// -----

func.func @dynamic_update_slice_unsigned_index(
    %target: tensor<3x3xi32>, %update: tensor<2x2xi32>,
    %idx: tensor<ui32>) -> tensor<3x3xi32> {
  %0 = "mhlo.dynamic_update_slice"(%target, %update, %idx, %idx)
    : (tensor<3x3xi32>, tensor<2x2xi32>, tensor<ui32>, tensor<ui32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// CHECK-LABEL: func @dynamic_update_slice_unsigned_index(
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract
// CHECK:         arith.index_castui %[[EXTRACT1]]

// -----

func.func @dynamic_update_slice_unsigned(%target: tensor<3x3xui32>, %update: tensor<2x2xui32>, %c0: tensor<i32>) -> tensor<3x3xui32> {
  %0 = "mhlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xui32>, tensor<2x2xui32>, tensor<i32>, tensor<i32>) -> tensor<3x3xui32>
  func.return %0 : tensor<3x3xui32>
}
// CHECK-LABEL: func @dynamic_update_slice_unsigned(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[SIGNLESS_UPDATE:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : tensor<2x2xui32> to tensor<2x2xi32>
// CHECK-DAG:     %[[SIGNLESS_TARGET:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : tensor<3x3xui32> to tensor<3x3xi32>
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[SIGNLESS_RES:.*]] = tensor.insert_slice %[[SIGNLESS_UPDATE]] into %[[SIGNLESS_TARGET]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
// CHECK-SAME:    : tensor<2x2xi32> into tensor<3x3xi32>
// CHECK:         %[[RES:.*]] = builtin.unrealized_conversion_cast %[[SIGNLESS_RES]] : tensor<3x3xi32> to tensor<3x3xui32>
// CHECK:         return %[[RES]] : tensor<3x3xui32>

// -----

func.func @dynamic_update_slice_float(%target: tensor<3x3xf32>,
                                 %update: tensor<2x2xf32>,
                                 %c0: tensor<i32>) -> tensor<3x3xf32> {
  %0 = "mhlo.dynamic_update_slice"(%target, %update, %c0, %c0)
    : (tensor<3x3xf32>, tensor<2x2xf32>, tensor<i32>, tensor<i32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @dynamic_update_slice_float(
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG2:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[EXTRACT1:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR1:.*]] = arith.index_cast %[[EXTRACT1]]
// CHECK:         %[[T1:.*]] = arith.maxsi %[[SCALAR1]], %[[C0]] : index
// CHECK:         %[[CLAMPED1:.*]] = arith.minsi %[[T1]], %[[C1]] : index
// CHECK:         %[[EXTRACT2:.*]] = tensor.extract %[[ARG2]][] : tensor<i32>
// CHECK:         %[[SCALAR2:.*]] = arith.index_cast %[[EXTRACT2]]
// CHECK:         %[[T2:.*]] = arith.maxsi %[[SCALAR2]], %[[C0]] : index
// CHECK:         %[[CLAMPED2:.*]] = arith.minsi %[[T2]], %[[C1]] : index
// CHECK:         %[[RES:.*]] = tensor.insert_slice %[[ARG1]] into %[[ARG0]]
// CHECK-SAME:      [%[[CLAMPED1]], %[[CLAMPED2]]] [2, 2] [1, 1]
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
//       CHECK: %[[INIT:.+]] = tensor.empty() : tensor<29x15xi32>
//       CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PAD]] : i32) outs(%[[INIT]] : tensor<29x15xi32>) -> tensor<29x15xi32>
//       CHECK: %[[INSERT:.+]] = tensor.insert_slice %[[CAST0]] into %[[FILL]][4, 5] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x15xi32>

// -----

func.func @pad_interior_negative(%arg0: tensor<12x4xui32>, %arg1: tensor<ui32>) -> tensor<25x9xui32> {
  %0 = arith.constant dense<0> : tensor<ui32>
  %1 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<[-2, 3]> : tensor<2xi64>,
    edge_padding_low = dense<[4, -1]> : tensor<2xi64>,
    interior_padding = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<12x4xui32>, tensor<ui32>) -> tensor<25x9xui32>
  func.return %1 : tensor<25x9xui32>
}
// CHECK-LABEL: func @pad_interior_negative
//       CHECK: %[[PAD:.*]] = tensor.insert_slice %{{.+}} into %{{.+}}[4, 0] [12, 4] [2, 2] : tensor<12x4xi32> into tensor<29x10xi32>
//       CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[PAD]][0, 1] [25, 9] [1, 1] : tensor<29x10xi32> to tensor<25x9xi32>

// -----

func.func @linalg.conv_0d_nc(%arg0: tensor<3x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<3x3xf32> {

  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f]x[i, o]->[b, f], window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []} {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} : (tensor<3x2xf32>, tensor<2x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @linalg.conv_0d_nc
// CHECK-DAG: %[[CST:.+]] = arith.constant 0.000000e+00
// CHECK-DAG: %[[INIT:.+]] = tensor.empty()
// CHECK-DAG: %[[FILL:.+]] = linalg.fill ins(%cst{{.*}}outs(%[[INIT]]
// CHECK: linalg.matmul ins(%arg0, %arg1 : tensor<3x2xf32>, tensor<2x3xf32>) outs(%[[FILL]] : tensor<3x3xf32>)

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
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x?xf32>
// CHECK:         %[[DIM2:.+]] = tensor.dim %[[ARG1]], %[[C2]] : tensor<2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM2]])
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
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x4x5x?xf32>
// CHECK:         %[[DIM3:.+]] = tensor.dim %[[ARG1]], %[[C3]] : tensor<3x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM3]])
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         linalg.conv_2d_nhwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[ARG0]], %[[ARG1]] : tensor<?x4x5x?xf32>, tensor<3x2x?x?xf32>)
// CHECK-SAME:    outs(%[[FILL]] : tensor<?x2x4x?xf32>) -> tensor<?x2x4x?xf32>

// -----

func.func @conv_transpose_2d(%arg0: tensor<2x9x10x3xf32>,
                             %arg1: tensor<4x4x3x3xf32>)
  -> tensor<2x15x25x3xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[6, 6], [6, 6]],
              lhs_dilate = [1, 2], rhs_dilate = [2, 2]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#mhlo<precision DEFAULT>,
                          #mhlo<precision DEFAULT>]
    } : (tensor<2x9x10x3xf32>, tensor<4x4x3x3xf32>) -> tensor<2x15x25x3xf32>
  return %0 : tensor<2x15x25x3xf32>
}

// CHECK-LABEL: func @conv_transpose_2d
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty()
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[INIT]]
// CHECK:         %[[LHS_INIT:.+]] = tensor.empty()
// CHECK:         %[[LHS_FILL:.+]] = linalg.fill ins(%[[ZERO]]{{.*}}outs(%[[LHS_INIT]]
// CHECK:         %[[LHS_PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[LHS_FILL]][0, 6, 6, 0] [2, 9, 10, 3] [1, 1, 2, 1] : tensor<2x9x10x3xf32> into tensor<2x21x31x3xf32>
// CHECK:         linalg.conv_2d_nhwc_hwcf
// CHECK-SAME:      {dilations = dense<2> : tensor<2xi64>
// CHECK-SAME:       strides = dense<1> : tensor<2xi64>}
// CHECK-SAME:     ins(%[[LHS_PAD]], %[[ARG1]] : tensor<2x21x31x3xf32>, tensor<4x4x3x3xf32>)
// CHECK-SAME:     outs(%[[FILL]] : tensor<2x15x25x3xf32>) -> tensor<2x15x25x3xf32>

// -----

func.func @conv_different_batch_dim_in_out(%arg0: tensor<1x1x1xf64>,
                                           %arg1: tensor<1x1x1xf64>)
  -> tensor<1x1x1xf64> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, b]x[i, o, 0]->[f, b, 0],
    window = {stride = [1], pad = [[0, 0]], lhs_dilate = [1],
             rhs_dilate = [1]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]
    } : (tensor<1x1x1xf64>, tensor<1x1x1xf64>) -> tensor<1x1x1xf64>
  return %0 : tensor<1x1x1xf64>
}

// Just check that this lowers successfully.
// CHECK-LABEL: func @conv_different_batch_dim_in_out

// -----

func.func @conv_different_batch_dim_in_out_with_feature_group_count(
    %arg0: tensor<4x6x7x1xf64>, %arg1: tensor<2x6x3x2xf64>)
  -> tensor<1x2x1x2xf64> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, -1]],
              lhs_dilate = [1, 1], rhs_dilate = [1, 2],
              reverse = [0, 0]}
    {
      batch_group_count = 1 : i64,
      feature_group_count = 2 : i64,
      precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]
    } : (tensor<4x6x7x1xf64>, tensor<2x6x3x2xf64>) -> tensor<1x2x1x2xf64>
  return %0 : tensor<1x2x1x2xf64>
}

// Just check that this lowers successfully.
// CHECK-LABEL: func @conv_different_batch_dim_in_out_with_feature_group_count

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
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[DIM0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x8x8x8x?xf32>
// CHECK:         %[[DIM4:.+]] = tensor.dim %[[ARG1]], %[[C4]] : tensor<2x2x2x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[DIM0]], %[[DIM4]])
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
// CHECK-DAG:     %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x2x4x3xf32>
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
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT: %[[INIT:.*]] = tensor.empty() : tensor<400x1024x1024x1xf16>
// CHECK-NEXT: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1024x1024x1xf16>) -> tensor<400x1024x1024x1xf16>
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
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<400x1040x1024x1xf16>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[INIT]] : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
// CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[INPUT]] low[0, 8, 16, 0] high[0, 8, 16, 0]  {
// CHECK-NEXT: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK-NEXT:   tensor.yield %[[ZERO]] : f16
// CHECK-NEXT: } : tensor<400x1024x1024x1xf16> to tensor<400x1040x1056x1xf16>
// CHECK-NEXT: %[[RESULT:.*]] = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%[[PAD]], %arg0 : tensor<400x1040x1056x1xf16>, tensor<1x33x1x1xf16>) outs(%[[FILL]] : tensor<400x1040x1024x1xf16>) -> tensor<400x1040x1024x1xf16>
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
// CHECK-DAG:       %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:       %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]] {{\[}}[0, 1, 2, 3]] : tensor<2x2x1x6xf32> into tensor<24xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]] {{\[}}[0, 1, 2, 3]] : tensor<24xf32> into tensor<2x2x2x3xf32>
// CHECK:       %[[INIT:.+]] = tensor.empty() : tensor<2x3x4x2x3xf32>
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
// CHECK-DAG:    %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 0, 1, 0] high[0, 0, 1, 0] {
// CHECK:        ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:          tensor.yield %[[ZERO]] : f32
// CHECK         } : tensor<2x4x5x2xf32> to tensor<2x4x7x2xf32>
// CHECK:        %[[COLLAPSE:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:    [0, 1, 2, 3]
// CHECK-SAME:    : tensor<2x2x1x4xf32> into tensor<16xf32>
// CHECK:       %[[EXPAND:.+]] = tensor.expand_shape %[[COLLAPSE]]
// CHECK-SAME:   [0, 1, 2, 3]
// CHECK-SAME:   tensor<16xf32> into tensor<2x2x2x2xf32>
// CHECK:        %[[INIT:.+]] = tensor.empty() : tensor<2x3x6x2x2xf32>
// CHECK:        %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<2x3x6x2x2xf32>) -> tensor<2x3x6x2x2xf32>
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
// CHECK-DAG:     %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x56x56x96xf32>
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
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[PAD:.*]] = tensor.pad %[[IN]] low[0, 1, 2, 0] high[0, 1, 2, 0]  {
// CHECK:         ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK          } : tensor<1x113x113x96xf32> to tensor<1x115x117x96xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x57x58x96xf32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[INIT]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>
// CHECK:         %[[RESHAPED_FILTER:.+]] = tensor.collapse_shape %[[FILTER]]
// CHECK-SAME:     [0], [1], [2, 3]
// CHECK-SAME:     : tensor<3x3x1x96xf32> into tensor<3x3x96xf32>
// CHECK:         %{{.+}} = linalg.depthwise_conv_2d_nhwc_hwc
// CHECK-SAME:      {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
// CHECK-SAME:       ins(%[[PAD]], %[[RESHAPED_FILTER]] : tensor<1x115x117x96xf32>, tensor<3x3x96xf32>)
// CHECK-SAME:       outs(%[[FILL]] : tensor<1x57x58x96xf32>) -> tensor<1x57x58x96xf32>

// -----

func.func @depthwise_conv1d(%arg0: tensor<1x10x8xf32>,
                            %arg1: tensor<3x1x16xf32>) -> tensor<1x10x16xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [1],
      pad = [[1, 1]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 8 : i64,
    someattr} : (tensor<1x10x8xf32>, tensor<3x1x16xf32>) -> tensor<1x10x16xf32>
  func.return %0 : tensor<1x10x16xf32>
}

// CHECK:      func @depthwise_conv1d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]

// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_1d_nwc_wcm
// CHECK:       %[[OUT:.+]] = tensor.collapse_shape %[[CONV]]
// CHECK:       return %[[OUT]]

// -----

func.func @depthwise_conv1d_m1(%arg0: tensor<1x10x8xf32>,
                               %arg1: tensor<3x1x8xf32>) -> tensor<1x10x8xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f],
    window = {
      stride = [1],
      pad = [[1, 1]],
      lhs_dilate = [1],
      rhs_dilate = [1],
      reverse = [0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 8 : i64,
    someattr} : (tensor<1x10x8xf32>, tensor<3x1x8xf32>) -> tensor<1x10x8xf32>
  func.return %0 : tensor<1x10x8xf32>
}

// CHECK:      func @depthwise_conv1d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]

// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_1d_nwc_wc
// CHECK:       return %[[CONV]]

// -----

func.func @depthwise_conv3d(%arg0: tensor<2x3x5x4x6xf32>,
                            %arg1: tensor<2x1x3x1x36xf32>)
                            -> tensor<2x3x13x4x36xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [2, 1, 3],
      pad = [[1, 2], [5, 3], [3, 5]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1],
      reverse = [0, 0, 0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 6 : i64,
    someattr} : (tensor<2x3x5x4x6xf32>, tensor<2x1x3x1x36xf32>)
              -> tensor<2x3x13x4x36xf32>
  func.return %0 : tensor<2x3x13x4x36xf32>
}

// CHECK:      func @depthwise_conv3d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]

// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_3d_ndhwc_dhwcm
// CHECK:       %[[OUT:.+]] = tensor.collapse_shape %[[CONV]]
// CHECK:       return %[[OUT]]

// -----

func.func @depthwise_conv3d_m1(%arg0: tensor<2x3x5x4x6xf32>,
                               %arg1: tensor<2x1x3x1x6xf32>)
                               -> tensor<2x3x13x4x6xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, 2, f]x[0, 1, 2, i, o]->[b, 0, 1, 2, f],
    window = {
      stride = [2, 1, 3],
      pad = [[1, 2], [5, 3], [3, 5]],
      lhs_dilate = [1, 1, 1],
      rhs_dilate = [1, 1, 1],
      reverse = [0, 0, 0]} {
    batch_group_count = 1 : i64,
    feature_group_count = 6 : i64,
    someattr} : (tensor<2x3x5x4x6xf32>, tensor<2x1x3x1x6xf32>)
              -> tensor<2x3x13x4x6xf32>
  func.return %0 : tensor<2x3x13x4x6xf32>
}

// CHECK:      func @depthwise_conv3d
// CHECK-SAME:   %[[IN:[a-zA-Z0-9_]*]]
// CHECK-SAME:   %[[FILTER:[a-zA-Z0-9_]*]]

// CHECK:       %[[CONV:.+]] = linalg.depthwise_conv_3d_ndhwc_dhwc
// CHECK:       return %[[CONV]]

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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
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

// CHECK-LABEL: func @reduce_window_max_nhwc_with_cst
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-DAG:     %[[CST:.+]] = arith.constant 0xFF800000
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x64xf32
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[CST]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
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
// CHECK:         %[[WINDOW0:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT0:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL0:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL0:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES0:.+]] = linalg.pooling_nhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW0]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL0]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[WINDOW1:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[INIT1:.+]] = tensor.empty() : tensor<1x8x8x64xf32>
// CHECK:         %[[INIT_VAL1:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL1:.+]] = linalg.fill ins(%[[INIT_VAL1]] : f32) outs(%[[INIT1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         %[[RES1:.+]] = linalg.pooling_nhwc_max
// CHECK-SAME:      {dilations = dense<1> : vector<2xi64>
// CHECK-SAME:       strides = dense<2> : vector<2xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW1]] : tensor<1x17x17x64xf32>, tensor<3x3xf32>)
// CHECK-SAME:      outs(%[[FILL1]] : tensor<1x8x8x64xf32>) -> tensor<1x8x8x64xf32>
// CHECK:         return %[[RES0]], %[[RES1]]

// -----

func.func @reduce_window_unsigned(%arg0: tensor<1x1xui32>) -> tensor<1x1xui32> {
  %0 = mhlo.constant dense<0> : tensor<ui32>
  %1 = "mhlo.reduce_window"(%arg0, %0) ({
  ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
    mhlo.return %arg1 : tensor<ui32>
  }) {
    window_dimensions = dense<[1, 1]> : tensor<2xi64>,
    window_strides = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<1x1xui32>, tensor<ui32>) -> tensor<1x1xui32>
  return %1 : tensor<1x1xui32>
}

// Just check that this lowers successfully.
// CHECK-LABEL: func @reduce_window_unsigned

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
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3xf32>
// CHECK:         %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C1]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[D1:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[T1:.+]] = tensor.dim %[[ARG0]], %[[C2]] : tensor<?x?x?x?xf32>
// CHECK:         %[[T2:.+]] = arith.subi %[[T1]], %[[C3]]
// CHECK:         %[[T3:.+]] = arith.divui %[[T2]], %[[C2]]
// CHECK:         %[[D2:.+]] = arith.addi %[[T3]], %[[C1]]
// CHECK:         %[[D3:.+]] = tensor.dim %[[ARG0]], %[[C3]] : tensor<?x?x?x?xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]], %[[D3]]) : tensor<?x?x?x?xf32>
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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
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
// CHECK:         %[[WINDOW:.+]] = tensor.empty() : tensor<3x3x3xf32>
// CHECK:         %[[INIT:.+]] = tensor.empty() : tensor<1x8x8x8x64xf32>
// CHECK:         %[[INIT_VAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK:         %[[FILL:.+]] = linalg.fill ins(%[[INIT_VAL]] : f32) outs(%[[INIT]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>
// CHECK:         %[[RES:.+]] = linalg.pooling_ndhwc_sum
// CHECK-SAME:      {dilations = dense<1> : vector<3xi64>
// CHECK-SAME:       strides = dense<2> : vector<3xi64>}
// CHECK-SAME:      ins(%[[ARG0]], %[[WINDOW]] : tensor<1x17x17x17x64xf32>, tensor<3x3x3xf32>)
// CHECK-SAME:      outs(%[[FILL]] : tensor<1x8x8x8x64xf32>) -> tensor<1x8x8x8x64xf32>

// -----

// CHECK-LABEL: func @reduce_window_sum_ndhwc_dilated_base
// CHECK: linalg.generic
func.func @reduce_window_sum_ndhwc_dilated_base(
    %arg0: tensor<1x17x17x17x64xf32>,
    %arg1: tensor<f32>) -> tensor<1x8x8x16x64xf32>{
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3 : tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = dense<[1, 1, 1, 2, 1]> : tensor<5xi64>,
      window_dimensions = dense<[1, 3, 3, 3, 1]> : tensor<5xi64>,
      window_strides = dense<[1, 2, 2, 2, 1]> : tensor<5xi64>} : (tensor<1x17x17x17x64xf32>, tensor<f32>) -> tensor<1x8x8x16x64xf32>
  func.return %0 : tensor<1x8x8x16x64xf32>
}

// -----

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1) -> ()>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d0 * 2, d1 + d2 * 2)>
// CHECK: #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK: #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func @reduce_window_generic
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]

// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x7xf32>
// CHECK: %[[FILL:.+]] = linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<f32>) outs(%[[INIT]] : tensor<4x7xf32>)
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
// CHECK:   linalg.yield %[[IN]] : f32

// CHECK: %[[PADVAL:.+]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[PAD:.+]] = tensor.pad %arg0 low[0, 1] high[3, 2]
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %{{[a-zA-Z0-9_]*}}: index
// CHECK-SAME: %{{[a-zA-Z0-9_]*}}: index
// CHECK:   tensor.yield %[[PADVAL]] : f32

// CHECK: %[[WINDOW:.+]] = tensor.empty() : tensor<2xf32>
// CHECK: %[[REDUCE:.+]] = linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[PAD]], %[[WINDOW]] : tensor<7x9xf32>, tensor<2xf32>) outs(%[[FILL]] : tensor<4x7xf32>) {
// CHECK: ^{{[a-z0-9_]*}}
// CHECK-SAME: %[[IN:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[IN2:[a-zA-Z0-9_]*]]: f32
// CHECK-SAME: %[[OUT:[a-zA-Z0-9_]*]]: f32
// CHECK:   %[[ADD:.+]] = arith.addf %[[OUT]], %[[IN]] : f32
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

// CHECK-LABEL: func @reduce_window_generic_captured_constant
func.func @reduce_window_generic_captured_constant(%arg0: tensor<4x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %c2 = mhlo.constant dense<2.0> : tensor<f32>
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    %2 = mhlo.multiply %1, %c2 : tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  }) {base_dilations = dense<1> : tensor<2xi64>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}

// CHECK: %[[C2:.*]] = arith.constant 2.0
// CHECK: linalg.generic
// CHECK: %[[SUM:.*]] = arith.addf
// CHECK: %[[PROD:.*]] = arith.mulf %[[SUM]], %[[C2]]
// CHECK: linalg.yield %[[PROD]]

// -----

// CHECK-LABEL: func @reduce_window_generic_padding
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]

// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[PAD:.+]] = tensor.pad %[[ARG0]] low[0, 1] high[3, 2]
// CHECK: tensor.yield %[[PADVAL]] : f32

func.func @reduce_window_generic_padding(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<3x7xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<3x7xf32>
  func.return %0 : tensor<3x7xf32>
}

// -----

// CHECK-LABEL: func @reduce_window_generic_base_dilation
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]

// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<5x6xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PADVAL]] : f32) outs(%[[INIT]] : tensor<5x6xf32>) -> tensor<5x6xf32>
// CHECK: %[[PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 0] [3, 6] [2, 1] : tensor<3x6xf32> into tensor<5x6xf32>

func.func @reduce_window_generic_base_dilation(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<3x4xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = dense<[2, 1]> : tensor<2xi64>, window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: func @reduce_window_generic_padding_base_dilation
// CHECK-SAME:    %[[ARG0:[a-zA-Z0-9_]*]]
// CHECK-SAME:    %[[ARG1:[a-zA-Z0-9_]*]]

// CHECK: %[[PADVAL:.+]] = tensor.extract %[[ARG1]][] : tensor<f32>
// CHECK: %[[INIT:.+]] = tensor.empty() : tensor<8x9xf32>
// CHECK: %[[FILL:.+]] = linalg.fill ins(%[[PADVAL]] : f32) outs(%[[INIT]] : tensor<8x9xf32>) -> tensor<8x9xf32>
// CHECK: %[[PAD:.+]] = tensor.insert_slice %[[ARG0]] into %[[FILL]][0, 1] [3, 6] [2, 1] : tensor<3x6xf32> into tensor<8x9xf32>

func.func @reduce_window_generic_padding_base_dilation(%arg0: tensor<3x6xf32>, %arg1: tensor<f32>) -> tensor<4x7xf32> {
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = dense<[2, 1]> : tensor<2xi64>, padding = dense<[[0, 3], [1, 2]]> : tensor<2x2xi64>, window_dilations = dense<[1, 2]> : tensor<2xi64>, window_dimensions = dense<[1, 2]> : tensor<2xi64>, window_strides = dense<[2, 1]> : tensor<2xi64>} : (tensor<3x6xf32>, tensor<f32>) -> tensor<4x7xf32>
  func.return %0 : tensor<4x7xf32>
}

// -----

func.func @reduce_window_generic_scalar(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {base_dilations = dense<> : tensor<0xi64>, padding = dense<> : tensor<0x2xi64>, window_dilations = dense<> : tensor<0xi64>, window_dimensions = dense<> : tensor<0xi64>, window_strides = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK: #[[MAP:.+]] = affine_map<() -> ()>
// CHECK-LABEL: func @reduce_window_generic_scalar
// CHECK: linalg.generic {indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]

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

// CHECK: #[[MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL:   func @gather(
// CHECK-SAME:        %[[OPERAND:[a-zA-Z0-9_]+]]
// CHECK-SAME:        %[[START_INDICES:[a-zA-Z0-9_]+]]
// CHECK-SAME:    )
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<1x8x8xi32>
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
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]]  : index
// CHECK-DAG:         %[[IN0:.+]] = arith.minsi %[[CLAMP0]], %[[C0]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.minsi %[[CLAMP1]], %[[C3]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IDX2]]] : tensor<1x4x8xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK-DAG:       return %[[RES]]

// -----

func.func @gather_unsigned_index(
    %operand : tensor<1x4x8xi32>, %start_indices : tensor<1x8x2xui32>)
    -> tensor<1x8x8xi32> {
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
  } : (tensor<1x4x8xi32>, tensor<1x8x2xui32>) -> tensor<1x8x8xi32>
  func.return %res : tensor<1x8x8xi32>
}

// CHECK-LABEL:   func @gather_unsigned_index(
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1
// CHECK:           %[[S0_INT:.+]] = tensor.extract {{.*}}[{{.*}}, %[[C0]]]
// CHECK:           arith.index_castui %[[S0_INT]] : i32 to index
// CHECK:           %[[S0_INT:.+]] = tensor.extract {{.*}}[{{.*}}, %[[C1]]]
// CHECK:           arith.index_castui %[[S1_INT]] : i32 to index

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
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<5x4x2xi32>
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
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[C2]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX1]] : index
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[C1]]
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX2]] : index
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
// CHECK-DAG:       %[[C2:.+]] = arith.constant 2
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<2x3x4x5xi32>
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
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C2]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[D1:.+]] = tensor.dim %[[OPERAND]], %[[C1]]
// CHECK-DAG:         %[[L1:.+]] = arith.subi %[[D1]], %[[C3]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[L1]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX1]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IN1]], %[[IDX2]]] : tensor<?x?x?xi32>
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
// CHECK-DAG:       %[[C5:.+]] = arith.constant 5
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<5x2x4xi32>
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
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP1:.+]] = arith.maxsi %[[S1]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP1_1:.+]] = arith.minsi %[[CLAMP1]], %[[C1]]
// CHECK-DAG:         %[[CLAMP2:.+]] = arith.maxsi %[[S2]], %[[C0]]  : index
// CHECK-DAG:         %[[IN2:.+]] = arith.minsi %[[CLAMP2]], %[[C1]]
// CHECK-DAG:         %[[CLAMP3:.+]] = arith.maxsi %[[S3]], %[[C0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.minsi %[[CLAMP3]], %[[C5]]
// CHECK-DAG:         %[[IN1:.+]] = arith.addi %[[CLAMP1_1]], %[[IDX1]] : index
// CHECK-DAG:         %[[IN3:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX2]] : index
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
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty() : tensor<3x4x5x2xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x5x2xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[IDX3:.+]] = linalg.index 3
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[IDX3]]] : tensor<5x2xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
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
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[DYN_DIM:.+]] = tensor.dim %[[START_INDICES]], %[[C0]]
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty(%[[DYN_DIM]]) : tensor<3x4x?xi32>
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor<3x4x?xi32>) {
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
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
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3
// CHECK-DAG:       %[[RES_DIM2:.+]] = tensor.dim %[[START_INDICES]], %[[C0]]
// CHECK-DAG:       %[[INIT:.+]] = tensor.empty(%[[RES_DIM2]])
// CHECK:           %[[RES:.+]] = linalg.generic
// CHECK-SAME:           outs(%[[INIT]] : tensor
// CHECK:           ^bb0
// CHECK-DAG:         %[[IDX0:.+]] = linalg.index 0
// CHECK-DAG:         %[[IDX1:.+]] = linalg.index 1
// CHECK-DAG:         %[[IDX2:.+]] = linalg.index 2
// CHECK-DAG:         %[[S0_INT:.+]] = tensor.extract %[[START_INDICES]][%[[IDX2]], %[[C0]]] : tensor<?x?xi32>
// CHECK-DAG:         %[[S0:.+]] = arith.index_cast %[[S0_INT]] : i32 to index
// CHECK-DAG:         %[[D0:.+]] = tensor.dim %[[OPERAND]], %[[C0]]
// CHECK-DAG:         %[[L0:.+]] = arith.subi %[[D0]], %[[C3]]
// CHECK-DAG:         %[[CLAMP0:.+]] = arith.maxsi %[[S0]], %[[C0]] : index
// CHECK-DAG:         %[[CLAMP0_1:.+]] = arith.minsi %[[CLAMP0]], %[[L0]] : index
// CHECK-DAG:         %[[IN0:.+]] = arith.addi %[[CLAMP0_1]], %[[IDX0]] : index
// CHECK-DAG:         %[[OPERAND_CASTED:.+]] = tensor.cast %[[OPERAND]] : tensor<*xi32> to tensor<?x?xi32>
// CHECK:             %[[Y:.+]] = tensor.extract %[[OPERAND_CASTED]][%[[IN0]], %[[IDX1]]] : tensor<?x?xi32>
// CHECK:             linalg.yield %[[Y]] : i32
// CHECK:           %[[CAST:.+]] = tensor.cast %[[RES]]
// CHECK:           return %[[CAST]]

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
//      CHECK: %[[INIT1:.+]] = tensor.empty() :
//      CHECK: %[[INIT2:.+]] = tensor.empty() :
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[INIT1]] :
// CHECK-SAME: outs(%[[INIT2]] :
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
  %0 = "mhlo.rng"(%min, %max, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}
// CHECK-LABEL: func @rng_uniform_1d
// CHECK-DAG:  ^{{.+}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
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
        %0 = "mhlo.rng"(%min, %max, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<2xi32>) -> tensor<3x3xf32>
        func.return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @rng_uniform_2d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
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
        %0 = "mhlo.rng"(%min, %max, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<3xi32>) -> tensor<2x2x2xf32>
        func.return %0 : tensor<2x2x2xf32>
}
// CHECK-LABEL: func @rng_uniform_3d
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:  %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:  %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:  %[[IDX1:.+]] = linalg.index 1 : index
// CHECK-DAG:  %[[IDX1_CAST:.+]] = arith.index_cast %[[IDX1]] : index to i32
// CHECK-DAG:  %[[IDX2:.+]] = linalg.index 2 : index
// CHECK-DAG:  %[[IDX2_CAST:.+]] = arith.index_cast %[[IDX2]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
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
  %0 = "mhlo.rng"(%min, %max, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>} : (tensor<f32>, tensor<f32>, tensor<1xi32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: func @rng_uniform_dynamic_1d
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[EX:.+]] = tensor.extract %{{.*}}[%[[C0]]]
// CHECK-DAG:    %[[IND:.+]] = arith.index_cast %[[EX]] : i32 to index
// CHECK-DAG:    %{{.+}} = tensor.empty(%[[IND]]) : tensor<?xf32>
// CHECK-DAG:  ^{{.*}}(%[[MIN:.+]]: f32, %[[MAX:.+]]: f32, %[[OUT:.+]]: f32
// CHECK-DAG:    %[[CST0:.+]] = arith.constant 1103515245 : i32
// CHECK-DAG:    %[[CST1:.+]] = arith.constant 12345 : i32
// CHECK-DAG:    %[[CST2:.+]] = arith.constant 2.32830644E-10 : f32
// CHECK-DAG:    %[[IDX0:.+]] = linalg.index 0 : index
// CHECK-DAG:    %[[IDX0_CAST:.+]] = arith.index_cast %[[IDX0]] : index to i32
// CHECK-DAG:    %[[VAL1:.+]] = arith.muli %[[IDX0_CAST]], %[[CST0]] : i32
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
//      CHECK:   %[[INIT:.*]] = tensor.empty() : tensor<1x5xi32>
//      CHECK:   %[[RES:.+]] = linalg.generic {
// CHECK-SAME:     indexing_maps
// CHECK-SAME:     #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[INIT]] : tensor<2xi32>, tensor<1x5xi32>)
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
//      CHECK: %[[T0:.+]] = tensor.empty() : tensor<8xf32>
//      CHECK: %[[T1:.+]] = tensor.empty() : tensor<8xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP1]]
// CHECK-SAME:   iterator_types = ["parallel"]
// CHECK-SAME:   ins(%[[INDEX]], %[[T0]] : tensor<i32>, tensor<8xf32>) outs(%[[T1]] : tensor<8xf32>)
//      CHECK:   ^{{.+}}(%[[VAL:[a-zA-Z0-9_]+]]: i32, %{{.+}}: f32):
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
//      CHECK: %[[INIT:.+]] = tensor.empty() : tensor<4x7x2xf32>
//      CHECK: linalg.generic {
// CHECK-SAME:   indexing_maps
// CHECK-SAME:   #[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:   iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME: ins(%[[INDEX]], %[[INIT]] :
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
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   %[[D0:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[D1:.+]] = tensor.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[D2:.+]] = tensor.dim %[[INDEX]], %[[C1]]
//      CHECK:   %[[D3:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[D4:.+]] = tensor.dim %[[INPUT]], %[[C0]]
//      CHECK:   %[[D5:.+]] = tensor.dim %[[INPUT]], %[[C1]]
//      CHECK:   %[[D6:.+]] = tensor.dim %[[INPUT]], %[[C3]]
//      CHECK:   %[[INIT0:.+]] = tensor.empty(%[[D4]], %[[D5]], %[[D6]]) : tensor<?x?x?xf32>
//      CHECK:   %[[INIT1:.+]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]], %[[D3]])
//      CHECK:   %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:     indexing_maps = [#[[MAP0]], #[[MAP1]], #[[MAP2]]
// CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK-SAME:     ins(%[[INDEX]], %[[INIT0]] : tensor<?x?xi32>, tensor<?x?x?xf32>)
// CHECK-SAME:     outs(%[[INIT1]] : tensor<?x?x?x?xf32>)
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_9:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_15:.*]] = tensor.dim %[[VAL_2]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_15]] : index
// CHECK:           %[[VAL_23:.*]] = tensor.empty(%[[VAL_5]], %[[VAL_17]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_23]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_25:.*]]: i32):
// CHECK:             %[[VAL_26:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_28:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_30:.*]] = tensor.dim %[[VAL_0]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_32:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_30]] : index
// CHECK:             %[[VAL_33:.*]] = scf.if %[[VAL_32]] -> (i32) {
// CHECK:               %[[VAL_35:.*]] = tensor.extract %[[VAL_0]][%[[VAL_26]], %[[VAL_28]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_35]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_37:.*]] = tensor.dim %[[VAL_1]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_30]], %[[VAL_37]] : index
// CHECK:               %[[VAL_39:.*]] = arith.cmpi ult, %[[VAL_28]], %[[VAL_38]] : index
// CHECK:               %[[VAL_40:.*]] = scf.if %[[VAL_39]] -> (i32) {
// CHECK:                 %[[VAL_41:.*]] = arith.subi %[[VAL_28]], %[[VAL_30]] : index
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
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_4:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK-DAG:       %[[VAL_3:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : tensor<?x?xui32> to tensor<?x?xi32>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_3]], %[[C0]] : tensor<?x?xi32>
// CHECK:           %[[VAL_10:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_14:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_18:.*]] = tensor.dim %[[VAL_5]], %[[C1]] : tensor<?x?xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_10]], %[[VAL_14]] : index
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : index
// CHECK:           %[[VAL_26:.*]] = tensor.empty(%[[VAL_8]], %[[VAL_20]]) : tensor<?x?xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%[[VAL_26]] : tensor<?x?xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = linalg.index 1 : index
// CHECK:             %[[VAL_33:.*]] = tensor.dim %[[VAL_3]], %[[C1]] : tensor<?x?xi32>
// CHECK:             %[[VAL_35:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_33]] : index
// CHECK:             %[[VAL_36:.*]] = scf.if %[[VAL_35]] -> (i32) {
// CHECK:               %[[VAL_38:.*]] = tensor.extract %[[VAL_3]][%[[VAL_29]], %[[VAL_30]]] : tensor<?x?xi32>
// CHECK:               scf.yield %[[VAL_38]] : i32
// CHECK:             } else {
// CHECK:               %[[VAL_40:.*]] = tensor.dim %[[VAL_4]], %[[C1]] : tensor<?x?xi32>
// CHECK:               %[[VAL_41:.*]] = arith.addi %[[VAL_33]], %[[VAL_40]] : index
// CHECK:               %[[VAL_42:.*]] = arith.cmpi ult, %[[VAL_30]], %[[VAL_41]] : index
// CHECK:               %[[VAL_43:.*]] = scf.if %[[VAL_42]] -> (i32) {
// CHECK:                 %[[VAL_44:.*]] = arith.subi %[[VAL_30]], %[[VAL_33]] : index
// CHECK:                 %[[VAL_45:.*]] = tensor.extract %[[VAL_4]][%[[VAL_29]], %[[VAL_44]]] : tensor<?x?xi32>
// CHECK:                 scf.yield %[[VAL_45]] : i32
// CHECK:               } else {
// CHECK:                 %[[VAL_46:.*]] = arith.subi %[[VAL_30]], %[[VAL_41]] : index
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
  // CHECK-DAG:   %[[VAL_7:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_8:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_10:.*]] = arith.constant 1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
  // CHECK:   %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_9]] : i32
  // CHECK:   %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_8]] : i32
  // CHECK:   %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_7]] : i32
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
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant -1 : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_12:.*]] = arith.constant 1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
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
  // CHECK-DAG:   %[[VAL_7:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[VAL_9:.*]] = arith.constant 1 : i32
  // CHECK-DAG:   %[[VAL_11:.*]] = arith.constant -2147483648 : i32
  // CHECK-DAG:   %[[VAL_13:.*]] = arith.constant -1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
  // CHECK:   %[[VAL_10:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_7]] : i32
  // CHECK:   %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_11]] : i32
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
  // CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : i32
  // CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : i32
  // CHECK: linalg.generic
  // CHECK: ^bb0(%[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32):
  // CHECK:   %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_7]], %[[C0]] : i32
  // CHECK:   %[[VAL_13:.*]] = arith.select %[[VAL_12]], %[[C1]], %[[VAL_7]] : i32
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
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2x2xui32>, tensor<2x2xui32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}

// -----

// CHECK-LABEL: pred_compare
func.func @pred_compare(%lhs: tensor<2x2xi1>, %rhs: tensor<2x2xi1>) -> tensor<2x2xi1> {
  // CHECK: linalg.generic
  // CHECK: cmpi ugt
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<2x2xi1>, tensor<2x2xi1>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}

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
// CHECK-SAME: (%[[ARG0:.*]]:
func.func @real_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.real"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: return %[[ARG0]]
  func.return %1 : tensor<?xf32>
}

// -----

// CHECK-LABEL: @imag_real
func.func @imag_real(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  %1 = "mhlo.imag"(%arg0) : (tensor<?xf32>) -> (tensor<?xf32>)
  // CHECK: %[[CST:.*]] = arith.constant 0
  // CHECK: linalg.generic
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
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
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[D0:.*]] = tensor.dim %[[ARG0]], %[[C1]]
// CHECK: %[[D1:.*]] = tensor.dim %[[ARG0]], %[[C0]]
// CHECK: %[[D2:.*]] = tensor.dim %[[ARG1]], %[[C0]]
// CHECK: %[[INIT:.*]] = tensor.empty(%[[D0]], %[[D1]], %[[D2]])
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
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
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
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

// -----

// CHECK-LABEL: func @reduce_precision(
// CHECK-DAG: %[[C2:.*]] = arith.constant 1048576 : i32
// CHECK-DAG: %[[C_21:.*]] = arith.constant 20 : i32
// CHECK-DAG: %[[C3:.*]] = arith.constant 524287 : i32
// CHECK-DAG: %[[C4:.*]] = arith.constant -1048576 : i32
// CHECK-DAG: %[[C5:.*]] = arith.constant 2139095040 : i32
// CHECK-DAG: %[[C6:.*]] = arith.constant 1090519040 : i32
// CHECK-DAG: %[[C7:.*]] = arith.constant 1040187392 : i32
// CHECK-DAG: %[[C8:.*]] = arith.constant -2147483648 : i32
// CHECK-DAG: %[[C9:.*]] = arith.constant 2147483647 : i32
// CHECK: linalg.generic
// CHECK: %[[X_AS_INT:.*]] = arith.bitcast %[[IN:.*]] : f32 to i32
// CHECK: %[[ABS_X:.*]] = arith.andi %[[X_AS_INT]], %[[C9]]
// CHECK: %[[IS_NAN:.*]] = arith.cmpi ugt, %[[ABS_X]], %[[C5]]
// CHECK: %[[MASKED:.*]] = arith.andi %[[X_AS_INT]], %[[C2]] : i32
// CHECK: %[[V0:.*]] = arith.shrui %[[MASKED]], %[[C_21]] : i32
// CHECK: %[[V1:.*]] = arith.addi %[[V0]], %[[C3]] : i32
// CHECK: %[[V2:.*]] = arith.addi %[[X_AS_INT]], %[[V1]] : i32
// CHECK: %[[V3:.*]] = arith.andi %[[V2]], %[[C4]] : i32
// CHECK: %[[V4:.*]] = arith.andi %[[V3]], %[[C5]] : i32
// CHECK: %[[V5:.*]] = arith.cmpi ugt, %[[V4]], %[[C6]] : i32
// CHECK: %[[V6:.*]] = arith.cmpi ule, %[[V4]], %[[C7]] : i32
// CHECK: %[[V7:.*]] = arith.andi %[[V3]], %[[C8]] : i32
// CHECK: %[[V8:.*]] = arith.ori %[[V7]], %[[C5]] : i32
// CHECK: %[[V9:.*]] = arith.select %[[V5]], %[[V8]], %[[V3]] : i32
// CHECK: %[[V10:.*]] = arith.select %[[V6]], %[[V7]], %[[V9]] : i32
// CHECK: %[[CONVERTED:.*]] = arith.bitcast %[[V10]] : i32 to f32
// CHECK: %[[RESULT:.*]] = arith.select %[[IS_NAN]], %[[IN]], %[[CONVERTED]]
// CHECK: linalg.yield %[[RESULT]]

// CHECK-PRIMITIVE-LABEL: func @reduce_precision(
// CHECK-PRIMITIVE: linalg.map
func.func @reduce_precision(%arg0: tensor<1x2x3x4xf32>)
                            -> tensor<1x2x3x4xf32> {
  %0 = "mhlo.reduce_precision"(%arg0) {exponent_bits=3:i32, mantissa_bits=3:i32} : (tensor<1x2x3x4xf32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>
}

// -----

// The following pattern only tests the general structure of the code and the
// affine maps as it is better tested by tests executing the result, and as it
// includes many ops which could lead to a high load of refactoring.
//
// This op has non-default values for everything except feature_group_count to
// ensure the correct default pattern is tested instead of any specializations.


// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (-d0 + 12, -d1 + 12, d2, d3)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d7, d5 * 2 + d6, d3 * 2 + d4, d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d6, d1, d0, d2)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d3, d5, d0, d2)>

// CHECK-LABEL: @batch_group_count_convolution
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x14x12x1xf64>
// CHECK-SAME: %[[ARG1:.*]]: tensor<7x7x1x2xf64>)
// CHECK-SAME: -> tensor<1x6x8x2xf64>

  // Check for padding and dilation
  // CHECK-DAG: %[[PADDED_LHS:.*]] = tensor.insert_slice %[[ARG0]] into %{{.*}}[0, 0, 1, 0] [2, 14, 12, 1] [1, 2, 2, 1] : tensor<2x14x12x1xf64> into tensor<2x28x24x1xf64>

  // Check for padding and dilation
  // CHECK-DAG: %[[PADDED_RHS:.*]] = tensor.insert_slice %[[ARG1]] into %{{.*}}[0, 0, 0, 0] [7, 7, 1, 2] [2, 2, 1, 1] : tensor<7x7x1x2xf64> into tensor<13x13x1x2xf64>

  // Check for reversing window
  // CHECK: linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[PADDED_RHS]] : tensor<13x13x1x2xf64>) outs(%{{.*}} : tensor<13x13x1x2xf64>) {

  // Check Convolution
  // CHECK: linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]],
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]}
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x1x28x24x1xf64>, tensor<13x13x1x2x1xf64>)
  // CHECK-SAME: outs(%{{.*}} : tensor<1x6x8x2x1xf64>) {
  // CHECK: ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64, %[[OUT:.*]]: f64):
    // CHECK: %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f64
    // CHECK: %[[RES:.*]] = arith.addf %[[OUT]], %[[MUL]] : f64
    // CHECK: linalg.yield %[[RES]] : f64
  // CHECK: } -> tensor<1x6x8x2x1xf64>
  // Check proper shape is returned
  // CHECK: return {{.*}} : tensor<1x6x8x2xf64>

func.func @batch_group_count_convolution(%arg0: tensor<2x14x12x1xf64>, %arg1: tensor<7x7x1x2xf64>)
                            -> tensor<1x6x8x2xf64> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[1, 0], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [2, 2], reverse = [1, 1]}
    {batch_group_count = 2 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]}
    : (tensor<2x14x12x1xf64>, tensor<7x7x1x2xf64>) -> tensor<1x6x8x2xf64>
  return %0 : tensor<1x6x8x2xf64>
}

// -----
// The following pattern only tests the general structure of the code and the
// affine maps as it is better tested by tests executing the result, and as it
// includes many ops which could lead to a high load of refactoring.
//
// This op has non-default values for everything except batch_group_count to
// ensure the correct default pattern is tested instead of any specializations.

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (-d0 + 12, -d1 + 12, d2, d3)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d5 * 2 + d6, d3 * 2 + d4, d0, d1)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d6, d1, d0, d2)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d7, d3, d5, d0, d2)>

// CHECK-LABEL: @feature_group_count_convolution
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x14x12x2xf64>
// CHECK-SAME: %[[ARG1:.*]]: tensor<7x7x1x2xf64>)
// CHECK-SAME: -> tensor<2x6x8x2xf64>

  // Check for padding and dilation
  // CHECK-DAG: %[[PADDED_LHS:.*]] = tensor.insert_slice %[[ARG0]] into %{{.*}}[0, 0, 1, 0] [2, 14, 12, 2] [1, 2, 2, 1] : tensor<2x14x12x2xf64> into tensor<2x28x24x2xf64>

  // Check for padding and dilation
  // CHECK-DAG: %[[PADDED_RHS:.*]] = tensor.insert_slice %[[ARG1]] into %{{.*}}[0, 0, 0, 0] [7, 7, 1, 2] [2, 2, 1, 1] : tensor<7x7x1x2xf64> into tensor<13x13x1x2xf64>

  // Check for reversing window
  // CHECK: linalg.generic {indexing_maps = [#[[MAP0]], #[[MAP1]]],
  // CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
  // CHECK-SAME: ins(%[[PADDED_RHS]] : tensor<13x13x1x2xf64>)
  // CHECK-SAME: outs(%{{.*}} : tensor<13x13x1x2xf64>) {

  // Check Convolution
  // CHECK: linalg.generic {indexing_maps = [#[[MAP2]], #[[MAP3]], #[[MAP4]]],
  // CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "reduction", "parallel"]}
  // CHECK-SAME: ins(%{{.*}}, %{{.*}} : tensor<2x28x24x2x1xf64>, tensor<13x13x1x2x1xf64>)
  // CHECK-SAME: outs(%{{.*}} : tensor<2x6x8x2x1xf64>) {
  // CHECK: ^bb0(%[[LHS:.*]]: f64, %[[RHS:.*]]: f64, %[[OUT:.*]]: f64):
    // CHECK: %[[MUL:.*]] = arith.mulf %[[LHS]], %[[RHS]] : f64
    // CHECK: %[[RES:.*]] = arith.addf %[[OUT]], %[[MUL]] : f64
    // CHECK: linalg.yield %[[RES]] : f64
  // CHECK: } -> tensor<2x6x8x2x1xf64>
  // Check proper shape is returned
  // CHECK: return {{.*}} : tensor<2x6x8x2xf64>

func.func @feature_group_count_convolution(%arg0: tensor<2x14x12x2xf64>, %arg1: tensor<7x7x1x2xf64>)
                            -> tensor<2x6x8x2xf64> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[1, 0], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [2, 2], reverse = [1, 1]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]}
    : (tensor<2x14x12x2xf64>, tensor<7x7x1x2xf64>) -> tensor<2x6x8x2xf64>
  return %0 : tensor<2x6x8x2xf64>
}

// -----
// The following test is identical to the previous one, except that the
// `mhlo.convolution` op lacks the (optional) `window_stride` and
// `window_reverse` attributes. The goal of this test is to make sure that the
// compiler does not segfault, so we simply check for the existence of the
// function in the output IR.

// CHECK-LABEL: @convolution_without_reversing_and_stride
// CHECK-SAME: %[[ARG0:.*]]: tensor<2x14x12x2xf64>
// CHECK-SAME: %[[ARG1:.*]]: tensor<7x7x1x2xf64>)
// CHECK-SAME -> tensor<2x12x16x2xf64>

func.func @convolution_without_reversing_and_stride(%arg0: tensor<2x14x12x2xf64>, %arg1: tensor<7x7x1x2xf64>) -> tensor<2x12x16x2xf64> {
  %0 = mhlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {pad = [[1, 0], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [2, 2]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision HIGHEST>, #mhlo<precision HIGHEST>]}
    : (tensor<2x14x12x2xf64>, tensor<7x7x1x2xf64>) -> tensor<2x12x16x2xf64>
  return %0 : tensor<2x12x16x2xf64>
}

// -----

// CHECK-DAG: affine_map<(d0, d1, d2, d3) -> (-d0 + 2, -d1 + 2, d2, d3)>
// CHECK-LABEL: @normal_convolution_with_reversal
func.func @normal_convolution_with_reversal(%arg0: tensor<1x3x3x3xf32>,
    %arg1: tensor<3x3x3x1xf32>) -> tensor<1x1x1x1xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {
        stride = [1, 1],
        pad = [[0, 0], [0, 0]],
        lhs_dilate = [1, 1],
        rhs_dilate = [1, 1],
        reverse = [1, 1]
      } {
        batch_group_count = 1 : i64,
        feature_group_count = 1 : i64, precision_config = [
          #mhlo<precision DEFAULT>,
          #mhlo<precision DEFAULT>]
      } : (tensor<1x3x3x3xf32>, tensor<3x3x3x1xf32>) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}

// -----

// CHECK-LABEL: set_dimension_size
// CHECK-SAME: %[[VALUE:.*]]: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>
func.func @set_dimension_size(
  %value: tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>,
  %dimension: tensor<i32>)
  -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>> {
  // CHECK: tensor.extract_slice %[[VALUE]][0, 0] [2, %{{.*}}] [1, 1] : tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>> to tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
  %0 = "mhlo.set_dimension_size"(%value, %dimension) { dimension = 1 }
    : (tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>, tensor<i32>)
    -> tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
  func.return %0 : tensor<2x?xf32, #mhlo.type_extensions<bounds = [?, 2]>>
}

// -----
// The following test checks that an EmptyOp is emitted for mhlo.convolution
// when the output shape has a zero-sized dimension. This goes through
// ConvolutionOpGeneralConversion rewrite pattern.

// CHECK-LABEL: @general_convolution_with_zero_sized_dimension_in_output
//  CHECK-SAME: %[[LHS:.*]]: tensor<2x4x9x0xi64>
//  CHECK-SAME: %[[RHS:.*]]: tensor<4x5x2x4xi64>
//  CHECK-SAME: -> tensor<2x5x0x4xi64>
//  CHECK-NEXT: %[[RES:.*]] = tensor.empty
//  CHECK-NEXT: return %[[RES]]

func.func @general_convolution_with_zero_sized_dimension_in_output(%arg0: tensor<2x4x9x0xi64> {bufferization.writable = false, xla_framework.input_mapping = 2 : i32},
%arg1: tensor<4x5x2x4xi64> {bufferization.writable = false, xla_framework.input_mapping = 0 : i32})
-> tensor<2x5x0x4xi64> attributes {xla_framework.result_mapping = 1 : i32} {
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 1], pad = [[1, 2], [2, 0]], lhs_dilate = [1, 4], rhs_dilate = [1, 1], reverse = [0, 0]}
    {batch_group_count = 1 : i64, feature_group_count = 2 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
    : (tensor<2x4x9x0xi64>, tensor<4x5x2x4xi64>) -> tensor<2x5x0x4xi64>
  return %0 : tensor<2x5x0x4xi64>
}

// -----
// This test is similar to the previous one, but runs through a different
// rewrite pattern (NormalConvolutionOpConversion).

// CHECK-LABEL: @normal_convolution_with_zero_sized_dimension_in_output
//  CHECK-SAME: %[[LHS:.*]]: tensor<3x9x0x2xi16>
//  CHECK-SAME: %[[RHS:.*]]: tensor<4x5x2x2xi16>
//  CHECK-SAME: -> tensor<3x9x0x2xi16>
//  CHECK-NEXT: %[[RES:.*]] = tensor.empty
//  CHECK-NEXT: return %[[RES]]

func.func @normal_convolution_with_zero_sized_dimension_in_output(%arg0: tensor<3x9x0x2xi16> {bufferization.writable = false, xla_framework.input_mapping = 2 : i16},
%arg1: tensor<4x5x2x2xi16> {bufferization.writable = false, xla_framework.input_mapping = 0 : i16})
-> tensor<3x9x0x2xi16> attributes {xla_framework.result_mapping = 1 : i16} {
  %0 = mhlo.convolution(%arg0, %arg1) dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[1, 2], [2, 0]], lhs_dilate = [1, 2], rhs_dilate = [1, 4], reverse = [0, 0]}
    {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
    : (tensor<3x9x0x2xi16>, tensor<4x5x2x2xi16>) -> tensor<3x9x0x2xi16>
  return %0 : tensor<3x9x0x2xi16>
}
