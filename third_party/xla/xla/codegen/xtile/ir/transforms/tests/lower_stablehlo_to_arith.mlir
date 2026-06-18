// RUN: emitters_opt %s -stablehlo-lower-to-arith | FileCheck %s

func.func @lower_add_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.addi
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_add_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.addi
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_add_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.addf
  %0 = stablehlo.add %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_sub_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.subi
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_sub_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.subi
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_sub_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.subf
  %0 = stablehlo.subtract %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_divide_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.divsi
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_divide_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: arith.divui
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_divide_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.divf
  %0 = stablehlo.divide %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_rem_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.remsi
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_rem_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: arith.remui
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_rem_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.remf
  %0 = stablehlo.remainder %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_multiply_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.muli
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_multiply_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.muli
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_multiply_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.mulf
  %0 = stablehlo.multiply %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_xor_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.xori
  %0 = stablehlo.xor %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_xor_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.xori
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.xor %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_or_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.ori
  %0 = stablehlo.or %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_or_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.ori
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.or %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_and_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.andi
  %0 = stablehlo.and %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_and_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui32> to tensor<2x4xi32>
  // CHECK: arith.andi
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.and %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_maximum_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.maxsi
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_maximum_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: arith.maxui
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_maximum_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.maximumf
  %0 = stablehlo.maximum %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

func.func @lower_minimum_with_signless_operands(%arg0 : tensor<2x4xi32>, %arg1 : tensor<2x4xi32>) -> tensor<2x4xi32> {
  // CHECK: arith.minsi
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

func.func @lower_minimum_with_unsigned_operands(%arg0 : tensor<2x4xui32>, %arg1 : tensor<2x4xui32>) -> tensor<2x4xui32> {
  // CHECK: arith.minui
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

func.func @lower_minimum_with_float_operands(%arg0 : tensor<2x4xf32>, %arg1 : tensor<2x4xf32>) -> tensor<2x4xf32> {
  // CHECK: arith.minimumf
  %0 = stablehlo.minimum %arg0, %arg1 : tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}