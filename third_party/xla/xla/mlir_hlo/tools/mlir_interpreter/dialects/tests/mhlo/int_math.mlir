// RUN: mlir-interpreter-runner %s -run-all | FileCheck %s

func.func @abs() -> tensor<1xi32> {
  %c-3 = mhlo.constant dense<-3> : tensor<1xi32>
  %ret = mhlo.abs %c-3 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @abs
// CHECK-NEXT: Results
// CHECK-NEXT: [3]

func.func @add_tensor() -> tensor<2x3xi32> {
  %lhs = mhlo.constant dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi32>
  %rhs = mhlo.constant dense<[[10, 20, 30], [40, 50, 60]]> : tensor<2x3xi32>
  %result = mhlo.add %lhs, %rhs : tensor<2x3xi32>
  return %result : tensor<2x3xi32>
}

// CHECK-LABEL: @add_tensor
// CHECK-NEXT: Results
// CHECK-NEXT{LITERAL}: [[10, 21, 32], [43, 54, 65]]

func.func @add_scalar_i8() -> tensor<i8> {
  %lhs = mhlo.constant dense<40> : tensor<i8>
  %rhs = mhlo.constant dense<2> : tensor<i8>
  %result = mhlo.add %lhs, %rhs : tensor<i8>
  return %result : tensor<i8>
}

// CHECK-LABEL: @add_scalar_i8
// CHECK-NEXT: Results
// CHECK-NEXT: <i8>: 42

func.func @add_scalar_i16() -> tensor<i16> {
  %lhs = mhlo.constant dense<40> : tensor<i16>
  %rhs = mhlo.constant dense<2> : tensor<i16>
  %result = mhlo.add %lhs, %rhs : tensor<i16>
  return %result : tensor<i16>
}

// CHECK-LABEL: @add_scalar_i16
// CHECK-NEXT: Results
// CHECK-NEXT: <i16>: 42

func.func @add_scalar_i32() -> tensor<i32> {
  %lhs = mhlo.constant dense<40> : tensor<i32>
  %rhs = mhlo.constant dense<2> : tensor<i32>
  %result = mhlo.add %lhs, %rhs : tensor<i32>
  return %result : tensor<i32>
}

// CHECK-LABEL: @add_scalar_i32
// CHECK-NEXT: Results
// CHECK-NEXT: <i32>: 42

func.func @and() -> tensor<1xi32> {
  %c63 = mhlo.constant dense<63> : tensor<1xi32>
  %c131 = mhlo.constant dense<131> : tensor<1xi32>
  %ret = mhlo.and %c63, %c131 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @and
// CHECK-NEXT: Results
// CHECK-NEXT: [3]

func.func @and_i16() -> tensor<i16> {
  %c63 = mhlo.constant dense<63> : tensor<i16>
  %c131 = mhlo.constant dense<131> : tensor<i16>
  %ret = mhlo.and %c63, %c131 : tensor<i16>
  return %ret : tensor<i16>
}

// CHECK-LABEL: @and_i16
// CHECK-NEXT: Results
// CHECK-NEXT: <i16>: 3

func.func @clz_negative()
    -> (tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>) {
  %c-1_8 = mhlo.constant dense<-1> : tensor<1xi8>
  %c-2_16 = mhlo.constant dense<-2> : tensor<1xi16>
  %c-4_32 = mhlo.constant dense<-4> : tensor<1xi32>
  %c-8_64 = mhlo.constant dense<-8> : tensor<1xi64>
  %clz_8 = mhlo.count_leading_zeros %c-1_8 : tensor<1xi8>
  %clz_16 = mhlo.count_leading_zeros %c-2_16 : tensor<1xi16>
  %clz_32 = mhlo.count_leading_zeros %c-4_32 : tensor<1xi32>
  %clz_64 = mhlo.count_leading_zeros %c-8_64 : tensor<1xi64>
  return %clz_8, %clz_16, %clz_32, %clz_64
    : tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>
}

// CHECK-LABEL: @clz_negative
// CHECK-NEXT: Results
// CHECK-NEXT: [0]
// CHECK-NEXT: [0]
// CHECK-NEXT: [0]
// CHECK-NEXT: [0]

func.func @clz_signed()
    -> (tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>) {
  %c1_8 = mhlo.constant dense<1> : tensor<1xi8>
  %c2_16 = mhlo.constant dense<2> : tensor<1xi16>
  %c4_32 = mhlo.constant dense<4> : tensor<1xi32>
  %c8_64 = mhlo.constant dense<8> : tensor<1xi64>
  %clz_8 = mhlo.count_leading_zeros %c1_8 : tensor<1xi8>
  %clz_16 = mhlo.count_leading_zeros %c2_16 : tensor<1xi16>
  %clz_32 = mhlo.count_leading_zeros %c4_32 : tensor<1xi32>
  %clz_64 = mhlo.count_leading_zeros %c8_64 : tensor<1xi64>
  return %clz_8, %clz_16, %clz_32, %clz_64
    : tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>
}

// CHECK-LABEL: @clz_signed
// CHECK-NEXT: Results
// CHECK-NEXT: [7]
// CHECK-NEXT: [14]
// CHECK-NEXT: [29]
// CHECK-NEXT: [60]

func.func @clz_unsigned()
    -> (tensor<1xui8>, tensor<1xui16>, tensor<1xui32>, tensor<1xui64>) {
  %c1_8 = mhlo.constant dense<1> : tensor<1xui8>
  %c2_16 = mhlo.constant dense<2> : tensor<1xui16>
  %c4_32 = mhlo.constant dense<4> : tensor<1xui32>
  %c8_64 = mhlo.constant dense<8> : tensor<1xui64>
  %clz_8 = mhlo.count_leading_zeros %c1_8 : tensor<1xui8>
  %clz_16 = mhlo.count_leading_zeros %c2_16 : tensor<1xui16>
  %clz_32 = mhlo.count_leading_zeros %c4_32 : tensor<1xui32>
  %clz_64 = mhlo.count_leading_zeros %c8_64 : tensor<1xui64>
  return %clz_8, %clz_16, %clz_32, %clz_64
    : tensor<1xui8>, tensor<1xui16>, tensor<1xui32>, tensor<1xui64>
}

// CHECK-LABEL: @clz_unsigned
// CHECK-NEXT: Results
// CHECK-NEXT: [7]
// CHECK-NEXT: [14]
// CHECK-NEXT: [29]
// CHECK-NEXT: [60]

func.func @divide() -> tensor<1xi32> {
  %c-10 = mhlo.constant dense<-10> : tensor<1xi32>
  %c-2 = mhlo.constant dense<-2> : tensor<1xi32>
  %ret = mhlo.divide %c-10, %c-2 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @divide
// CHECK-NEXT: Results
// CHECK-NEXT: [5]

func.func @subtract() -> tensor<1xi32> {
  %c10 = mhlo.constant dense<10> : tensor<1xi32>
  %c3 = mhlo.constant dense<3> : tensor<1xi32>
  %ret = mhlo.subtract %c10, %c3 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @subtract
// CHECK-NEXT: Results
// CHECK-NEXT: [7]

func.func @or() -> tensor<1xi32> {
  %c3 = mhlo.constant dense<3> : tensor<1xi32>
  %c10 = mhlo.constant dense<10> : tensor<1xi32>
  %ret = mhlo.or %c3, %c10 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @or
// CHECK-NEXT: Results
// CHECK-NEXT: [11]

func.func @max_scalar() -> tensor<i32> {
  %lhs = mhlo.constant dense<40> : tensor<i32>
  %rhs = mhlo.constant dense<2> : tensor<i32>
  %result = mhlo.maximum %lhs, %rhs : tensor<i32>
  return %result : tensor<i32>
}

// CHECK-LABEL: @max_scalar
// CHECK-NEXT: Results
// CHECK-NEXT: 40

func.func @multiply() -> tensor<1xi32> {
  %c3 = mhlo.constant dense<3> : tensor<1xi32>
  %c-5 = mhlo.constant dense<-5> : tensor<1xi32>
  %ret = mhlo.multiply %c3, %c-5 : tensor<1xi32>
  return %ret : tensor<1xi32>
}

// CHECK-LABEL: @multiply
// CHECK-NEXT: Results
// CHECK-NEXT: [-15]

func.func @multiply_scalar_i16() -> tensor<i16> {
  %lhs = mhlo.constant dense<40> : tensor<i16>
  %rhs = mhlo.constant dense<2> : tensor<i16>
  %result = mhlo.multiply %lhs, %rhs : tensor<i16>
  return %result : tensor<i16>
}

// CHECK-LABEL: @multiply_scalar_i16
// CHECK-NEXT: Results
// CHECK-NEXT: <i16>: 80

func.func @multiply_scalar_ui16() -> tensor<ui16> {
  %lhs = mhlo.constant dense<40> : tensor<ui16>
  %rhs = mhlo.constant dense<2> : tensor<ui16>
  %result = mhlo.multiply %lhs, %rhs : tensor<ui16>
  return %result : tensor<ui16>
}

// CHECK-LABEL: @multiply_scalar_ui16
// CHECK-NEXT: Results
// CHECK-NEXT: <ui16>: 80

func.func @not_i1() -> tensor<2xi1> {
  %cst = mhlo.constant dense<[false, true]> : tensor<2xi1>
  %not = mhlo.not %cst : tensor<2xi1>
  return %not : tensor<2xi1>
}

// CHECK-LABEL: @not_i1
// CHECK-NEXT: Results
// CHECK-NEXT: [true, false]

func.func @not_ui16() -> tensor<ui16> {
  %cst = mhlo.constant dense<1> : tensor<ui16>
  %not = mhlo.not %cst : tensor<ui16>
  return %not : tensor<ui16>
}

// CHECK-LABEL: @not_ui16
// CHECK-NEXT: Results
// CHECK-NEXT: 65534

func.func @popcnt_negative()
    -> (tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>) {
  %c-1_8 = mhlo.constant dense<-1> : tensor<1xi8>
  %c-2_16 = mhlo.constant dense<-2> : tensor<1xi16>
  %c-4_32 = mhlo.constant dense<-4> : tensor<1xi32>
  %c-8_64 = mhlo.constant dense<-8> : tensor<1xi64>
  %pop_8 = mhlo.popcnt %c-1_8 : tensor<1xi8>
  %pop_16 = mhlo.popcnt %c-2_16 : tensor<1xi16>
  %pop_32 = mhlo.popcnt %c-4_32 : tensor<1xi32>
  %pop_64 = mhlo.popcnt %c-8_64 : tensor<1xi64>
  return %pop_8, %pop_16, %pop_32, %pop_64
    : tensor<1xi8>, tensor<1xi16>, tensor<1xi32>, tensor<1xi64>
}

// CHECK-LABEL: @popcnt_negative
// CHECK-NEXT: Results
// CHECK-NEXT: [8]
// CHECK-NEXT: [15]
// CHECK-NEXT: [30]
// CHECK-NEXT: [61]

func.func @pow() -> tensor<ui16> {
  %c2 = mhlo.constant dense<2> : tensor<ui16>
  %c5 = mhlo.constant dense<5> : tensor<ui16>
  %pow = mhlo.power %c2, %c5 : tensor<ui16>
  return %pow : tensor<ui16>
}

// CHECK-LABEL: @pow
// CHECK-NEXT: Results
// CHECK-NEXT: <ui16>: 32

func.func @pow_negative() -> tensor<3xi64> {
  %c = mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
  %c2 = mhlo.constant dense<-1> : tensor<3xi64>
  %pow = mhlo.power %c, %c2 : tensor<3xi64>
  return %pow : tensor<3xi64>
}

// CHECK-LABEL: @pow_negative
// CHECK-NEXT: Results
// CHECK-NEXT: [0, 1, 0]

func.func @pow_non_double() -> tensor<i64> {
  %c3 = mhlo.constant dense<3> : tensor<i64>
  %c35 = mhlo.constant dense<35> : tensor<i64>
  // The result of this operation cannot be represented by a double.
  %pow = mhlo.power %c3, %c35 : tensor<i64>
  return %pow : tensor<i64>
}

// CHECK-LABEL: @pow_non_double
// CHECK-NEXT: Results
// CHECK-NEXT: 50031545098999707

func.func @rem() -> tensor<4xi32> {
  %0 = mhlo.constant dense<[5, 66, 5, -1]> : tensor<4xi32>
  %1 = mhlo.constant dense<[3, 5, 1, -2]> : tensor<4xi32>
  %2 = mhlo.remainder %0, %1 : tensor<4xi32>
  func.return %2 : tensor<4xi32>
}

// CHECK-LABEL: @rem
// CHECK-NEXT: Results
// CHECK-NEXT: [2, 1, 0, -1]

func.func @shift_left() -> tensor<2xi32> {
  %0 = mhlo.constant dense<[3, 7]> : tensor<2xi32>
  %1 = mhlo.constant dense<[4, 31]> : tensor<2xi32>
  %ret = mhlo.shift_left %0, %1 : tensor<2xi32>
  func.return %ret : tensor<2xi32>
}

// CHECK-LABEL: @shift_left
// CHECK-NEXT: Results
// CHECK-NEXT: [48, -2147483648]

func.func @shift_right_arith() -> tensor<2xi32> {
  %0 = mhlo.constant dense<[100, -100]> : tensor<2xi32>
  %1 = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %ret = mhlo.shift_right_arithmetic %0, %1 : tensor<2xi32>
  func.return %ret : tensor<2xi32>
}

// CHECK-LABEL: @shift_right_arith
// CHECK-NEXT: Results
// CHECK-NEXT: [25, -25]

func.func @shift_right_arith_ui32() -> tensor<ui32> {
  %0 = mhlo.constant dense<100> : tensor<ui32>
  %1 = mhlo.constant dense<2> : tensor<ui32>
  %ret = mhlo.shift_right_arithmetic %0, %1 : tensor<ui32>
  func.return %ret : tensor<ui32>
}

// CHECK-LABEL: @shift_right_arith_ui32
// CHECK-NEXT: Results
// CHECK-NEXT: 25

func.func @shift_right_logical() -> tensor<2xi32> {
  %0 = mhlo.constant dense<[100, -100]> : tensor<2xi32>
  %1 = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %ret = mhlo.shift_right_logical %0, %1 : tensor<2xi32>
  func.return %ret : tensor<2xi32>
}

// CHECK-LABEL: @shift_right_logical
// CHECK-NEXT: Results
// CHECK-NEXT: [25, 1073741799]

func.func @shift_right_logical_ui32() -> tensor<ui32> {
  %0 = mhlo.constant dense<100> : tensor<ui32>
  %1 = mhlo.constant dense<2> : tensor<ui32>
  %ret = mhlo.shift_right_logical %0, %1 : tensor<ui32>
  func.return %ret : tensor<ui32>
}

// CHECK-LABEL: @shift_right_logical_ui32
// CHECK-NEXT: Results
// CHECK-NEXT: 25
