// RUN: mlir-hlo-opt -split-input-file %s | FileCheck %s
// RUN: mlir-hlo-opt -split-input-file %s | mlir-hlo-opt -split-input-file | FileCheck %s

// -----

func.func @zero_input() -> !mhlo.token {
  // CHECK:      %0 = mhlo.replica_id : tensor<ui32>
  // CHECK-NEXT: %1 = mhlo.partition_id : tensor<ui32>
  // CHECK-NEXT: %2 = mhlo.create_token : !mhlo.token
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  %1 = "mhlo.partition_id"() : () -> tensor<ui32>
  %2 = "mhlo.create_token"() : () -> !mhlo.token
  return %2 : !mhlo.token
}

// -----

func.func @zero_output_ret2(%arg0 : tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
  // CHECK:      mhlo.trace %arg0, "This is a test" : tensor<3xi64>
  // CHECK-NEXT: mhlo.return %arg0, %arg0 : tensor<3xi64>, tensor<3xi64>
  "mhlo.trace"(%arg0) {tag = "This is a test"} : (tensor<3xi64>) -> ()
  "mhlo.return"(%arg0, %arg0) : (tensor<3xi64>, tensor<3xi64>) -> ()
}

func.func @zero_output_ret1(%arg0 : tensor<3xi64>) -> (tensor<3xi64>) {
  // CHECK:     mhlo.return %arg0 : tensor<3xi64>
  "mhlo.return"(%arg0) : (tensor<3xi64>) -> ()
}

func.func @zero_output_ret0(%arg0 : tensor<3xi64>) -> () {
  // CHECK:     mhlo.return
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @unary_ops
func.func @unary_ops(%arg0 : tensor<2xi32>, %arg1 : tensor<2xf32>) -> () {
  // CHECK:      %0 = mhlo.abs %arg0 : tensor<2xi32>
  // CHECK-NEXT: %1 = mhlo.ceil %arg1 : tensor<2xf32>
  // CHECK-NEXT: %2 = mhlo.count_leading_zeros %arg0 : tensor<2xi32>
  // CHECK-NEXT: %3 = mhlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK-NEXT: %4 = mhlo.cosine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = mhlo.exponential %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = mhlo.exponential_minus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = mhlo.floor %arg1 : tensor<2xf32>
  // CHECK-NEXT: %8 = mhlo.imag %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = mhlo.is_finite %arg1 : (tensor<2xf32>) -> tensor<2xi1>
  // CHECK-NEXT: %10 = mhlo.log %arg1 : tensor<2xf32>
  // CHECK-NEXT: %11 = mhlo.log_plus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %12 = mhlo.logistic %arg1 : tensor<2xf32>
  // CHECK-NEXT: %13 = mhlo.not %arg0 : tensor<2xi32>
  // CHECK-NEXT: %14 = mhlo.negate %arg1 : tensor<2xf32>
  // CHECK-NEXT: %15 = mhlo.popcnt %arg0 : tensor<2xi32>
  // CHECK-NEXT: %16 = mhlo.real %arg1 : tensor<2xf32>
  // CHECK-NEXT: %17 = mhlo.round_nearest_afz %arg1 : tensor<2xf32>
  // CHECK-NEXT: %18 = mhlo.round_nearest_even %arg1 : tensor<2xf32>
  // CHECK-NEXT: %19 = mhlo.sign %arg1 : tensor<2xf32>
  // CHECK-NEXT: %20 = mhlo.sine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %21 = mhlo.sqrt %arg1 : tensor<2xf32>
  // CHECK-NEXT: %22 = mhlo.tanh %arg1 : tensor<2xf32>
  %0 = "mhlo.abs"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %1 = "mhlo.ceil"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "mhlo.count_leading_zeros"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %3 = "mhlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %4 = "mhlo.cosine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %5 = "mhlo.exponential"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %6 = "mhlo.exponential_minus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %7 = "mhlo.floor"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %8 = "mhlo.imag"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %9 = "mhlo.is_finite"(%arg1) : (tensor<2xf32>) -> tensor<2xi1>
  %10 = "mhlo.log"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %11 = "mhlo.log_plus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %12 = "mhlo.logistic"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %13 = "mhlo.not"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %14 = "mhlo.negate"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %15 = "mhlo.popcnt"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %16 = "mhlo.real"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %17 = "mhlo.round_nearest_afz"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %18 = "mhlo.round_nearest_even"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %19 = "mhlo.sign"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %20 = "mhlo.sine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %21 = "mhlo.sqrt"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %22 = "mhlo.tanh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  "mhlo.return"(%0) : (tensor<2xi32>) -> ()
}

// -----

// CHECK-LABEL: func @binary_ops
func.func @binary_ops(%arg0: tensor<2xi1>, %arg1 : tensor<2xf32>, %arg2 : tensor<2xi32>) -> tensor<2xi1> {
  // CHECK:      %0 = mhlo.add %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %1 = mhlo.and %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %2 = mhlo.atan2 %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %3 = mhlo.divide %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %4 = mhlo.maximum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = mhlo.minimum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = mhlo.multiply %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = mhlo.or %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %8 = mhlo.power %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = mhlo.remainder %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %10 = mhlo.shift_left %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %11 = mhlo.shift_right_arithmetic %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %12 = mhlo.shift_right_logical %arg2, %arg2 : tensor<2xi32>
  // CHECK-NEXT: %13 = mhlo.subtract %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %14 = mhlo.xor %arg0, %arg0 : tensor<2xi1>
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "mhlo.and"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %2 = "mhlo.atan2"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %3 = "mhlo.divide"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %4 = "mhlo.maximum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %5 = "mhlo.minimum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %6 = "mhlo.multiply"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %7 = "mhlo.or"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %8 = "mhlo.power"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %9 = "mhlo.remainder"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %10 = "mhlo.shift_left"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %11 = "mhlo.shift_right_arithmetic"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %12 = "mhlo.shift_right_logical"(%arg2, %arg2) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  %13 = "mhlo.subtract"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %14 = "mhlo.xor"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// -----

// CHECK-LABEL: func @type_convert_ops
func.func @type_convert_ops(%arg0 : tensor<2xf32>) -> () {
  // CHECK:      %0 = mhlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
  // CHECK-NEXT: %1 = mhlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %2 = mhlo.bitcast_convert %arg0 : (tensor<2xf32>) -> tensor<2xi32>
  // CHECK-NEXT: %3 = mhlo.bitcast %arg0 : (tensor<2xf32>) -> tensor<2x1xf32>
  %0 = "mhlo.convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf64>
  %1 = "mhlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  %2 = "mhlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  %3 = "mhlo.bitcast"(%arg0) : (tensor<2xf32>) -> tensor<2x1xf32>
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @no_attr_ops
func.func @no_attr_ops(%arg0 : tensor<4xf32>, %arg1 : !mhlo.token,
                       %arg2 : tensor<4xi32>, %arg3 : index) -> !mhlo.token {
  // CHECK:      mhlo.add_dependency %arg0, %arg1 : (tensor<4xf32>, !mhlo.token) -> tensor<4xf32>
  // CHECK-NEXT: mhlo.clamp %arg0, %arg0, %arg0 : tensor<4xf32>
  // CHECK-NEXT: mhlo.complex %arg0, %arg0 : tensor<4xcomplex<f32>>
  // CHECK-NEXT: mhlo.copy %arg2 : tensor<4xi32>
  // CHECK-NEXT: mhlo.uniform_quantize %arg0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>
  // CHECK-NEXT: mhlo.uniform_dequantize %[[_:[0-9]+]] : (tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<4xf32>
  // CHECK-NEXT: mhlo.after_all %arg1, %arg1 : !mhlo.token
  // CHECK-NEXT: mhlo.after_all : !mhlo.token
  %0 = "mhlo.add_dependency"(%arg0, %arg1) : (tensor<4xf32>, !mhlo.token) -> tensor<4xf32>
  %1 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %2 = "mhlo.complex"(%arg0, %arg0) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %3 = "mhlo.copy"(%arg2) : (tensor<4xi32>) -> tensor<4xi32>
  %4 = "mhlo.uniform_quantize"(%arg0) : (tensor<4xf32>) -> tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
  %5 = "mhlo.uniform_dequantize"(%4) : (tensor<4x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4xf32>
  %6 = "mhlo.after_all"(%arg1, %arg1) : (!mhlo.token, !mhlo.token) -> !mhlo.token
  %7 = "mhlo.after_all"() : () -> !mhlo.token
  "mhlo.return"(%arg1) : (!mhlo.token) -> ()
}

// -----

// CHECK-LABEL: func @tuple_ops
func.func @tuple_ops(%arg0 : tensor<i32>) -> () {
  // CHECK:      %0 = mhlo.tuple %arg0, %arg0 : tuple<tensor<i32>, tensor<i32>>
  // CHECK-NEXT: %1 = mhlo.tuple %arg0 : tuple<tensor<i32>>
  // CHECK-NEXT: %2 = mhlo.tuple : tuple<>
  // CHECK-NEXT: %3 = mhlo.get_tuple_element %1[0] : (tuple<tensor<i32>>) -> tensor<i32>
  %0 = "mhlo.tuple"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
  %1 = "mhlo.tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  %2 = "mhlo.tuple"() : () -> tuple<>
  %3 = "mhlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @pairwise_ops
func.func @pairwise_ops(%arg0 : tensor<4xf32>) -> () {
  // CHECK:      mhlo.optimization_barrier()
  // CHECK-NEXT: %0 = mhlo.optimization_barrier %arg0 : tensor<4xf32>
  // CHECK-NEXT: %1:2 = mhlo.optimization_barrier %arg0, %arg0 : tensor<4xf32>, tensor<4xf32>
  "mhlo.optimization_barrier"() : () -> ()
  %0 = "mhlo.optimization_barrier"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1:2 = "mhlo.optimization_barrier"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @select_op
func.func @select_op(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>,
                  %arg2: tensor<2x?xi32>, %arg3: tensor<?x2xi32>) -> () {
  // CHECK      %0 = mhlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>, tensor<2x3xi32>
  // CHECK-NEXT %1 = mhlo.select %arg0, %arg2, %arg3 : (tensor<2x3xi1>, tensor<2x?xi32>, tensor<?x2xi32>) -> tensor<2x?xi32>
  %0 = "mhlo.select"(%arg0, %arg1, %arg1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  %1 = "mhlo.select"(%arg0, %arg2, %arg3) : (tensor<2x3xi1>, tensor<2x?xi32>, tensor<?x2xi32>) -> tensor<2x?xi32>
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @compare_op
func.func @compare_op(%arg0 : tensor<3xi32>) -> () {
  // CHECK:      %0 = mhlo.compare LT, %arg0, %arg0 : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: %1 = mhlo.compare LT, %arg0, %arg0, TOTALORDER : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %0 = "mhlo.compare"(%arg0, %arg0) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %1 = "mhlo.compare"(%arg0, %arg0) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  "mhlo.return"() : () -> ()
}

// -----

// CHECK-LABEL: func @extensions
func.func @extensions(%arg0 : tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>,
                %arg1 : tensor<i32>) -> () {
  // CHECK:      %0 = "mhlo.set_dimension_size"(%arg0, %arg1) <{dimension = 1 : i64}> : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>
  %0 = "mhlo.set_dimension_size"(%arg0, %arg1) <{dimension = 1 : i64}> : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, ?]>>, tensor<i32>) -> tensor<?x?xf32>
  "mhlo.return"() : () -> ()
}

// -----

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed, d1 : compressed)
}>

// CHECK: #[[$CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK: #[[$DCSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : compressed, d1 : compressed) }>
// CHECK-LABEL: func @encodings
func.func @encodings(%arg0: tensor<10x20xf32, #CSR>,
                     %arg1: tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32> {
  // CHECK:      %0 = mhlo.add %arg0, %arg1 : (tensor<10x20xf32, #[[$CSR]]>, tensor<10x20xf32, #[[$DCSR]]>) -> tensor<10x20xf32>
  // CHECK-NEXT: %1 = mhlo.add %arg1, %arg1 : tensor<10x20xf32, #[[$DCSR]]
  // CHECK-NEXT: %2 = mhlo.abs %arg0 : (tensor<10x20xf32, #[[$CSR]]>) -> tensor<10x20xf32>
  // CHECK-NEXT: %3 = mhlo.abs %arg0 : tensor<10x20xf32, #[[$CSR]]>
  // CHECK-NEXT: %4 = mhlo.complex %arg0, %arg0 : (tensor<10x20xf32, #[[$CSR]]>, tensor<10x20xf32, #[[$CSR]]>) -> tensor<10x20xcomplex<f32>>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32>
  %1 = "mhlo.add"(%arg1, %arg1) : (tensor<10x20xf32, #DCSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32, #DCSR>
  %2 = "mhlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32>
  %3 = "mhlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #CSR>
  %4 = "mhlo.complex"(%arg0, %arg0) : (tensor<10x20xf32, #CSR>, tensor<10x20xf32, #CSR>) -> tensor<10x20xcomplex<f32>>
  func.return %0 : tensor<10x20xf32>
}
