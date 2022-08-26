// RUN: stablehlo-opt %s | FileCheck %s
// RUN: stablehlo-opt %s | stablehlo-opt | FileCheck %s

// CHECK-LABEL: func @zero_input
func.func @zero_input() -> !stablehlo.token {
  // CHECK:      %0 = stablehlo.replica_id : tensor<ui32>
  // CHECK-NEXT: %1 = stablehlo.create_token : !stablehlo.token
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  %1 = "stablehlo.create_token"() : () -> !stablehlo.token
  return %1 : !stablehlo.token
}

// CHECK-LABEL: func @zero_output_ret2
func.func @zero_output_ret2(%arg0 : tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
  // CHECK:      stablehlo.trace %arg0, "This is a test" : tensor<3xi64>
  // CHECK-NEXT: stablehlo.return %arg0, %arg0 : tensor<3xi64>, tensor<3xi64>
  "stablehlo.trace"(%arg0) {tag = "This is a test"} : (tensor<3xi64>) -> ()
  "stablehlo.return"(%arg0, %arg0) : (tensor<3xi64>, tensor<3xi64>) -> ()
}

// CHECK-LABEL: func @zero_output_ret1
func.func @zero_output_ret1(%arg0 : tensor<3xi64>) -> (tensor<3xi64>) {
  // CHECK:     stablehlo.return %arg0 : tensor<3xi64>
  "stablehlo.return"(%arg0) : (tensor<3xi64>) -> ()
}

// CHECK-LABEL: func @zero_output_ret0
func.func @zero_output_ret0(%arg0 : tensor<3xi64>) -> () {
  // CHECK:     stablehlo.return
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @unary_ops
func.func @unary_ops(%arg0 : tensor<2xi32>, %arg1 : tensor<2xf32>) -> () {
  // CHECK:      %0 = stablehlo.abs %arg0 : tensor<2xi32>
  // CHECK-NEXT: %1 = stablehlo.ceil %arg1 : tensor<2xf32>
  // CHECK-NEXT: %2 = stablehlo.count_leading_zeros %arg0 : tensor<2xi32>
  // CHECK-NEXT: %3 = stablehlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
  // CHECK-NEXT: %4 = stablehlo.cosine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = stablehlo.exponential %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = stablehlo.exponential_minus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = stablehlo.floor %arg1 : tensor<2xf32>
  // CHECK-NEXT: %8 = stablehlo.imag %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = stablehlo.is_finite %arg1 : (tensor<2xf32>) -> tensor<2xi1>
  // CHECK-NEXT: %10 = stablehlo.log %arg1 : tensor<2xf32>
  // CHECK-NEXT: %11 = stablehlo.log_plus_one %arg1 : tensor<2xf32>
  // CHECK-NEXT: %12 = stablehlo.logistic %arg1 : tensor<2xf32>
  // CHECK-NEXT: %13 = stablehlo.not %arg0 : tensor<2xi32>
  // CHECK-NEXT: %14 = stablehlo.negate %arg1 : tensor<2xf32>
  // CHECK-NEXT: %15 = stablehlo.popcnt %arg0 : tensor<2xi32>
  // CHECK-NEXT: %16 = stablehlo.real %arg1 : tensor<2xf32>
  // CHECK-NEXT: %17 = stablehlo.round_nearest_afz %arg1 : tensor<2xf32>
  // CHECK-NEXT: %18 = stablehlo.round_nearest_even %arg1 : tensor<2xf32>
  // CHECK-NEXT: %19 = stablehlo.sign %arg1 : tensor<2xf32>
  // CHECK-NEXT: %20 = stablehlo.sine %arg1 : tensor<2xf32>
  // CHECK-NEXT: %21 = stablehlo.sqrt %arg1 : tensor<2xf32>
  // CHECK-NEXT: %22 = stablehlo.tanh %arg1 : tensor<2xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %1 = "stablehlo.ceil"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %2 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %3 = "stablehlo.convert"(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %4 = "stablehlo.cosine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %5 = "stablehlo.exponential"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %6 = "stablehlo.exponential_minus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %7 = "stablehlo.floor"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %8 = "stablehlo.imag"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %9 = "stablehlo.is_finite"(%arg1) : (tensor<2xf32>) -> tensor<2xi1>
  %10 = "stablehlo.log"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %11 = "stablehlo.log_plus_one"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %12 = "stablehlo.logistic"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %13 = "stablehlo.not"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %14 = "stablehlo.negate"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %15 = "stablehlo.popcnt"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  %16 = "stablehlo.real"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %17 = "stablehlo.round_nearest_afz"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %18 = "stablehlo.round_nearest_even"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %19 = "stablehlo.sign"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %20 = "stablehlo.sine"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %21 = "stablehlo.sqrt"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  %22 = "stablehlo.tanh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
  "stablehlo.return"(%0) : (tensor<2xi32>) -> ()
}

// CHECK-LABEL: func @binary_ops
func.func @binary_ops(%arg0: tensor<2xi1>, %arg1 : tensor<2xf32>) -> tensor<2xi1> {
  // CHECK:      %0 = stablehlo.add %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %1 = stablehlo.and %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %2 = stablehlo.atan2 %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %3 = stablehlo.divide %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %4 = stablehlo.maximum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %5 = stablehlo.minimum %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %6 = stablehlo.multiply %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %7 = stablehlo.or %arg0, %arg0 : tensor<2xi1>
  // CHECK-NEXT: %8 = stablehlo.power %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %9 = stablehlo.remainder %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %10 = stablehlo.shift_left %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %11 = stablehlo.shift_right_arithmetic %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %12 = stablehlo.shift_right_logical %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %13 = stablehlo.subtract %arg1, %arg1 : tensor<2xf32>
  // CHECK-NEXT: %14 = stablehlo.xor %arg0, %arg0 : tensor<2xi1>
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %1 = "stablehlo.and"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %2 = "stablehlo.atan2"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %3 = "stablehlo.divide"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %4 = "stablehlo.maximum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %5 = "stablehlo.minimum"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %6 = "stablehlo.multiply"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %7 = "stablehlo.or"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  %8 = "stablehlo.power"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %9 = "stablehlo.remainder"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %10 = "stablehlo.shift_left"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %11 = "stablehlo.shift_right_arithmetic"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %12 = "stablehlo.shift_right_logical"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %13 = "stablehlo.subtract"(%arg1, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %14 = "stablehlo.xor"(%arg0, %arg0) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// CHECK-LABEL: func @type_convert_ops
func.func @type_convert_ops(%arg0 : tensor<2xf32>) -> () {
  // CHECK:      %0 = stablehlo.convert %arg0 : (tensor<2xf32>) -> tensor<2xf64>
  // CHECK-NEXT: %1 = stablehlo.reshape %arg0 : (tensor<2xf32>) -> tensor<1x2xf32>
  // CHECK-NEXT: %2 = stablehlo.bitcast_convert %arg0 : (tensor<2xf32>) -> tensor<2xi32>
  %0 = "stablehlo.convert"(%arg0) : (tensor<2xf32>) -> tensor<2xf64>
  %1 = "stablehlo.reshape"(%arg0) : (tensor<2xf32>) -> tensor<1x2xf32>
  %2 = "stablehlo.bitcast_convert"(%arg0) : (tensor<2xf32>) -> tensor<2xi32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @no_attr_ops
func.func @no_attr_ops(%arg0 : tensor<4xf32>, %arg1 : !stablehlo.token,
                       %arg2 : tensor<4xi32>, %arg3 : index) -> !stablehlo.token {
  // CHECK-NEXT: %0 = stablehlo.clamp %arg0, %arg0, %arg0 : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  // CHECK-NEXT: %1 = stablehlo.complex(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  // CHECK-NEXT: %2 = stablehlo.compute_reshape_shape %arg3, %arg2 : index, tensor<4xi32> -> tensor<4xi32>
  // CHECK-NEXT: %3 = stablehlo.uniform_quantize %arg0 : (tensor<4xf32>) -> tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>
  // CHECK-NEXT: %4 = stablehlo.uniform_dequantize %3 : (tensor<4x!quant.uniform<u8:f32, 3.400000e+01:16>>) -> tensor<4xf32>
  // CHECK-NEXT: %5 = stablehlo.after_all %arg1, %arg1 : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
  %0 = "stablehlo.clamp"(%arg0, %arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  %1 = "stablehlo.complex"(%arg0, %arg0) {} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %2 = "stablehlo.compute_reshape_shape"(%arg3, %arg2) : (index, tensor<4xi32>) -> tensor<4xi32>
  %3 = "stablehlo.uniform_quantize"(%arg0) : (tensor<4xf32>) -> tensor<4x!quant.uniform<ui8:f32, 34.0:16>>
  %4 = "stablehlo.uniform_dequantize"(%3) : (tensor<4x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<4xf32>
  %5 = "stablehlo.after_all"(%arg1, %arg1) : (!stablehlo.token, !stablehlo.token) -> !stablehlo.token
  "stablehlo.return"(%arg1) : (!stablehlo.token) -> ()
}

// CHECK-LABEL: func @tuple_ops
func.func @tuple_ops(%arg0 : tensor<i32>) -> () {
  // CHECK:      %0 = stablehlo.tuple %arg0, %arg0 : tuple<tensor<i32>, tensor<i32>>
  // CHECK-NEXT: %1 = stablehlo.tuple %arg0 : tuple<tensor<i32>>
  // CHECK-NEXT: %2 = stablehlo.tuple : tuple<>
  // CHECK-NEXT: %3 = stablehlo.get_tuple_element %1[0] : (tuple<tensor<i32>>) -> tensor<i32>
  %0 = "stablehlo.tuple"(%arg0, %arg0) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
  %1 = "stablehlo.tuple"(%arg0) : (tensor<i32>) -> tuple<tensor<i32>>
  %2 = "stablehlo.tuple"() : () -> tuple<>
  %3 = "stablehlo.get_tuple_element"(%1) {index = 0 : i32} : (tuple<tensor<i32>>) -> tensor<i32>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @pairwise_ops
func.func @pairwise_ops(%arg0 : tensor<4xf32>) -> () {
  // CHECK:      stablehlo.optimization_barrier()
  // CHECK-NEXT: %0 = stablehlo.optimization_barrier %arg0 : tensor<4xf32>
  // CHECK-NEXT: %1:2 = stablehlo.optimization_barrier %arg0, %arg0 : tensor<4xf32>, tensor<4xf32>
  "stablehlo.optimization_barrier"() : () -> ()
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  %1:2 = "stablehlo.optimization_barrier"(%arg0, %arg0) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @compare_op
func.func @compare_op(%arg0 : tensor<3xi32>) -> () {
  // CHECK:      %0 = stablehlo.compare LT, %arg0, %arg0 : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  // CHECK-NEXT: %1 = stablehlo.compare LT, %arg0, %arg0, TOTALORDER : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %0 = "stablehlo.compare"(%arg0, %arg0) {comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
   %1 = "stablehlo.compare"(%arg0, %arg0) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction LT>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  "stablehlo.return"() : () -> ()
}

// CHECK-LABEL: func @extensions
func.func @extensions(%arg0 : tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>,
                %arg1 : tensor<i32>) -> () {
  // CHECK:      %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>
  %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {dimension = 1 : i64} : (tensor<?x?xf32, #stablehlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>
  "stablehlo.return"() : () -> ()
}

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

// CHECK-LABEL: func @encodings
func.func @encodings(%arg0: tensor<10x20xf32, #CSR>,
                     %arg1: tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32> {
  // CHECK:      %0 = stablehlo.add %arg0, %arg1 : (tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>, tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>) -> tensor<10x20xf32>
  // CHECK-NEXT: %1 = stablehlo.add %arg1, %arg1 : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ] }>>
  // CHECK-NEXT: %2 = stablehlo.abs %arg0 : (tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>) -> tensor<10x20xf32>
  // CHECK-NEXT: %3 = stablehlo.abs %arg0 : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ] }>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32>
  %1 = "stablehlo.add"(%arg1, %arg1) : (tensor<10x20xf32, #DCSR>,
                                   tensor<10x20xf32, #DCSR>) -> tensor<10x20xf32, #DCSR>
  %2 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32>
  %3 = "stablehlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>) -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32>
}
