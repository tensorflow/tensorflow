// RUN: emitters_opt 
// %s -split-input-file -stablehlo-lower-to-xtile 
// | FileCheck %s

func.func @lower_convert_bf16_to_f32(%arg0: tensor<2x4xbf16>) -> tensor<2x4xf32> {
  // CHECK: %[[RES:.*]] = arith.extf %arg0 : tensor<2x4xbf16> to tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xbf16>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @lower_convert_f32_to_bf16(%arg0: tensor<2x4xf32>) -> tensor<2x4xbf16> {
  // CHECK: %[[RES:.*]] = arith.truncf %arg0 : tensor<2x4xf32> to tensor<2x4xbf16>
  // CHECK: return %[[RES]] : tensor<2x4xbf16>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf32>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// -----

func.func @lower_convert_i8_to_bf16(%arg0: tensor<2x4xi8>) -> tensor<2x4xbf16> {
  // CHECK: %[[RES:.*]] = arith.sitofp %arg0 : tensor<2x4xi8> to tensor<2x4xbf16>
  // CHECK: return %[[RES]] : tensor<2x4xbf16>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi8>) -> tensor<2x4xbf16>
  return %0 : tensor<2x4xbf16>
}

// -----

func.func @lower_convert_f64_to_f32(%arg0: tensor<2x4xf64>) -> tensor<2x4xf32> {
  // CHECK: %[[RES:.*]] = arith.truncf %arg0 : tensor<2x4xf64> to tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @lower_convert_f16_to_f32(%arg0: tensor<2x4xf16>) -> tensor<2x4xf32> {
  // CHECK: %[[RES:.*]] = arith.extf %arg0 : tensor<2x4xf16> to tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf16>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @lower_convert_f8E4M3FN_to_f8E5M2(%arg0: tensor<2x4xf8E4M3FN>) -> tensor<2x4xf8E5M2> {
  // CHECK: %[[FP16:.*]] = arith.extf %arg0 : tensor<2x4xf8E4M3FN> to tensor<2x4xf16>
  // CHECK: %[[RES:.*]] = arith.truncf %[[FP16]] : tensor<2x4xf16> to tensor<2x4xf8E5M2>
  // CHECK: return %[[RES]] : tensor<2x4xf8E5M2>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf8E4M3FN>) -> tensor<2x4xf8E5M2>
  return %0 : tensor<2x4xf8E5M2>
}

// -----

func.func @lower_convert_i8_to_i32(%arg0: tensor<2x4xi8>) -> tensor<2x4xi32> {
  // CHECK: %[[RES:.*]] = arith.extsi %arg0 : tensor<2x4xi8> to tensor<2x4xi32>
  // CHECK: return %[[RES]] : tensor<2x4xi32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi8>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @lower_convert_i1_to_i32(%arg0: tensor<2x4xi1>) -> tensor<2x4xi32> {
  // CHECK: %[[RES:.*]] = arith.extui %arg0 : tensor<2x4xi1> to tensor<2x4xi32>
  // CHECK: return %[[RES]] : tensor<2x4xi32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi1>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}

// -----

func.func @lower_convert_ui8_to_ui32(%arg0: tensor<2x4xui8>) -> tensor<2x4xui32> {
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xui8> to tensor<2x4xi8>
  // CHECK: %[[RES:.*]] = arith.extui %{{.*}} : tensor<2x4xi8> to tensor<2x4xi32>
  // CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<2x4xi32> to tensor<2x4xui32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xui8>) -> tensor<2x4xui32>
  return %0 : tensor<2x4xui32>
}

// -----

func.func @lower_convert_i32_to_i8(%arg0: tensor<2x4xi32>) -> tensor<2x4xi8> {
  // CHECK: %[[RES:.*]] = arith.trunci %arg0 : tensor<2x4xi32> to tensor<2x4xi8>
  // CHECK: return %[[RES]] : tensor<2x4xi8>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi32>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @lower_convert_i32_to_i1(%arg0: tensor<2x4xi32>) -> tensor<2x4xi1> {
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0> : tensor<2x4xi32> 
  // CHECK: %[[RES:.*]] = arith.cmpi ne, %arg0, %[[ZERO]] : tensor<2x4xi32>
  // CHECK: return %[[RES]] : tensor<2x4xi1>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// -----

func.func @lower_convert_i1_to_f32(%arg0: tensor<2x4xi1>) -> tensor<2x4xf32> {
  // CHECK: %[[RES:.*]] = arith.uitofp %arg0 : tensor<2x4xi1> to tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi1>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @lower_convert_i32_to_f32(%arg0: tensor<2x4xi32>) -> tensor<2x4xf32> {
  // CHECK: %[[RES:.*]] = arith.sitofp %arg0 : tensor<2x4xi32> to tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xf32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xi32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @lower_convert_f32_to_i1(%arg0: tensor<2x4xf32>) -> tensor<2x4xi1> {
  // CHECK: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<2x4xf32>
  // CHECK: %[[RES:.*]] = arith.cmpf une, %arg0, %[[ZERO]] : tensor<2x4xf32>
  // CHECK: return %[[RES]] : tensor<2x4xi1>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf32>) -> tensor<2x4xi1>
  return %0 : tensor<2x4xi1>
}

// -----

func.func @lower_convert_f32_to_i32(%arg0: tensor<2x4xf32>) -> tensor<2x4xi32> {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<2x4xi32>
  // CHECK-DAG: %[[MAX_INT:.*]] = arith.constant dense<2147483647> : tensor<2x4xi32>
  // CHECK-DAG: %[[MAX_FLOAT:.*]] = arith.constant dense<2.14748365E+9> : tensor<2x4xf32>
  // CHECK-DAG: %[[MIN_INT:.*]] = arith.constant dense<-2147483648> : tensor<2x4xi32>
  // CHECK-DAG: %[[MIN_FLOAT:.*]] = arith.constant dense<-2.14748365E+9> : tensor<2x4xf32>
  // CHECK: %[[FP2SI:.*]] = arith.fptosi %arg0 : tensor<2x4xf32> to tensor<2x4xi32>
  // CHECK: %[[CMP_MIN:.*]] = arith.cmpf ole, %arg0, %[[MIN_FLOAT]] : tensor<2x4xf32>
  // CHECK: %[[CLAMPED_MIN:.*]] = arith.select %[[CMP_MIN]], %[[MIN_INT]], %[[FP2SI]] : tensor<2x4xi1>, tensor<2x4xi32>
  // CHECK: %[[CMP_MAX:.*]] = arith.cmpf oge, %arg0, %[[MAX_FLOAT]] : tensor<2x4xf32>
  // CHECK: %[[CLAMPED_MAX:.*]] = arith.select %[[CMP_MAX]], %[[MAX_INT]], %[[CLAMPED_MIN]] : tensor<2x4xi1>, tensor<2x4xi32>
  // CHECK: %[[NAN_CMP:.*]] = arith.cmpf uno, %arg0, %arg0 : tensor<2x4xf32>
  // CHECK: %[[RES:.*]] = arith.select %[[NAN_CMP]], %[[ZERO]], %[[CLAMPED_MAX]] : tensor<2x4xi1>, tensor<2x4xi32>
  // CHECK: return %[[RES]] : tensor<2x4xi32>
  %0 = stablehlo.convert %arg0 : (tensor<2x4xf32>) -> tensor<2x4xi32>
  return %0 : tensor<2x4xi32>
}
