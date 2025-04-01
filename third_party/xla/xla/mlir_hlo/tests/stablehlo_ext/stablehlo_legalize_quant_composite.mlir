// RUN: mlir-hlo-opt --stablehlo-ext-legalize-quant-composite --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @fake_quant
func.func @fake_quant(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32:1, {0.0016193275805562735,0.0016197443474084139}>>
  // CHECK: %[[TEMP_1:.*]] = stablehlo.uniform_dequantize %[[TEMP_0]] : (tensor<2x2x!quant.uniform<i8:f32:1, {0.0016193275805562735,0.0016197443474084139}>>) -> tensor<2x2xf32>
    // CHECK: return %[[TEMP_1]] : tensor<2x2xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.fake_quant.impl} : (tensor<2x2xf32>) -> tensor<2x2xf32>

  return %0 : tensor<2x2xf32>
}

func.func private @quant.fake_quant.impl(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  return %arg0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @quantize_feed_to_return
func.func @quantize_feed_to_return(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32:1, {0.0016193275805562735,0.0016197443474084139}>>
  // CHECK: return %[[TEMP_2:.*]]
  %0 = stablehlo.composite "quant.quantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.quant.impl} : (tensor<2x2xf32>) -> tensor<2x2xi8>
  return %0 : tensor<2x2xi8>
}


func.func private @quant.quant.impl(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf32>) ->  tensor<2x2xi8>
  return %0 : tensor<2x2xi8>
}

// -----

// CHECK-LABEL: func @args_feeding_dequantize
// CHECK-SAME:      arg0: tensor<2x2x!quant.uniform
func.func @args_feeding_dequantize(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  // CHECK:         %[[TEMP_0:.*]] = stablehlo.uniform_dequantize %arg0
  // CHECK-NEXT:    return %[[TEMP_1:.*]] : tensor<2x2xf32>
  %0 = stablehlo.composite "quant.dequantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.dequant.impl} : (tensor<2x2xi8>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}


func.func private @quant.dequant.impl(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi8>) ->  tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @quantize_feeding_dequantize
func.func @quantize_feeding_dequantize(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.uniform_quantize %arg0 : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform
  // CHECK: %[[TEMP_1:.*]] = stablehlo.uniform_dequantize %[[TEMP_0]]
  // CHECK: return %[[TEMP_1]] : tensor<2x2xf32>
  %0 = stablehlo.composite "quant.quantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.quant.impl} : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %1 = stablehlo.composite "quant.dequantize" %0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.dequant.impl} : (tensor<2x2xi8>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}


func.func private @quant.quant.impl(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf32>) ->  tensor<2x2xi8>
  return %0 : tensor<2x2xi8>
}

func.func private @quant.dequant.impl(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi8>) ->  tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @quantize_feeding_add
func.func @quantize_feeding_add(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.composite "quant.quantize" %arg0
  // CHECK-NEXT: %[[TEMP_1:.*]] = stablehlo.add %[[TEMP_0]], %[[TEMP_0]]
  // CHECK-NEXT: return %[[TEMP_1]]
  %0 = stablehlo.composite "quant.quantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.00161974435]> : tensor<2xf32>, storage_type_max = 127 : i64, storage_type_min = -128 : i64, zero_point = dense<0> : tensor<2xi64>}, decomposition = @quant.quant.impl} : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %1 = stablehlo.add %0, %0 : tensor<2x2xi8>
    return %1 : tensor<2x2xi8>
}

func.func private @quant.quant.impl(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  %0 = stablehlo.convert %arg0 : (tensor<2x2xf32>) -> tensor<2x2xi8>
  return %0 : tensor<2x2xi8>
}

// -----

// CHECK-LABEL: func @add_feading_dequantize
func.func @add_feading_dequantize(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.add %arg0, %arg0
  // CHECK-NEXT: %[[TEMP_1:.*]] = stablehlo.composite "quant.dequantize" %[[TEMP_0]]
  // CHECK-NEXT: return %[[TEMP_1]]
  %0 = stablehlo.add %arg0, %arg0 : tensor<2x2xi8>
  %1 = stablehlo.composite "quant.dequantize" %0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.dequant.impl} : (tensor<2x2xi8>) -> tensor<2x2xf32>
  return %1 : tensor<2x2xf32>
}


func.func private @quant.dequant.impl(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi8>) ->  tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @quantize_feeding_dequantize_and_return
func.func @quantize_feeding_dequantize_and_return(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32> , tensor<2x2xi8>) {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.uniform_quantize %arg0
  // CHECK: %[[TEMP_1:.*]] = stablehlo.uniform_dequantize %[[TEMP_0]]
  // CHECK: return %[[TEMP_1]], %[[TEMP_0]] : tensor<2x2xf32>, tensor<2x2x!quant.uniform
  %0 = stablehlo.composite "quant.quantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.quant.impl} : (tensor<2x2xf32>) -> tensor<2x2xi8>
  %1 = stablehlo.composite "quant.dequantize" %0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.dequant.impl} : (tensor<2x2xi8>) -> tensor<2x2xf32>
  return %1, %0 : tensor<2x2xf32> , tensor<2x2xi8>
 }


func.func private @quant.quant.impl(%arg0: tensor<2x2xf32>) -> tensor<2x2xi8> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xf32>) ->  tensor<2x2xi8>
  return %0 : tensor<2x2xi8>
}

func.func private @quant.dequant.impl(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi8>) ->  tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @args_feeding_dequantize_and_other
func.func @args_feeding_dequantize_and_other(%arg0: tensor<2x2xi8>) -> (tensor<2x2xf32>, tensor<2x2xi8>) {
  // CHECK: %[[TEMP_0:.*]] = stablehlo.composite "quant.dequantize" %arg0
  // CHECK-NEXT: return %[[TEMP_0]], %arg0 : tensor<2x2xf32>, tensor<2x2xi8>
  %0 = stablehlo.composite "quant.dequantize" %arg0 {composite_attributes = {dtype = i8, quantization_dimension = 1 : i32, scale = dense<[0.00161932758, 0.0016197443]> : tensor<2xf32>, zero_point = dense<0> : tensor<2xi64>, storage_type_min=-128, storage_type_max=127}, decomposition = @quant.dequant.impl} : (tensor<2x2xi8>) -> tensor<2x2xf32>
  return %0, %arg0 : tensor<2x2xf32>, tensor<2x2xi8>
}


func.func private @quant.dequant.impl(%arg0: tensor<2x2xi8>) -> tensor<2x2xf32> {
  %0 = "stablehlo.convert"(%arg0) : (tensor<2x2xi8>) ->  tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
