// RUN: tac-opt-all-backends -tfl-pick-subgraphs %s -split-input-file -verify-diagnostics | FileCheck %s

module {
  func.func @main(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>, %arg3: tensor<100xf32>) -> tensor<2x100xf32> {
    %0 = func.call @func_0_GPU_FLOAT(%arg0, %arg1, %arg2) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
    %1 = func.call @func_1_GPU_FLOAT(%arg0, %arg3) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
    %2 = func.call @func_2_CPU_FLOAT(%0, %1) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<2x100xf32>
    func.return %2 : tensor<2x100xf32>
  }
  func.func @func_2_CPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<2x100xf32> attributes {tac.cost = 2.000000e+01 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", values_count = 2 : i32} : (tensor<100xf32>, tensor<100xf32>) -> tensor<2x100xf32>
    func.return %0 : tensor<2x100xf32>
  }
  func.func @func_0_GPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>) -> tensor<100xf32> attributes {tac.cost = 4.000000e+01 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    %1 = tfl.mul %0, %arg2 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    func.return %1 : tensor<100xf32>
  }
  func.func @func_1_GPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> attributes {tac.cost = 2.000000e+01 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    func.return %0 : tensor<100xf32>
  }
  func.func @func_2_GPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<2x100xf32> attributes {tac.cost = 8.040000e+01 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %cst = arith.constant dense<[2, 100]> : tensor<2xi64>
    %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<200xf32>
    %1 = "tfl.reshape"(%0, %cst) : (tensor<200xf32>, tensor<2xi64>) -> tensor<2x100xf32>
    func.return %1 : tensor<2x100xf32>
  }
  func.func @func_0_CPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>, %arg2: tensor<100xf32>) -> tensor<100xf32> attributes {tac.cost = 2.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    %1 = tfl.mul %0, %arg2 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    func.return %1 : tensor<100xf32>
  }
  func.func @func_1_CPU_FLOAT(%arg0: tensor<100xf32>, %arg1: tensor<100xf32>) -> tensor<100xf32> attributes {tac.cost = 1.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "GPU"} : tensor<100xf32>
    func.return %0 : tensor<100xf32>
  }

// CHECK:       func @main([[VAL_0:%.*]]: tensor<100xf32>, [[VAL_1:%.*]]: tensor<100xf32>, [[VAL_2:%.*]]: tensor<100xf32>, [[VAL_3:%.*]]: tensor<100xf32>) -> tensor<2x100xf32> {
// CHECK:           [[VAL_4:%.*]] = call @func_0_GPU_FLOAT([[VAL_0]], [[VAL_1]], [[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<100xf32>, tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
// CHECK:           [[VAL_5:%.*]] = call @func_1_GPU_FLOAT([[VAL_0]], [[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<100xf32>
// CHECK:           [[VAL_6:%.*]] = call @func_2_GPU_FLOAT([[VAL_4]], [[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<100xf32>, tensor<100xf32>) -> tensor<2x100xf32>
// CHECK:           return [[VAL_6]] : tensor<2x100xf32>
// CHECK:         }
}

// -----

module {
  func.func @main(%arg0: tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32> attributes {tf.entry_function = {inputs = "Placeholder", outputs = "mul_1"}} {
    %0 = "tfl.pseudo_const"() {value = dense<0.962260901> : tensor<1xf32>} : () -> tensor<1xf32>
    %1 = func.call @func_0_GPU_FLOAT(%arg0, %0) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    %2 = "tfl.pseudo_const"() {value = dense<0.895973444> : tensor<1xf32>} : () -> tensor<1xf32>
    %3 = func.call @func_1_GPU_FLOAT(%arg0, %2) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    %4 = func.call @func_2_CPU_FLOAT(%3, %1) {tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<1x200x200x200xf32>, tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32>
    %5 = "tfl.pseudo_const"() {value = dense<0.0778453499> : tensor<1xf32>} : () -> tensor<1xf32>
    %6 = func.call @func_3_GPU_FLOAT(%4, %5) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} : (tensor<2x1x200x200x200xf32>, tensor<1xf32>) -> tensor<2x1x200x200x200xf32>
    func.return %6 : tensor<2x1x200x200x200xf32>
  }
  func.func @func_2_CPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32> attributes {tac.cost = 1.600000e+06 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", values_count = 2 : i32} : (tensor<1x200x200x200xf32>, tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32>
    func.return %0 : tensor<2x1x200x200x200xf32>
  }
  func.func @func_0_GPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<1x200x200x200xf32> attributes {tac.cost = 1.600000e+06 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    func.return %0 : tensor<1x200x200x200xf32>
  }
  func.func @func_1_GPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<1x200x200x200xf32> attributes {tac.cost = 1.600000e+06 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    func.return %0 : tensor<1x200x200x200xf32>
  }
  func.func @func_3_GPU_FLOAT(%arg0: tensor<2x1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<2x1x200x200x200xf32> attributes {tac.cost = 3.200000e+06 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} {
    %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<2x1x200x200x200xf32>, tensor<1xf32>) -> tensor<2x1x200x200x200xf32>
    func.return %0 : tensor<2x1x200x200x200xf32>
  }
  func.func @func_2_GPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32> attributes {tac.cost = 0x4AC35002 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %cst = arith.constant dense<[2, 1, 200, 200, 200]> : tensor<5xi64>
    %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 0 : i32, fused_activation_function = "NONE"} : (tensor<1x200x200x200xf32>, tensor<1x200x200x200xf32>) -> tensor<2x200x200x200xf32>
    %1 = "tfl.reshape"(%0, %cst) : (tensor<2x200x200x200xf32>, tensor<5xi64>) -> tensor<2x1x200x200x200xf32>
    func.return %1 : tensor<2x1x200x200x200xf32>
  }
  func.func @func_0_CPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<1x200x200x200xf32> attributes {tac.cost = 8.000000e+06 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    func.return %0 : tensor<1x200x200x200xf32>
  }
  func.func @func_1_CPU_FLOAT(%arg0: tensor<1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<1x200x200x200xf32> attributes {tac.cost = 8.000000e+06 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name= "func_1"} {
    %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
    func.return %0 : tensor<1x200x200x200xf32>
  }
  func.func @func_3_CPU_FLOAT(%arg0: tensor<2x1x200x200x200xf32>, %arg1: tensor<1xf32>) -> tensor<2x1x200x200x200xf32> attributes {tac.cost = 1.600000e+07 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} {
    %0 = "tfl.mul"(%arg0, %arg1) {fused_activation_function = "NONE", tac.device = "GPU"} : (tensor<2x1x200x200x200xf32>, tensor<1xf32>) -> tensor<2x1x200x200x200xf32>
    func.return %0 : tensor<2x1x200x200x200xf32>
  }

// CHECK:       func @main([[VAL_0:%.*]]: tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32> attributes {tf.entry_function = {inputs = "Placeholder", outputs = "mul_1"}} {
// CHECK:           [[VAL_1:%.*]] = "tfl.pseudo_const"() <{value = dense<0.962260901> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAL_2:%.*]] = call @func_0_GPU_FLOAT([[VAL_0]], [[VAL_1]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
// CHECK:           [[VAL_3:%.*]] = "tfl.pseudo_const"() <{value = dense<0.895973444> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAL_4:%.*]] = call @func_1_GPU_FLOAT([[VAL_0]], [[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<1x200x200x200xf32>, tensor<1xf32>) -> tensor<1x200x200x200xf32>
// CHECK:           [[VAL_5:%.*]] = call @func_2_GPU_FLOAT([[VAL_4]], [[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<1x200x200x200xf32>, tensor<1x200x200x200xf32>) -> tensor<2x1x200x200x200xf32>
// CHECK:           [[VAL_6:%.*]] = "tfl.pseudo_const"() <{value = dense<0.0778453499> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK:           [[VAL_7:%.*]] = call @func_3_GPU_FLOAT([[VAL_5]], [[VAL_6]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_3"} : (tensor<2x1x200x200x200xf32>, tensor<1xf32>) -> tensor<2x1x200x200x200xf32>
// CHECK:           return [[VAL_7]] : tensor<2x1x200x200x200xf32>
// CHECK:         }
}

// -----

module {
  func.func @main(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg2: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg3: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tf.entry_function = {inputs = "input0,input1,input2,input3", outputs = "output"}} {
    %0 = func.call @func_0_CPU_QUANTIZED_INT8(%arg0, %arg1, %arg2) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    %1 = func.call @func_1_CPU_QUANTIZED_INT8(%arg0, %arg3) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_1"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    %2 = func.call @func_2_CPU_QUANTIZED_INT8(%0, %1) {tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %2 : tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_0_CPU_QUANTIZED_INT8(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg2: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 2.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_0"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    %1 = tfl.mul %0, %arg2 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %1 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_2_CPU_QUANTIZED_INT8(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 1.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_2"} {
    %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", values_count = 2 : i32} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %0 : tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_1_CPU_QUANTIZED_INT8(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 1.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8", tac.interface_name = "func_1"} {
    %0 = tfl.add %arg0, %arg1 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "QUANTIZED_INT8"} : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %0 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_0_GPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg2: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 4.000000e+01 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = "tfl.dequantize"(%arg0) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = tfl.add %0, %1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %3 = "tfl.dequantize"(%arg2) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %4 = tfl.mul %2, %3 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %5 = "tfl.quantize"(%4) {qtype = tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %5 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_0_CPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg2: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 2.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} {
    %0 = "tfl.dequantize"(%arg0) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = tfl.add %0, %1 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %3 = "tfl.dequantize"(%arg2) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %4 = tfl.mul %2, %3 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %5 = "tfl.quantize"(%4) {qtype = tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %5 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_2_GPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 162.200012 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %cst = arith.constant dense<[1, 1, 1, 100]> : tensor<4xi32>
    %cst_0 = arith.constant dense<200> : tensor<1xi32>
    %cst_1 = arith.constant dense<[2, 100]> : tensor<2xi32>
    %0 = "tfl.dequantize"(%arg0) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = "tfl.reshape"(%0, %cst) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>, tensor<4xi32>) -> tensor<1x1x1x100xf32>
    %3 = "tfl.reshape"(%1, %cst) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>, tensor<4xi32>) -> tensor<1x1x1x100xf32>
    %4 = "tfl.concatenation"(%2, %3) {axis = 3 : i32, fused_activation_function = "NONE", tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x1x100xf32>, tensor<1x1x1x100xf32>) -> tensor<1x1x1x200xf32>
    %5 = "tfl.reshape"(%4, %cst_0) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<1x1x1x200xf32>, tensor<1xi32>) -> tensor<200xf32>
    %6 = "tfl.reshape"(%5, %cst_1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<200xf32>, tensor<2xi32>) -> tensor<2x100xf32>
    %7 = "tfl.quantize"(%6) {qtype = tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<2x100xf32>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %7 : tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_2_CPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 1.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} {
    %0 = "tfl.dequantize"(%arg0) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = "tfl.pack"(%0, %1) {axis = 0 : i32, tac.device = "CPU", tac.inference_type = "FLOAT", values_count = 2 : i32} : (tensor<100xf32>, tensor<100xf32>) -> tensor<2x100xf32>
    %3 = "tfl.quantize"(%2) {qtype = tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<2x100xf32>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %3 : tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_1_GPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 2.000000e+01 : f32, tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = "tfl.dequantize"(%arg0) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = tfl.add %0, %1 {fused_activation_function = "RELU6", tac.device = "GPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %3 = "tfl.quantize"(%2) {qtype = tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "GPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %3 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }
  func.func private @func_1_CPU_FLOAT(%arg0: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %arg1: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tac.cost = 1.000000e+02 : f32, tac.device = "CPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} {
    %0 = "tfl.dequantize"(%arg0) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %1 = "tfl.dequantize"(%arg1) {tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100xf32>
    %2 = tfl.add %0, %1 {fused_activation_function = "RELU6", tac.device = "CPU", tac.inference_type = "FLOAT"} : tensor<100xf32>
    %3 = "tfl.quantize"(%2) {qtype = tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tac.device = "CPU", tac.inference_type = "FLOAT"} : (tensor<100xf32>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
    func.return %3 : tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
  }

// CHECK: @main(%[[VAL_0:.*]]: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %[[VAL_1:.*]]: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %[[VAL_2:.*]]: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, %[[VAL_3:.*]]: tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>> attributes {tf.entry_function = {inputs = "input0,input1,input2,input3", outputs = "output"}} {
// CHECK:           %[[VAL_4:.*]] = call @func_0_GPU_FLOAT(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_0"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
// CHECK:           %[[VAL_5:.*]] = call @func_1_GPU_FLOAT(%[[VAL_0]], %[[VAL_3]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_1"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
// CHECK:           %[[VAL_6:.*]] = call @func_2_GPU_FLOAT(%[[VAL_4]], %[[VAL_5]]) {tac.device = "GPU", tac.inference_type = "FLOAT", tac.interface_name = "func_2"} : (tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>, tensor<100x!quant.uniform<i8:f32, 2.000000e-01:-3>>) -> tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
// CHECK:           return %[[VAL_6]] : tensor<2x100x!quant.uniform<i8:f32, 2.000000e-01:-3>>
// CHECK:         }

}
