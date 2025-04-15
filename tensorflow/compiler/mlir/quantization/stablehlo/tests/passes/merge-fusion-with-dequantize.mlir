// RUN: stablehlo-quant-opt %s -stablehlo-merge-fusion-with-dequantize -split-input-file -verify-diagnostics | FileCheck %s

// Merge fusion with dequantize for relu case.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @merge_relu_fusion
  func.func private @merge_relu_fusion(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_relu_fn
    // CHECK-SAME: -> tensor<1x3xf32>
    %2 = call @quantized_dot_general_relu_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_relu_fn
  func.func private @quantized_dot_general_relu_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    // CHECK: %[[MIN:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1
    // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
    // CHECK: %[[MAX:.*]] = chlo.broadcast_maximum %[[DQ]], %[[MIN]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %1 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}

// -----

// Merge fusion with dequantize for relu6 case.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @merge_relu6_fusion
  func.func private @merge_relu6_fusion(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_relu6_fn
    // CHECK-SAME: -> tensor<1x3xf32>
    %2 = call @quantized_dot_general_relu6_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_relu6_fn
  func.func private @quantized_dot_general_relu6_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    // CHECK-DAG: %[[MIN:.*]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: %[[MAX:.*]] = stablehlo.constant dense<6.000000e+00> : tensor<f32>
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1
    // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
    // CHECK: %[[CLAMP:.*]] = stablehlo.clamp %[[MIN]], %[[DQ]], %[[MAX]]
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %1 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}

// -----

// Merge fusion with dequantize for no activation case.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @merge_no_act_fusion
  func.func private @merge_no_act_fusion(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_fn
    // CHECK-SAME: -> tensor<1x3xf32>
    %2 = call @quantized_dot_general_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_fn
  func.func private @quantized_dot_general_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    // CHECK: %[[DOT:.*]] = stablehlo.dot_general %arg0, %arg1
    // CHECK: %[[DQ:.*]] = stablehlo.uniform_dequantize %[[DOT]]
    // CHECK: return %[[DQ]] : tensor<1x3xf32>
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %1 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}

// -----

// Do not merge when quant.uniform result is used directly.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @no_merge_fusion_direct_usage
  func.func private @no_merge_fusion_direct_usage(%arg0: tensor<1x4xf32>) -> (tensor<1x3xf32>, tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_relu_fn
    // CHECK-SAME: -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %2 = call @quantized_dot_general_relu_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3, %2 : tensor<1x3xf32>, tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_relu_fn
  func.func private @quantized_dot_general_relu_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %1 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}

// -----

// Do not merge when fusion and dequantize is already merged.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @no_merge_fusion_already_merged
  func.func private @no_merge_fusion_already_merged(%arg0: tensor<1x4xf32>) -> (tensor<1x3xf32>) {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_fn
    // CHECK-SAME: -> tensor<1x3xf32>
    %2 = call @quantized_dot_general_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3xf32>
    return %2 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_fn
  func.func private @quantized_dot_general_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3xf32> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_dequantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3xf32>
    return %1 : tensor<1x3xf32>
  }
}

// -----

// Do not merge when function is not quantized function.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @merge_relu_fusion
  func.func private @merge_relu_fusion(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @some_func
    // CHECK-SAME: -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %2 = call @some_func(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @some_func
  func.func private @some_func(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>
    %1 = stablehlo.uniform_quantize %0 : (tensor<1x3x!quant.uniform<i32:f32:1, {3.000000e-05,3.000000e-05,3.000000e-05}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %1 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}

// -----

// Do not merge when the quantized fusion is invalid.

module attributes {tf_saved_model.semantics} {
  // CHECK-LABEL: func.func private @merge_relu_fusion
  func.func private @merge_relu_fusion(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
    %0 = stablehlo.constant() {value = dense<127> : tensor<4x3xi8>} : () -> tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    %1 = stablehlo.uniform_quantize %arg0 : (tensor<1x4xf32>) -> tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>
    // CHECK: call @quantized_dot_general_relu_fn
    // CHECK-SAME: -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %2 = call @quantized_dot_general_relu_fn(%1, %0) : (tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>, tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    %3 = stablehlo.uniform_dequantize %2 : (tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>) -> tensor<1x3xf32>
    return %3 : tensor<1x3xf32>
  }

  // CHECK-LABEL: func.func private @quantized_dot_general_relu_fn
  func.func private @quantized_dot_general_relu_fn(
      %arg0: tensor<1x4x!quant.uniform<i8:f32, 6.000000e-03:-128>>,
      %arg1: tensor<4x3x!quant.uniform<i8<-127:127>:f32:1, {5.000000e-03,5.000000e-03,5.000000e-03}>>
    ) -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>> attributes {_from_xla_call_module} {
    %0 = stablehlo.constant() {value = dense<2> : tensor<1x3xi8>} : () -> tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
    return %0 : tensor<1x3x!quant.uniform<i8:f32, 1.000000e-03:-3>>
  }
}
