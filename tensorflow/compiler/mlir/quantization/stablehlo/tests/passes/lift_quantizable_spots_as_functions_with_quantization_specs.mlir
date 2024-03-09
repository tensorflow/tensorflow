// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs \
// RUN:   -split-input-file | FileCheck %s

// CHECK: @main
func.func @main(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}
// Tests that `composite_dot_general_fn_1` and its corresponding XlaCallModuleOp
// is missing attributes required for quantization.

// CHECK: %[[CONST:.*]] = stablehlo.constant dense<2.000000e+00>
// CHECK: %[[XLA_CALL_MODULE:.*]] = "tf.XlaCallModule"(%arg0, %[[CONST]])

// Check that the `_quantization_method` attribute contains the quantization
// method in textproto format. Also check that it doesn't contain attributes
// required for quantization.
// CHECK-SAME: _entry_function = @composite_dot_general_fn_1
// CHECK-SAME: _quantization_method = "no_quantization {}"
// CHECK-NOT: _original_entry_function
// CHECK-NOT: _tfl_quant_trait

// CHECK: return %[[XLA_CALL_MODULE:.*]] : tensor<1x1x64xf32>
// CHECK: }

// CHECK-LABEL: private @composite_dot_general_fn_1
// CHECK-NOT: tf_quant.composite_function
// CHECK: %[[DOT_GENERAL:.*]] = stablehlo.dot_general %arg0, %arg1
// CHECK: return %[[DOT_GENERAL:.*]] : tensor<1x1x64xf32>
// CHECK: }
