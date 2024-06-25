// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs="quantization-specs=disable-all-dot-general" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=DISABLE-ALL-DOT-GENERAL

// Tests that `composite_dot_general_fn_1` and its corresponding XlaCallModuleOp
// contains attributes required for quantization, including the
// `_quantization_method` attribute that contains textpb of `Method`.

// DISABLE-ALL-DOT-GENERAL: @main
func.func @main(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}

// DISABLE-ALL-DOT-GENERAL: %[[CONST:.+]] = stablehlo.constant dense<2.000000e+00>
// DISABLE-ALL-DOT-GENERAL: %[[XLA_CALL_MODULE:.+]] = "tf.XlaCallModule"(%arg0, %[[CONST]])

// Check that the `_quantization_method` attribute contains the quantization
// method in textproto format. The dot_general op quantization is explicitly
// disabled by having `_quantization_method = "no_quantization { }"`.
// DISABLE-ALL-DOT-GENERAL-SAME: _entry_function = @composite_dot_general_fn_1
// DISABLE-ALL-DOT-GENERAL-SAME: _original_entry_function
// DISABLE-ALL-DOT-GENERAL-SAME: _quantization_method = "no_quantization { }"
// DISABLE-ALL-DOT-GENERAL-SAME: _tfl_quant_trait = "fully_quantizable"

// DISABLE-ALL-DOT-GENERAL: return %[[XLA_CALL_MODULE:.+]] : tensor<1x1x64xf32>
// DISABLE-ALL-DOT-GENERAL: }

// DISABLE-ALL-DOT-GENERAL-LABEL: private @composite_dot_general_fn_1
// DISABLE-ALL-DOT-GENERAL-SAME: tf_quant.composite_function
// DISABLE-ALL-DOT-GENERAL: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1
// DISABLE-ALL-DOT-GENERAL: return %[[DOT_GENERAL:.+]] : tensor<1x1x64xf32>
// DISABLE-ALL-DOT-GENERAL: }

// -----

// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs="quantization-specs=empty" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=EMPTY

// Tests that `composite_dot_general_fn_1` and its corresponding XlaCallModuleOp
// contains attributes required for quantization. `_quantization_method` is not
// set, as it is implicitly disabled.

// EMPTY: @main
func.func @main(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}

// EMPTY: %[[CONST:.+]] = stablehlo.constant dense<2.000000e+00>
// EMPTY: %[[XLA_CALL_MODULE:.+]] = "tf.XlaCallModule"(%arg0, %[[CONST]])

// Check that the `_quantization_method` attribute doesn't contain the
// quantization method, implying "no_quantization".
// EMPTY-SAME: _entry_function = @composite_dot_general_fn_1
// EMPTY-SAME: _original_entry_function
// EMPTY-NOT: _quantization_method
// EMPTY-SAME: _tfl_quant_trait = "fully_quantizable"

// EMPTY: return %[[XLA_CALL_MODULE:.+]] : tensor<1x1x64xf32>
// EMPTY: }

// EMPTY-LABEL: private @composite_dot_general_fn_1
// EMPTY-SAME: tf_quant.composite_function
// EMPTY: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1
// EMPTY: return %[[DOT_GENERAL:.+]] : tensor<1x1x64xf32>
// EMPTY: }

// -----

// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs="quantization-specs=static-range-ptq-to-all" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=STATIC-RANGE-PTQ-TO-ALL

// STATIC-RANGE-PTQ-TO-ALL: @main
func.func @main(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}
// Tests that `composite_dot_general_fn_1` and its corresponding XlaCallModuleOp
// contains attributes required for quantization, including the
// `_quantization_method` attribute that contains textpb of `Method`.

// STATIC-RANGE-PTQ-TO-ALL: %[[CONST:.+]] = stablehlo.constant dense<2.000000e+00>
// STATIC-RANGE-PTQ-TO-ALL: %[[XLA_CALL_MODULE:.+]] = "tf.XlaCallModule"(%arg0, %[[CONST]])

// Check that the `_quantization_method` attribute contains the quantization
// method in textproto format, enabling static-range PTQ.
// STATIC-RANGE-PTQ-TO-ALL-SAME: _entry_function = @composite_dot_general_fn_1
// STATIC-RANGE-PTQ-TO-ALL-SAME: _original_entry_function
// STATIC-RANGE-PTQ-TO-ALL-SAME: _quantization_method = "static_range_ptq { }"
// STATIC-RANGE-PTQ-TO-ALL-SAME: _tfl_quant_trait = "fully_quantizable"

// STATIC-RANGE-PTQ-TO-ALL: return %[[XLA_CALL_MODULE:.+]] : tensor<1x1x64xf32>
// STATIC-RANGE-PTQ-TO-ALL: }

// STATIC-RANGE-PTQ-TO-ALL-LABEL: private @composite_dot_general_fn_1
// STATIC-RANGE-PTQ-TO-ALL-SAME: tf_quant.composite_function
// STATIC-RANGE-PTQ-TO-ALL: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1
// STATIC-RANGE-PTQ-TO-ALL: return %[[DOT_GENERAL:.+]] : tensor<1x1x64xf32>
// STATIC-RANGE-PTQ-TO-ALL: }

// -----

// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs="quantization-specs=static-range-ptq-to-compute-heavy" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY

// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: @main
func.func @main(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = stablehlo.add %arg0, %arg0 : tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// Tests that `composite_add_fn_1` does not quantize when quantizing
// only compute-heavy ops.

// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: %[[CONST:.+]] = stablehlo.constant dense<2.000000e+00>
// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: %[[XLA_CALL_MODULE:.+]] = "tf.XlaCallModule"(%arg0, %arg0)

// Check that the `_quantization_method` attribute contains the quantization
// method in textproto format, enabling static-range PTQ.
// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: _entry_function = @composite_add_fn_1
// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: _original_entry_function
// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY-NOT: _quantization_method
// STATIC-RANGE-PTQ-TO-COMPUTE-HEAVY: _tfl_quant_trait = "fully_quantizable"

// -----

// RUN: stablehlo-quant-opt %s -stablehlo-test-lift-quantizable-spots-as-functions-with-quantization-specs="quantization-specs=static-range-ptq-to-all" \
// RUN:   -split-input-file | FileCheck %s --check-prefix=STATIC-RANGE-PTQ-TO-ALL

// STATIC-RANGE-PTQ-TO-ALL-LABEL: @some_func
func.func @some_func(%arg0: tensor<1x1x167xf32>) -> tensor<1x1x64xf32> {
  %0 = stablehlo.constant dense<2.000000e+00> : tensor<167x64xf32>
  %1 = stablehlo.dot_general %arg0, %0, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %1 : tensor<1x1x64xf32>
}
// Tests that XlaCallModuleOp in non-main function has attributes set correctly.

// STATIC-RANGE-PTQ-TO-ALL: %[[CONST:.+]] = stablehlo.constant dense<2.000000e+00>
// STATIC-RANGE-PTQ-TO-ALL: %[[XLA_CALL_MODULE:.+]] = "tf.XlaCallModule"(%arg0, %[[CONST]])

// Check that the `_quantization_method` attribute contains the quantization
// method in textproto format, enabling static-range PTQ.
// STATIC-RANGE-PTQ-TO-ALL-SAME: _entry_function = @composite_dot_general_fn_1
// STATIC-RANGE-PTQ-TO-ALL-SAME: _original_entry_function
// STATIC-RANGE-PTQ-TO-ALL-SAME: _quantization_method = "static_range_ptq { }"
// STATIC-RANGE-PTQ-TO-ALL-SAME: _tfl_quant_trait = "fully_quantizable"

// STATIC-RANGE-PTQ-TO-ALL: return %[[XLA_CALL_MODULE:.+]] : tensor<1x1x64xf32>
// STATIC-RANGE-PTQ-TO-ALL: }

// STATIC-RANGE-PTQ-TO-ALL-LABEL: private @composite_dot_general_fn_1
// STATIC-RANGE-PTQ-TO-ALL-SAME: tf_quant.composite_function
// STATIC-RANGE-PTQ-TO-ALL: %[[DOT_GENERAL:.+]] = stablehlo.dot_general %arg0, %arg1
// STATIC-RANGE-PTQ-TO-ALL: return %[[DOT_GENERAL:.+]] : tensor<1x1x64xf32>
// STATIC-RANGE-PTQ-TO-ALL: }
