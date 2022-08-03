// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range -tfl-quantize="enable-dynamic-range-quantization=true" -tfl-post-quantize  | FileCheck %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1" -tfl-quantize="enable-dynamic-range-quantization=true enable-custom-op-weight-only=CustomTestOp=false" -tfl-post-quantize="enable-no-side-effect=CustomTestOp=false" | FileCheck --check-prefix=NotPrune %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1" -tfl-quantize="enable-dynamic-range-quantization=true enable-custom-op-weight-only=CustomTestOp=false" -tfl-post-quantize="enable-no-side-effect=CustomTestOp=true" | FileCheck --check-prefix=NoSideEffect %s
// RUN: tf-opt %s -tfl-prepare-quantize-dynamic-range="enable-custom-op-quantization=CustomTestOp=1" -tfl-quantize="enable-dynamic-range-quantization=true enable-custom-op-weight-only=CustomTestOp=true"  -tfl-post-quantize="enable-no-side-effect=CustomTestOp=true" | FileCheck --check-prefix=NoSideEffectWeightOnly %s

// CHECK-LABEL: PruneUnusedCustomOp
func.func @PruneUnusedCustomOp(%arg0: tensor<1x1x1x1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %q_w = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<1024x1x1x1xi8>} : () -> tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
  %dq_w = "tfl.dequantize"(%q_w) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
  %custom_1 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  %custom_2 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  %custom_3 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom_3 : tensor<*xf32>

// CHECK: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<1024x1x1x1xi8>} : () -> tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
// CHECK: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
// CHECK: return %[[custom_3:.*]]
}

// CHECK-LABEL: NotPruneUnusedCustomOp
func.func @NotPruneUnusedCustomOp(%arg0: tensor<1x1x1x1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %q_w = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<1024x1x1x1xi8>} : () -> tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
  %dq_w = "tfl.dequantize"(%q_w) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
  %custom_1 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  %custom_2 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  %custom_3 = "tfl.custom"(%arg0, %dq_w) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom_3 : tensor<*xf32>

// CHECK: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>, value = dense<127> : tensor<1024x1x1x1xi8>} : () -> tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CHECK: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
// CHECK: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
// CHECK: return %[[custom_3:.*]]
}

// CHECK-LABEL: PruneQuantizedCustomOp
// NotPrune-LABEL: PruneQuantizedCustomOp
// NoSideEffect-LABEL: PruneQuantizedCustomOp
// NoSideEffectWeightOnly-LABEL: PruneQuantizedCustomOp
func.func @PruneQuantizedCustomOp(%arg0: tensor<1x1x1x1xf32>) -> tensor<*xf32> attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %w = arith.constant dense<127.0> : tensor<1024x1x1x1xf32>
  %custom = "tfl.custom"(%0, %w) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<1024x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom : tensor<*xf32>

// CHECK: %[[w:.*]] = arith.constant dense<1.270000e+02> : tensor<1024x1x1x1xf32>
// CHECK: %[[custom:.*]] = "tfl.custom"(%arg0, %[[w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}
// CHECK: return %[[custom:.*]]

// NotPrune: %[[w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// NotPrune: %[[dq_w:.*]] = "tfl.dequantize"(%[[w:.*]]) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
// NotPrune: %[[custom:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}
// NotPrune: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}
// NotPrune: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}

// NoSideEffect: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// NoSideEffect: %[[custom:.*]] = "tfl.custom"(%arg0, %[[q_w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}
// NoSideEffect: return %[[custom:.*]]

// NoSideEffectWeightOnly: %[[q_w:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// NoSideEffectWeightOnly: %[[dq_w:.*]] = "tfl.dequantize"(%[[q_w:.*]]) : (tensor<1024x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<1024x1x1x1xf32>
// NoSideEffectWeightOnly: %[[custom:.*]] = "tfl.custom"(%arg0, %[[dq_w:.*]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>}
// NoSideEffectWeightOnly: return %[[custom:.*]]
}

// CHECK-LABEL: QuantizeCustomOp
// CustomOp-LABEL: QuantizeCustomOp
func.func @QuantizeCustomOp(%arg0: tensor<1x1x1x1xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) attributes {tf.entry_function = {inputs = "input", outputs = "custom_op"}} {
  %0 = "quantfork.stats"(%arg0) {layerStats = dense<[0.000000e+00, 2.550000e+02]> : tensor<2xf32>} : (tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
  %w_1 = arith.constant dense<127.0> : tensor<4096x1x1x1xf32>
  %w_2 = arith.constant dense<127.0> : tensor<128x1x1x1xf32>
  %b = arith.constant dense<127.0> : tensor<2048x1x1x1xf32>
  %custom_1 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  %custom_2 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  %custom_3 = "tfl.custom"(%0, %w_1, %w_2, %b) {custom_code = "CustomTestOp3", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
  func.return %custom_1, %custom_2, %custom_3 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

// CHECK: %[[w_1:.*]] = arith.constant dense<1.270000e+02> : tensor<4096x1x1x1xf32>
// CHECK: %[[w_2:.*]] = arith.constant dense<1.270000e+02> : tensor<128x1x1x1xf32>
// CHECK: %[[b:.*]] = arith.constant dense<1.270000e+02> : tensor<2048x1x1x1xf32>
// CHECK: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp3", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CHECK: return %[[custom_1:.*]], %[[custom_2:.*]], %[[custom_3:.*]]

// CustomOpWeightOnly: %[[w_1:.*]] = arith.constant dense<1.270000e+02> : tensor<4096x1x1x1xf32>
// CustomOpWeightOnly: %[[q_w1:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOpWeightOnly: %[[dq_w1:.*]] = "tfl.dequantize"(%[[q_w1]]) : (tensor<4096x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<4096x1x1x1xf32>
// CustomOpWeightOnly: %[[w_2:.*]] = arith.constant dense<1.270000e+02> : tensor<128x1x1x1xf32>
// CustomOpWeightOnly: %[[q_b:.*]] = "tfl.pseudo_qconst"() {qtype = tensor<2048x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>
// CustomOpWeightOnly: %[[dq_b:.*]] = "tfl.dequantize"(%[[q_b]]) : (tensor<2048x1x1x1x!quant.uniform<i8<-127:127>:f32, 1.000000e+00>>) -> tensor<2048x1x1x1xf32>
// CustomOpWeightOnly: %[[custom_1:.*]] = "tfl.custom"(%arg0, %[[dq_w1]], %[[w_2]], %[[dq_b]]) {custom_code = "CustomTestOp", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOpWeightOnly: %[[custom_2:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[b]]) {custom_code = "CustomTestOp2", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOpWeightOnly: %[[custom_3:.*]] = "tfl.custom"(%arg0, %[[w_1]], %[[w_2]], %[[q_b]]) {custom_code = "CustomTestOp3", custom_option = opaque<"tfl", "0x"> : tensor<0xi8>} : (tensor<1x1x1x1xf32>, tensor<4096x1x1x1xf32>, tensor<128x1x1x1xf32>, tensor<2048x1x1x1xf32>) -> tensor<*xf32>
// CustomOpWeightOnly: return %[[custom_1:.*]], %[[custom_2:.*]], %[[custom_3:.*]]
}
