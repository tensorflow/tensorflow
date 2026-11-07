// RUN: litert-opt %s -tfl-mark-optimization-regions -split-input-file | FileCheck %s --check-prefix=MARK
// RUN: litert-opt %s -tfl-mark-optimization-regions -tfl-legalize-hlo -split-input-file | FileCheck %s --check-prefix=LEGALIZE
// RUN: litert-opt %s -tfl-mark-optimization-regions -tfl-legalize-hlo -tfl-skip-optimization-regions='pipeline=canonicalize,cse,tfl-optimize,tfl-optimize-batch-matmul,tfl-quantize{qdq-conversion-mode=Strict}' -tfl-materialize-optimization-regions -split-input-file | FileCheck %s --check-prefix=MATERIALIZE
// RUN: litert-opt %s -tfl-mark-optimization-regions -tfl-legalize-hlo -tfl-skip-optimization-regions='pipeline=canonicalize,cse,tfl-optimize,tfl-optimize-batch-matmul,tfl-quantize{qdq-conversion-mode=Strict}' -tfl-materialize-optimization-regions -tfl-skip-optimization-regions=pipeline=canonicalize -tfl-inline-optimization-regions -split-input-file | FileCheck %s --check-prefix=END
// RUN: not litert-opt %s -tfl-mark-optimization-regions -tfl-legalize-hlo -tfl-inline-optimization-regions -split-input-file 2>&1 | FileCheck %s --check-prefix=RAW

// The contract of `optimize = "none"`: the `tfl.*` ops a frontend emitted on
// the custom_call carrier are the ops in the result, untouched by
// legalization-time patterns, canonicalization, CSE or Optimize.
//
// Both regions below contain the same body: reshape -> reshape, which the
// canonicalizer folds into one reshape, and batch_matmul(const) -> add, which
// OptimizeBatchMatmul turns into a fully_connected (the add stays separate
// here; the converter's next Optimize round would fuse it as bias). The
// ordinary region gets both rewrites; the as-emitted one gets neither.
//
// Inlining without materialization is an error: a carrier must never reach
// the exporter as a raw custom_call.
// RAW: error: odml.optimization_region: carrier was not materialized before inlining

// MARK-LABEL: func.func private @asis_body
// MARK-SAME: attributes {odml.optimization = {optimize = "none"}}
// MARK-LABEL: func.func private @ordinary_body
// MARK-SAME: attributes {odml.optimization = {}}

// LEGALIZE-LABEL: func.func private @asis_body
// LEGALIZE: mhlo.custom_call @tfl.reshape
// LEGALIZE: mhlo.custom_call @tfl.reshape
// LEGALIZE: mhlo.custom_call @tfl.batch_matmul
// LEGALIZE: mhlo.custom_call @tfl.add
// LEGALIZE-LABEL: func.func private @ordinary_body
// LEGALIZE: "tfl.reshape"
// LEGALIZE: "tfl.reshape"
// LEGALIZE: "tfl.batch_matmul"
// LEGALIZE: tfl.add

// MATERIALIZE-LABEL: func.func private @asis_body
// MATERIALIZE-SAME: attributes {odml.optimization = {optimize = "none"}}
// MATERIALIZE: %[[R0:.*]] = "tfl.reshape"(%arg0
// MATERIALIZE-NEXT: %[[R1:.*]] = "tfl.reshape"(%[[R0]]
// MATERIALIZE-NEXT: %[[MM:.*]] = "tfl.batch_matmul"(%[[R1]]
// MATERIALIZE-NEXT: %[[ADD:.*]] = tfl.add(%[[MM]]
// MATERIALIZE-NEXT: return %[[ADD]]
// MATERIALIZE-LABEL: func.func private @ordinary_body
// MATERIALIZE-NOT: tfl.reshape
// MATERIALIZE-NOT: tfl.batch_matmul
// MATERIALIZE: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %{{.*}}, %{{.*}}) <{fused_activation_function = "NONE", keep_num_dims = true
// MATERIALIZE-NEXT: %[[ADD:.*]] = tfl.add(%[[FC]]
// MATERIALIZE-NEXT: return %[[ADD]]

// After materialization the only rewriting passes left are wrapped in
// -tfl-skip-optimization-regions (here: canonicalize, which would fold the
// reshape pair and drop the explicit zero FC bias in the as-emitted bodies
// too if it ran there); the regions are then inlined and their functions
// dropped, so the exported model has no subgraph boundary.
// END-LABEL: func.func @main
// END-NOT: stablehlo.composite
// END: %[[R0:.*]] = "tfl.reshape"(%arg0
// END-NEXT: %[[R1:.*]] = "tfl.reshape"(%[[R0]]
// END-NEXT: %[[MM:.*]] = "tfl.batch_matmul"(%[[R1]]
// END-NEXT: %[[ADD:.*]] = tfl.add(%[[MM]]
// END-NOT: tfl.reshape
// END: %[[FC:.*]] = "tfl.fully_connected"(%arg0
// END-NEXT: %[[ADD2:.*]] = tfl.add(%[[FC]]
// END-NEXT: return %[[ADD]], %[[ADD2]]
// END-NOT: func.func private
func.func @main(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {
    composite_attributes = {optimize = "none"}, decomposition = @asis_body
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = stablehlo.composite "odml.optimization_region" %arg0 {
    decomposition = @ordinary_body
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  func.return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

func.func private @asis_body(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %s0 = mhlo.constant dense<[4, 8]> : tensor<2xi32>
  %s1 = mhlo.constant dense<[1, 4, 8]> : tensor<3xi32>
  %r0 = mhlo.custom_call @tfl.reshape(%arg0, %s0) {backend_config = "{}"} : (tensor<1x4x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
  %r1 = mhlo.custom_call @tfl.reshape(%r0, %s1) {backend_config = "{}"} : (tensor<4x8xf32>, tensor<3xi32>) -> tensor<1x4x8xf32>
  %w = mhlo.constant dense<1.0> : tensor<8x8xf32>
  %mm = mhlo.custom_call @tfl.batch_matmul(%r1, %w) {backend_config = "{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}"} : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %b = mhlo.constant dense<2.0> : tensor<8xf32>
  %add = mhlo.custom_call @tfl.add(%mm, %b) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  func.return %add : tensor<1x4x8xf32>
}

func.func private @ordinary_body(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %s0 = mhlo.constant dense<[4, 8]> : tensor<2xi32>
  %s1 = mhlo.constant dense<[1, 4, 8]> : tensor<3xi32>
  %r0 = mhlo.custom_call @tfl.reshape(%arg0, %s0) {backend_config = "{}"} : (tensor<1x4x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
  %r1 = mhlo.custom_call @tfl.reshape(%r0, %s1) {backend_config = "{}"} : (tensor<4x8xf32>, tensor<3xi32>) -> tensor<1x4x8xf32>
  %w = mhlo.constant dense<1.0> : tensor<8x8xf32>
  %mm = mhlo.custom_call @tfl.batch_matmul(%r1, %w) {backend_config = "{adj_x = false, adj_y = false, asymmetric_quantize_inputs = false}"} : (tensor<1x4x8xf32>, tensor<8x8xf32>) -> tensor<1x4x8xf32>
  %b = mhlo.constant dense<2.0> : tensor<8xf32>
  %add = mhlo.custom_call @tfl.add(%mm, %b) {backend_config = "{fused_activation_function = \"NONE\"}"} : (tensor<1x4x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  func.return %add : tensor<1x4x8xf32>
}

// -----

// Explicit quantize / dequantize inside an optimize = "none" region are
// literal. Their carriers are never kept opaque (legalization derives the
// quantized type from backend_config, which the generic carrier
// materialization cannot do), so they legalize in place while the other
// carriers of the body stay opaque; from then on the gated passes leave them
// alone. The body is a fully annotated fully_connected (Q/DQ on the input,
// dequantized int8 weights, Q/DQ on the output): in the ordinary region the
// quantizer fuses it into one int8 fully_connected between a quantize and a
// dequantize; in the as-emitted region every Q/DQ stays where it was written
// and the fully_connected runs in float, with its zero bias.

// MARK-LABEL: func.func private @qdq_asis
// MARK-SAME: attributes {odml.optimization = {optimize = "none"}}

// LEGALIZE-LABEL: func.func private @qdq_asis
// LEGALIZE-NOT: mhlo.custom_call @tfl.quantize
// LEGALIZE-NOT: mhlo.custom_call @tfl.dequantize
// LEGALIZE: %[[Q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x8x!quant.uniform<i8:f32, 5.000000e-01:3>>}>
// LEGALIZE-NEXT: %[[DQ:.*]] = "tfl.dequantize"(%[[Q]]) : (tensor<1x4x8x!quant.uniform<i8:f32, 5.000000e-01:3>>) -> tensor<1x4x8xf32>
// LEGALIZE-NEXT: %[[W:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<8x8x!quant.uniform<i8:f32, 3.125000e-02>>
// LEGALIZE-NEXT: %[[DW:.*]] = "tfl.dequantize"(%[[W]]) : (tensor<8x8x!quant.uniform<i8:f32, 3.125000e-02>>) -> tensor<8x8xf32>
// LEGALIZE-NEXT: %[[FC:.*]] = mhlo.custom_call @tfl.fully_connected(%[[DQ]], %[[DW]]
// LEGALIZE-NEXT: %[[Q2:.*]] = "tfl.quantize"(%[[FC]]) <{qtype = tensor<1x4x8x!quant.uniform<i8:f32, 2.500000e-01>>}>
// LEGALIZE-NEXT: %[[DQ2:.*]] = "tfl.dequantize"(%[[Q2]])
// LEGALIZE-NEXT: return %[[DQ2]]

// MATERIALIZE-LABEL: func.func private @qdq_asis
// MATERIALIZE: %[[Q:.*]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x4x8x!quant.uniform<i8:f32, 5.000000e-01:3>>}>
// MATERIALIZE-NEXT: %[[DQ:.*]] = "tfl.dequantize"(%[[Q]])
// MATERIALIZE-NEXT: %[[W:.*]] = "tfl.pseudo_qconst"()
// MATERIALIZE-NEXT: %[[DW:.*]] = "tfl.dequantize"(%[[W]])
// MATERIALIZE-NEXT: %[[FC:.*]] = "tfl.fully_connected"(%[[DQ]], %[[DW]], %{{.*}}) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true
// MATERIALIZE-SAME: (tensor<1x4x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
// MATERIALIZE-NEXT: %[[Q2:.*]] = "tfl.quantize"(%[[FC]]) <{qtype = tensor<1x4x8x!quant.uniform<i8:f32, 2.500000e-01>>}>
// MATERIALIZE-NEXT: %[[DQ2:.*]] = "tfl.dequantize"(%[[Q2]])
// MATERIALIZE-NEXT: return %[[DQ2]]
// MATERIALIZE-LABEL: func.func private @qdq_ordinary
// MATERIALIZE: %[[Q:.*]] = "tfl.quantize"(%arg0)
// MATERIALIZE-NEXT: %[[W:.*]] = "tfl.pseudo_qconst"()
// MATERIALIZE-NEXT: %[[FC:.*]] = "tfl.fully_connected"(%[[Q]], %[[W]], %{{.*}})
// MATERIALIZE-SAME: -> tensor<1x4x8x!quant.uniform<i8:f32, 2.500000e-01>>
// MATERIALIZE-NEXT: %[[DQ:.*]] = "tfl.dequantize"(%[[FC]])
// MATERIALIZE-NEXT: return %[[DQ]]

// END-LABEL: func.func @qdq_main
// END: %[[Q:.*]] = "tfl.quantize"(%arg0)
// END-NEXT: %[[DQ:.*]] = "tfl.dequantize"(%[[Q]])
// END-NEXT: %[[W:.*]] = "tfl.pseudo_qconst"()
// END-NEXT: %[[DW:.*]] = "tfl.dequantize"(%[[W]])
// END-NEXT: %[[FC:.*]] = "tfl.fully_connected"(%[[DQ]], %[[DW]], %{{.*}}) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true
// END-SAME: (tensor<1x4x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
// END-NEXT: %[[Q2:.*]] = "tfl.quantize"(%[[FC]])
// END-NEXT: %[[DQ2:.*]] = "tfl.dequantize"(%[[Q2]])
// END: "tfl.fully_connected"(%{{.*}}, %{{.*}}, %{{.*}}) <{asymmetric_quantize_inputs = false, fused_activation_function = "NONE", keep_num_dims = true
// END-SAME: -> tensor<1x4x8x!quant.uniform<i8:f32, 2.500000e-01>>
func.func @qdq_main(%arg0: tensor<1x4x8xf32>) -> (tensor<1x4x8xf32>, tensor<1x4x8xf32>) {
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {
    composite_attributes = {optimize = "none"}, decomposition = @qdq_asis
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = stablehlo.composite "odml.optimization_region" %arg0 {
    decomposition = @qdq_ordinary
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  func.return %0, %1 : tensor<1x4x8xf32>, tensor<1x4x8xf32>
}

// Mentions a quantized type so the quant dialect is loaded for this split;
// in the converter pipeline earlier passes take care of that.
func.func private @load_quant_dialect(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0>>) -> tensor<1x!quant.uniform<i8:f32, 1.0>> {
  func.return %arg0 : tensor<1x!quant.uniform<i8:f32, 1.0>>
}

func.func private @qdq_asis(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %q = mhlo.custom_call @tfl.quantize(%arg0) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
  %dq = mhlo.custom_call @tfl.dequantize(%q) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
  %w = mhlo.constant dense<3> : tensor<8x8xi8>
  %dw = mhlo.custom_call @tfl.dequantize(%w) {backend_config = "{scale = [3.125000e-02 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<8x8xi8>) -> tensor<8x8xf32>
  %bias = mhlo.constant dense<0.0> : tensor<8xf32>
  %fc = mhlo.custom_call @tfl.fully_connected(%dq, %dw, %bias) {backend_config = "{fused_activation_function = \"NONE\", keep_num_dims = true, asymmetric_quantize_inputs = false, weights_format = \"DEFAULT\"}"} : (tensor<1x4x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %q2 = mhlo.custom_call @tfl.quantize(%fc) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
  %dq2 = mhlo.custom_call @tfl.dequantize(%q2) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
  func.return %dq2 : tensor<1x4x8xf32>
}

func.func private @qdq_ordinary(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %q = mhlo.custom_call @tfl.quantize(%arg0) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
  %dq = mhlo.custom_call @tfl.dequantize(%q) {backend_config = "{scale = [5.000000e-01 : f32], zero_point = [3 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
  %w = mhlo.constant dense<3> : tensor<8x8xi8>
  %dw = mhlo.custom_call @tfl.dequantize(%w) {backend_config = "{scale = [3.125000e-02 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<8x8xi8>) -> tensor<8x8xf32>
  %bias = mhlo.constant dense<0.0> : tensor<8xf32>
  %fc = mhlo.custom_call @tfl.fully_connected(%dq, %dw, %bias) {backend_config = "{fused_activation_function = \"NONE\", keep_num_dims = true, asymmetric_quantize_inputs = false, weights_format = \"DEFAULT\"}"} : (tensor<1x4x8xf32>, tensor<8x8xf32>, tensor<8xf32>) -> tensor<1x4x8xf32>
  %q2 = mhlo.custom_call @tfl.quantize(%fc) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xf32>) -> tensor<1x4x8xi8>
  %dq2 = mhlo.custom_call @tfl.dequantize(%q2) {backend_config = "{scale = [2.500000e-01 : f32], zero_point = [0 : i64], num_bits = 8 : i64, is_signed = true}"} : (tensor<1x4x8xi8>) -> tensor<1x4x8xf32>
  func.return %dq2 : tensor<1x4x8xf32>
}

// -----

// Regions nest: the body of an optimize = "none" region contains an ordinary
// region. Every decomposition is its own function with its own stamp, so the
// inner body is optimized (its reshape pair, a round trip to the same shape,
// folds away entirely) although it sits inside a `none` region, and the outer
// body's identical reshape pair is left alone.
// Inlining proceeds innermost first until no composite and no decomposition
// is left. Two composites may share one decomposition; it is removed once.

// MARK-LABEL: func.func private @outer
// MARK-SAME: attributes {odml.optimization = {optimize = "none"}}
// MARK-LABEL: func.func private @inner
// MARK-SAME: attributes {odml.optimization = {}}

// END-LABEL: func.func @nested_main
// END-NOT: stablehlo.composite
// END: %[[R0:.*]] = "tfl.reshape"(%arg0
// END-NEXT: %[[R1:.*]] = "tfl.reshape"(%[[R0]]
// END-NEXT: return %[[R1]]
// END-NOT: func.func private
func.func @nested_main(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %0 = stablehlo.composite "odml.optimization_region" %arg0 {
    composite_attributes = {optimize = "none"}, decomposition = @outer
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  func.return %0 : tensor<1x4x8xf32>
}

func.func private @outer(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %s0 = mhlo.constant dense<[4, 8]> : tensor<2xi32>
  %s1 = mhlo.constant dense<[1, 4, 8]> : tensor<3xi32>
  %r0 = mhlo.custom_call @tfl.reshape(%arg0, %s0) {backend_config = "{}"} : (tensor<1x4x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
  %r1 = mhlo.custom_call @tfl.reshape(%r0, %s1) {backend_config = "{}"} : (tensor<4x8xf32>, tensor<3xi32>) -> tensor<1x4x8xf32>
  %0 = stablehlo.composite "odml.optimization_region" %r1 {
    decomposition = @inner
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  %1 = stablehlo.composite "odml.optimization_region" %0 {
    decomposition = @inner
  } : (tensor<1x4x8xf32>) -> tensor<1x4x8xf32>
  func.return %1 : tensor<1x4x8xf32>
}

func.func private @inner(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x8xf32> {
  %s0 = mhlo.constant dense<[4, 8]> : tensor<2xi32>
  %s1 = mhlo.constant dense<[1, 4, 8]> : tensor<3xi32>
  %r0 = mhlo.custom_call @tfl.reshape(%arg0, %s0) {backend_config = "{}"} : (tensor<1x4x8xf32>, tensor<2xi32>) -> tensor<4x8xf32>
  %r1 = mhlo.custom_call @tfl.reshape(%r0, %s1) {backend_config = "{}"} : (tensor<4x8xf32>, tensor<3xi32>) -> tensor<1x4x8xf32>
  func.return %r1 : tensor<1x4x8xf32>
}

// -----

// `mhlo.composite` regions (what a pipeline that marks while the module is
// MHLO sees) are stamped too. (Inlining handles `stablehlo.composite`, the
// form every region has by the end of the converter pipeline.)

// MARK-LABEL: func.func private @mhlo_body
// MARK-SAME: attributes {odml.optimization = {optimize = "none"}}
// LEGALIZE-LABEL: func.func private @mhlo_body
// LEGALIZE: mhlo.custom_call @tfl.exp
// END-LABEL: func.func @mhlo_main
func.func @mhlo_main(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mhlo.composite "odml.optimization_region" %arg0 {
    composite_attributes = {optimize = "none"}, decomposition = @mhlo_body
  } : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

func.func private @mhlo_body(%arg0: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %0 = mhlo.custom_call @tfl.exp(%arg0) {backend_config = "{}"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}
