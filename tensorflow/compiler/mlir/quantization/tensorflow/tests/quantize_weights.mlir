// RUN: tf-quant-opt %s -split-input-file -tf-quant-quantize-weights | FileCheck %s

module {
  func.func @not_quantize_const() -> (tensor<2x1024xf32>) {
    // Nothing happens if not connected wiht quantizable op.
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    func.return %cst_0: tensor<2x1024xf32>
  }

// CHECK-LABEL: func @not_quantize_const
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<2.000000e+00> : tensor<2x1024xf32>
// CHECK: return %[[W]] : tensor<2x1024xf32>
}

// -----

module {
  func.func @matmul(%arg0: tensor<1x2x2x2xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %0 = "tf.MatMul"(%arg0, %cst_0) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %0: tensor<*xf32>
  }

// CHECK-LABEL: func @matmul
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %[[DEQUANTIZED]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL]] : tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0157480314> : tensor<f32>
// CHECK: %[[CASTED_W:.*]] = "tf.Cast"(%arg0) <{Truncate = false}> : (tensor<*xi8>) -> tensor<*xf32>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.Mul"(%[[CASTED_W]], %[[SCALE]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
// CHECK: return %[[DEQUANTIZED]] : tensor<*xf32>
}

// -----

module {
  func.func @not_quantize_matmul_without_const(%arg0: tensor<1x2x2x2xf32>, %arg1: tensor<2x1024xf32>) -> (tensor<*xf32>) {
    %arg0_identity = "tf.Identity"(%arg0) {device = ""} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
    %arg1_identity = "tf.Identity"(%arg1) {device = ""} : (tensor<2x1024xf32>) -> tensor<2x1024xf32>
    %0 = "tf.MatMul"(%arg0_identity, %arg1_identity) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %0: tensor<*xf32>
  }

// CHECK-LABEL: func @not_quantize_matmul_without_const
// CHECK: %[[ORIGINAL_IDENTITY_1:.*]] = "tf.Identity"(%arg0) {device = ""} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
// CHECK: %[[ORIGINAL_IDENTITY_2:.*]] = "tf.Identity"(%arg1) {device = ""} : (tensor<2x1024xf32>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%[[ORIGINAL_IDENTITY_1]], %[[ORIGINAL_IDENTITY_2]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL]] : tensor<*xf32>
}

// -----

module {
  func.func @quantize_xladotv2_bf16(%arg0: tensor<1x2x2x2xbf16>) -> (tensor<1x2x2x1024xbf16>) {
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x1024xbf16>} : () -> tensor<2x1024xbf16>
    %0 = "tf.XlaDotV2"(%arg0, %cst_0) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<1x2x2x2xbf16>, tensor<2x1024xbf16>) -> tensor<1x2x2x1024xbf16>
    // Check dequantize performed in bf16.
    func.return %0: tensor<1x2x2x1024xbf16>
  }

// CHECK-LABEL: func @quantize_xladotv2_bf16
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[IDENTITY]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xbf16>
// CHECK: %[[MATMUL:.*]] = "tf.XlaDotV2"(%arg0, %[[DEQUANTIZED]]) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<1x2x2x2xbf16>, tensor<2x1024xbf16>) -> tensor<1x2x2x1024xbf16>
// CHECK: return %[[MATMUL]] : tensor<1x2x2x1024xbf16>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xbf16>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<1.574710e-02> : tensor<bf16>
}

// -----

module {
  func.func @matmul_with_identity_and_reshape(%arg0: tensor<1x2x2x2xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<1024x2xf32>} : () -> tensor<1024x2xf32>
    %cst_1 = "tf.Const"() {value = dense<[2, 1024]> : tensor<2xi32>} : () -> tensor<2xi32>
    // Original identity preserved.
    %cst_identity = "tf.Identity"(%cst_0) {device = ""} : (tensor<1024x2xf32>) -> tensor<1024x2xf32>
    %0 = "tf.Reshape"(%cst_identity, %cst_1) : (tensor<1024x2xf32>, tensor<2xi32>) -> tensor<2x1024xf32>
    %1 = "tf.MatMul"(%arg0, %0) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }

// CHECK-LABEL: func @matmul_with_identity_and_reshape
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<1024x2xi8>
// CHECK-DAG: %[[SHAPE:.*]] = "tf.Const"() <{value = dense<[2, 1024]> : tensor<2xi32>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<1024x2xi8>) -> tensor<1024x2xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1024x2xi8>) -> tensor<1024x2xf32>
// CHECK: %[[ORIGINAL_IDENTITY:.*]] = "tf.Identity"(%[[DEQUANTIZED]]) {device = ""} : (tensor<1024x2xf32>) -> tensor<1024x2xf32>
// CHECK: %[[RESHAPED_W:.*]] = "tf.Reshape"(%[[ORIGINAL_IDENTITY]], %[[SHAPE]]) : (tensor<1024x2xf32>, tensor<2xi32>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %[[RESHAPED_W]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL]] : tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0157480314> : tensor<f32>
}

// -----

module {
  func.func @conv2d(%arg0: tensor<1x3x4x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_1 = "tf.Const"() {value = dense<3.000000e+00> : tensor<2x3x3x512xf32>} : () -> tensor<2x3x3x512xf32>
    %0 = "tf.Conv2D"(%arg0, %cst_1) {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    // Dequantize added before BiasAdd.
    %2 = "tf.BiasAdd"(%0, %cst_0) {data_format = "NHWC", device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
    func.return %2: tensor<*xf32>
  }

// CHECK-LABEL: func @conv2d
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x3x3x512xi8>
// CHECK-DAG: %[[BIAS:.*]] = "tf.Const"() <{value = dense<0.000000e+00> : tensor<2xf32>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x3x3x512xi8>) -> tensor<2x3x3x512xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x3x3x512xi8>) -> tensor<2x3x3x512xf32>
// CHECK: %[[CONV2D:.*]] = "tf.Conv2D"(%arg0, %[[DEQUANTIZED:.*]]) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}> {attr_map = "0:strides,1:use_cudnn_on_gpu,2:padding,3:explicit_paddings,4:dilations", device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
// CHECK: %[[BIASADD:.*]] = "tf.BiasAdd"(%[[CONV2D]], %[[BIAS]]) <{data_format = "NHWC"}> {device = ""} : (tensor<*xf32>, tensor<2xf32>) -> tensor<*xf32>
// CHECK: return %[[BIASADD]] : tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0236220472> : tensor<f32>
}

// -----

module {
  func.func @depthwise_conv(%arg0: tensor<1x3x4x512xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<0.000000e+00> : tensor<2xf32>} : () -> tensor<2xf32>
    %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<2x3x3x512xf32>} : () -> tensor<2x3x3x512xf32>
    %0 = "tf.DepthwiseConv2dNative"(%arg0, %cst_1) {
      attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]
    } : (tensor<1x3x4x512xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
    func.return %0: tensor<*xf32>
  }

// CHECK-LABEL: func @depthwise_conv
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x3x3x512xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x3x3x512xi8>) -> tensor<2x3x3x512xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x3x3x512xi8>) -> tensor<2x3x3x512xf32>
// CHECK: %[[DEPTHWISE_CONV2D:.*]] = "tf.DepthwiseConv2dNative"(%arg0, %[[DEQUANTIZED]]) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1]}> {attr_map = "0:strides,1:padding,2:explicit_paddings,3:dilations", device = ""} : (tensor<1x3x4x512xf32>, tensor<2x3x3x512xf32>) -> tensor<*xf32>
// CHECK: return %[[DEPTHWISE_CONV2D]] : tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.00787401571> : tensor<f32>
}

// -----

module {
  func.func @quantize_sharded_weights_with_xladot(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xbf16> {
    %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
    %cst_sharded = "tf.XlaSharding"(%cst) {_XlaSharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", device = "", sharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", unspecified_dims = []} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = "tf.XlaDotV2"(%arg0, %cst_sharded) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<?x?x?x?xf32>, tensor<512x512xf32>) -> tensor<?x?x?x?xf32>
    %2 = "tf.Cast"(%1) {Truncate = false} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xbf16>
    return %2 : tensor<?x?x?x?xbf16>
  }

// CHECK-LABEL: func @quantize_sharded_weights_with_xladot
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<512x512xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<512x512xi8>) -> tensor<512x512xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<512x512xi8>) -> tensor<512x512xf32>
// CHECK: %[[SHARDED_W:.*]] = "tf.XlaSharding"(%[[DEQUANTIZED]]) <{_XlaSharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", sharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01"}> {device = "", unspecified_dims = []} : (tensor<512x512xf32>) -> tensor<512x512xf32>
// CHECK: %[[XLADOT:.*]] = "tf.XlaDotV2"(%arg0, %[[SHARDED_W]]) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<?x?x?x?xf32>, tensor<512x512xf32>) -> tensor<?x?x?x?xf32>
// CHECK: %[[ORIGINAL_CAST:.*]] = "tf.Cast"(%[[XLADOT]]) <{Truncate = false}> : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xbf16>
// CHECK: return %[[ORIGINAL_CAST]] : tensor<?x?x?x?xbf16>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0787401571> : tensor<f32>
}

// -----

module {
  func.func @quantize_sharded_weights_with_xladot_with_identity(%arg0: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %cst = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<512x512xf32>} : () -> tensor<512x512xf32>
    %cst_identity = "tf.Identity"(%cst) {device = ""} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %cst_sharded = "tf.XlaSharding"(%cst_identity) {_XlaSharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", device = "", sharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", unspecified_dims = []} : (tensor<512x512xf32>) -> tensor<512x512xf32>
    %1 = "tf.XlaDotV2"(%arg0, %cst_sharded) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<?x?x?x?xf32>, tensor<512x512xf32>) -> tensor<?x?x?x?xf32>
    return %1 : tensor<?x?x?x?xf32>
  }

// CHECK-LABEL: func @quantize_sharded_weights_with_xladot_with_identity
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<512x512xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<512x512xi8>) -> tensor<512x512xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<512x512xi8>) -> tensor<512x512xf32>
// CHECK: %[[IDENTITY_W:.*]] = "tf.Identity"(%[[DEQUANTIZED]]) {device = ""} : (tensor<512x512xf32>) -> tensor<512x512xf32>
// CHECK: %[[SHARDED_W:.*]] = "tf.XlaSharding"(%[[IDENTITY_W]]) <{_XlaSharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01", sharding = "\08\03\1A\03\01\04\02\22\08\00\04\01\05\02\06\03\070\01"}> {device = "", unspecified_dims = []} : (tensor<512x512xf32>) -> tensor<512x512xf32>
// CHECK: %[[XLADOT:.*]] = "tf.XlaDotV2"(%arg0, %[[SHARDED_W]]) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<?x?x?x?xf32>, tensor<512x512xf32>) -> tensor<?x?x?x?xf32>
// CHECK: return %[[XLADOT]] : tensor<?x?x?x?xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0787401571> : tensor<f32>
}

// -----

module {
  func.func @quantize_xlagather(%arg0: tensor<10x2xi32>) -> tensor<1x300x10xf32> {
    %cst_0 = "tf.Const"() {device = "", value = dense<1.000000e+01> : tensor<200x100x300xf32>} : () -> tensor<200x100x300xf32>
    %cst = "tf.Const"() { value = dense<[1, 1, 300]> : tensor<3xi64> } : () -> tensor<3xi64>
    %0 = "tf.XlaGather"(%cst_0, %arg0, %cst) {dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01\20\01", indices_are_sorted = true} : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<1x300x10xf32>) -> tensor<1x300x10xf32>
    func.return %1 : tensor<1x300x10xf32>
  }

// CHECK-LABEL: func @quantize_xlagather
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<200x100x300xi8>}> : () -> tensor<200x100x300xi8>
// CHECK-DAG: %[[IDX:.*]] = "tf.Const"() <{value = dense<[1, 1, 300]> : tensor<3xi64>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<200x100x300xi8>) -> tensor<200x100x300xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<200x100x300xi8>) -> tensor<200x100x300xf32>
// CHECK: %[[GATHER:.*]] = "tf.XlaGather"(%[[DEQUANTIZED]], %arg0, %[[IDX]]) <{dimension_numbers = "\0A\02\00\01\12\01\00\1A\02\00\01 \01", indices_are_sorted = true}> : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<3xi64>) -> tensor<1x300x10xf32>
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[GATHER]]) {device = ""} : (tensor<1x300x10xf32>) -> tensor<1x300x10xf32>
// CHECK: return %[[IDENTITY]] : tensor<1x300x10xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0787401571> : tensor<f32>}> : () -> tensor<f32>
}

// -----

module {
  func.func @partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<4.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {_tfl_quant_trait = "fully_quantizable", config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }

  func.func private @composite_matmul_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x1024xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    // Dequantization performed here
    return %0 : tensor<*xf32>
  }

// CHECK-LABEL: func @partitioned_call
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %[[OUTPUT:.*]] = "tf.PartitionedCall"(%arg0, %[[DEQUANTIZED]]) <{config = "", config_proto = "", executor_type = "", f = @composite_matmul_fn}> {_tfl_quant_trait = "fully_quantizable"} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[OUTPUT]] : tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0314960629> : tensor<f32>

// CHECK-LABEL: func private @composite_matmul_fn
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL]] : tensor<*xf32>
}

// -----

module {
  func.func @recursive_partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> (tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<4.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %1 = "tf.PartitionedCall"(%arg0, %cst_0) {config = "", config_proto = "", executor_type = "", f = @outer_fn} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %1: tensor<*xf32>
  }

  func.func private @outer_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x1024xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.PartitionedCall"(%arg0, %arg1) {config = "", config_proto = "", executor_type = "", f = @inner_fn} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
  }

  func.func private @inner_fn(%arg0: tensor<1x2x2x3xf32>, %arg1: tensor<2x1024xf32>) -> tensor<*xf32> attributes {tf_quant.composite_function} {
    %0 = "tf.MatMul"(%arg0, %arg1) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    // Dequantization performed here
    return %0 : tensor<*xf32>
  }
}

// CHECK-LABEL: func @recursive_partitioned_call(%arg0: tensor<1x2x2x3xf32>) -> tensor<*xf32>
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %[[OUTPUT:.*]] = "tf.PartitionedCall"(%arg0, %[[DEQUANTIZED]]) <{config = "", config_proto = "", executor_type = "", f = @outer_fn}> : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[OUTPUT]] : tensor<*xf32>

// CHECK-LABEL: func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0314960629> : tensor<f32>

// CHECK-LABEL: func private @outer_fn
// CHECK: %[[OUTER_OUTPUT:.*]] = "tf.PartitionedCall"(%arg0, %arg1) <{config = "", config_proto = "", executor_type = "", f = @inner_fn}> : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[OUTER_OUTPUT]] : tensor<*xf32>

// CHECK-LABEL: func private @inner_fn
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x3xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL]] : tensor<*xf32>

// -----

module {
  func.func @matmul_multiuses(%arg0: tensor<1x2x2x2xf32>, %arg1: tensor<1x2x2x2xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) {
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %0 = "tf.MatMul"(%arg0, %cst_0) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    %1 = "tf.MatMul"(%arg1, %cst_0) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    %cst_identity = "tf.Identity"(%cst_0) {device = ""} : (tensor<2x1024xf32>) -> tensor<2x1024xf32>
    %2 = "tf.MatMul"(%arg0, %cst_identity) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    func.return %0, %1, %2 : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>
  }

// CHECK-LABEL: func @matmul_multiuses
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg0, %[[DEQUANTIZED]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%arg1, %[[DEQUANTIZED]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: %[[ORIGINAL_IDENTITY:.*]] = "tf.Identity"(%[[DEQUANTIZED]]) {device = ""} : (tensor<2x1024xf32>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL_3:.*]] = "tf.MatMul"(%arg0, %[[ORIGINAL_IDENTITY]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: return %[[MATMUL_1]], %[[MATMUL_2]], %[[MATMUL_3]] : tensor<*xf32>, tensor<*xf32>, tensor<*xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0157480314> : tensor<f32>
}

// -----

module {
  func.func @matmul_multiuses_with_unquantizable_op(%arg0: tensor<1x2x2x2xf32>, %arg1: tensor<2x1024xf32>) -> (tensor<*xf32>, tensor<2x1024xf32>) {
    %cst_0 = "tf.Const"() {value = dense<2.000000e+00> : tensor<2x1024xf32>} : () -> tensor<2x1024xf32>
    %0 = "tf.MatMul"(%arg0, %cst_0) {attr_map = "0:transpose_a,1:transpose_a", device = "", transpose_a = false, transpose_b = false} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
    // AddV2 not in quantizable op list.
    %1 = "tf.AddV2"(%arg1, %cst_0) {device = ""} : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
    func.return %0, %1 : tensor<*xf32>, tensor<2x1024xf32>
  }

// CHECK-LABEL: func @matmul_multiuses
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x1024xi8>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<2x1024xi8>) -> tensor<2x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<2x1024xi8>) -> tensor<2x1024xf32>
// CHECK: %[[MATMUL:.*]] = "tf.MatMul"(%arg0, %[[DEQUANTIZED]]) <{transpose_a = false, transpose_b = false}> {attr_map = "0:transpose_a,1:transpose_a", device = ""} : (tensor<1x2x2x2xf32>, tensor<2x1024xf32>) -> tensor<*xf32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%arg1, %[[DEQUANTIZED]]) {device = ""} : (tensor<2x1024xf32>, tensor<2x1024xf32>) -> tensor<2x1024xf32>
// CHECK: return %[[MATMUL]], %[[ADD]] : tensor<*xf32>, tensor<2x1024xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.0157480314> : tensor<f32>
}

// -----

module {
  func.func @matmul_with_while(%arg0: tensor<1x1024xf32>) -> tensor<1x1024xf32> {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"(){value = dense<1.0> : tensor<1024x1024xf32>} : () -> tensor<1024x1024xf32>
    %0:5 = "tf.While"(%cst_0, %cst, %cst_0, %arg0, %cst_1) {T = [i32, i32, i32, f32, f32],_lower_using_switch_merge = true, _num_original_outputs = 5 : i64, _read_only_resource_inputs = [], body = @while_body, cond = @while_cond, device = "", is_stateless = true, output_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<1x1024>, #tf_type.shape<1024x1024>], parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>)
    %1 = "tf.Identity"(%0#3) {device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    func.return %1 : tensor<1x1024xf32>
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1024x1024xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>)
  {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.AddV2"(%arg2, %cst) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<i32>) -> tensor<i32>
    %2 = "tf.MatMul"(%arg3, %arg4) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %4 = "tf.AddV2"(%arg0, %cst) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = "tf.Identity"(%arg4) {device = ""} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %6 = "tf.MatMul"(%arg3, %5) {device = "", transpose_a = false, transpose_b = false} : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
    %7 = "tf.AddV2"(%2, %6) {device = ""} : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %8 = "tf.Identity"(%4) {device = ""} : (tensor<i32>) -> tensor<i32>
    %9 = "tf.Identity"(%arg1) {device = ""} : (tensor<i32>) -> tensor<i32>
    func.return %8, %9, %1, %7, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>
  }

  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1024x1024xf32>) -> tensor<i1>
  {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Less"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    func.return %0 : tensor<i1>
  }
}

// CHECK-LABEL: func @matmul_with_while
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<1024x1024xi8>
// CHECK-DAG: %[[CNT:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[PRESERVE_W:.*]] = "tf.Identity"(%[[W]]) : (tensor<1024x1024xi8>) -> tensor<1024x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[PRESERVE_W]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1024x1024xi8>) -> tensor<1024x1024xf32>
// CHECK: %[[WHILE:.*]] = "tf.While"(%[[CNT]], %[[CNT]], %[[CNT]], %arg0, %[[DEQUANTIZED]]) <{body = @while_body, cond = @while_cond, is_stateless = true, parallel_iterations = 10 : i64, shape_invariant}> {T = [i32, i32, i32, f32, f32], _lower_using_switch_merge = true, _num_original_outputs = 5 : i64, _read_only_resource_inputs = [], device = "", output_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<1x1024>, #tf_type.shape<1024x1024>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>)
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[WHILE:.*]]) {device = ""} : (tensor<1x1024xf32>) -> tensor<1x1024xf32>
// CHECK: return %[[IDENTITY]] : tensor<1x1024xf32>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xf32>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.00787401571> : tensor<f32>

// CHECK-LABEL: func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1024x1024xf32>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xf32>, tensor<1024x1024xf32>)
// CHECK: %[[MATMUL_1:.*]] = "tf.MatMul"(%arg3, %arg4) <{transpose_a = false, transpose_b = false}> {device = ""} : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%arg4) {device = ""} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
// CHECK: %[[MATMUL_2:.*]] = "tf.MatMul"(%arg3, %[[IDENTITY]]) <{transpose_a = false, transpose_b = false}> {device = ""} : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[MATMUL_1]], %[[MATMUL_2]]) {device = ""} : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>

// CHECK-LABEL: func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xf32>, %arg4: tensor<1024x1024xf32>) -> tensor<i1>
// CHECK: return %0 : tensor<i1>

// -----

module {
  func.func @matmul_with_while_bf16(%arg0: tensor<1x1024xbf16>) -> tensor<1x1024xbf16> {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %cst_1 = "tf.Const"() {value = dense<1.0> : tensor<1024x1024xbf16>} : () -> tensor<1024x1024xbf16>
    %0:5 = "tf.While"(%cst_0, %cst, %cst_0, %arg0, %cst_1) {T = [i32, i32, i32, bf16, bf16],_lower_using_switch_merge = true, _num_original_outputs = 5 : i64, _read_only_resource_inputs = [], body = @while_body, cond = @while_cond, device = "", is_stateless = true, output_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<1x1024>, #tf_type.shape<1024x1024>], parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>)
    %1 = "tf.Identity"(%0#3) {device = ""} : (tensor<1x1024xbf16>) -> tensor<1x1024xbf16>
    func.return %1 : tensor<1x1024xbf16>
  }

  func.func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xbf16>, %arg4: tensor<1024x1024xbf16>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>)
  {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.AddV2"(%arg2, %cst) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %1 = "tf.Identity"(%0) {device = ""} : (tensor<i32>) -> tensor<i32>
    %2 = "tf.XlaDotV2"(%arg3, %arg4) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x1024xbf16>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x1024xbf16>) -> tensor<1x1024xbf16>
    %4 = "tf.AddV2"(%arg0, %cst) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %5 = "tf.Identity"(%arg4) {device = ""} : (tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    %6 = "tf.XlaDotV2"(%arg3, %5) {device = "", dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""} : (tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x1024xbf16>
    %7 = "tf.AddV2"(%2, %6) {device = ""} : (tensor<1x1024xbf16>, tensor<1x1024xbf16>) -> tensor<1x1024xbf16>
    %8 = "tf.Identity"(%4) {device = ""} : (tensor<i32>) -> tensor<i32>
    %9 = "tf.Identity"(%arg1) {device = ""} : (tensor<i32>) -> tensor<i32>
    func.return %8, %9, %1, %7, %arg4 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>
  }

  func.func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xbf16>, %arg4: tensor<1024x1024xbf16>) -> tensor<i1>
  {
    %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %0 = "tf.Less"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    func.return %0 : tensor<i1>
  }
}

// CHECK-LABEL: func @matmul_with_while_bf16
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<127> : tensor<1024x1024xi8>
// CHECK-DAG: %[[CNT:.*]] = "tf.Const"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
// CHECK: %[[IDENTITY:.*]] = "tf.Identity"(%[[W]]) : (tensor<1024x1024xi8>) -> tensor<1024x1024xi8>
// CHECK: %[[DEQUANTIZED:.*]] = "tf.PartitionedCall"(%[[IDENTITY]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1024x1024xi8>) -> tensor<1024x1024xbf16>
// CHECK: %[[WHILE:.*]] = "tf.While"(%[[CNT]], %[[CNT]], %[[CNT]], %arg0, %[[DEQUANTIZED]]) <{body = @while_body, cond = @while_cond, is_stateless = true, parallel_iterations = 10 : i64, shape_invariant}> {T = [i32, i32, i32, bf16, bf16], _lower_using_switch_merge = true, _num_original_outputs = 5 : i64, _read_only_resource_inputs = [], device = "", output_shapes = [#tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<>, #tf_type.shape<1x1024>, #tf_type.shape<1024x1024>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>)
// CHECK: %[[ORIGIANL_IDENTITY:.*]] = "tf.Identity"(%[[WHILE:.*]]) {device = ""} : (tensor<1x1024xbf16>) -> tensor<1x1024xbf16>

// CHECK-LABEL: func.func private @composite_dequantize_uniform(%arg0: tensor<*xi8>) -> tensor<*xbf16>
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<7.873530e-03> : tensor<bf16>

// CHECK-LABEL: func private @while_body(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xbf16>, %arg4: tensor<1024x1024xbf16>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<1x1024xbf16>, tensor<1024x1024xbf16>) {
// CHECK: %[[MATMUL_1:.*]] = "tf.XlaDotV2"(%arg3, %arg4) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x1024xbf16>
// CHECK: %[[IDENTITY_2:.*]] = "tf.Identity"(%arg4) {device = ""} : (tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
// CHECK: %[[MATMUL_2:.*]] = "tf.XlaDotV2"(%arg3, %[[IDENTITY_2]]) <{dimension_numbers = "\12\01\00\0A\01\03", precision_config = ""}> {device = ""} : (tensor<1x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<1x1024xbf16>
// CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[MATMUL_1]], %[[MATMUL_2]]) {device = ""} : (tensor<1x1024xbf16>, tensor<1x1024xbf16>) -> tensor<1x1024xbf16>

// CHECK-LABEL: func private @while_cond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<1x1024xbf16>, %arg4: tensor<1024x1024xbf16>) -> tensor<i1> {
// CHECK: return %0 : tensor<i1>

// -----

module {
  func.func @matmul_with_while_returning_mutated_value(%arg0: tensor<i32>, %arg2: tensor<*xf32>) -> (tensor<*xf32>) {
    // The constant should not be quantized.
    %cst = "tf.Const" () {value = dense<1.0> : tensor<1024x1024xf32>} : () -> tensor<1024x1024xf32>
    %0:3 = "tf.While"(%arg0, %cst, %arg2) {
      cond = @cond, body = @body, is_stateless = false
    } : (tensor<i32>, tensor<1024x1024xf32>, tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>, tensor<*xf32>)
    func.return %0#1 : tensor<*xf32>
  }

  func.func private @cond(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<i1> {
    %0 = "tf.Const" () {value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.greater"(%arg0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i1>
    func.return %1 : tensor<i1>
  }

  func.func private @body(%arg0: tensor<i32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>, tensor<*xf32>) {
    %0 = "tf.Const" () {value = dense<1> : tensor<i32>} : () -> tensor<i32>
    %1 = "tf.Sub"(%arg0, %0) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = "tf.MatMul"(%arg2, %arg1) {} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %3 = "tf.AddV2" (%arg1, %arg1)  : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %4 = "tf.Identity"(%1) {device = ""} : (tensor<i32>) -> tensor<i32>
    %5 = "tf.Identity"(%3) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    %6 = "tf.Identity"(%2) {device = ""} : (tensor<*xf32>) -> tensor<*xf32>
    func.return %4, %5, %6 : tensor<i32>, tensor<*xf32>, tensor<*xf32>
  }
}

// CHECK-LABEL: func @matmul_with_while_returning_mutated_value
// CHECK-DAG: %[[W:.*]] = "tf.Const"() <{value = dense<1.000000e+00> : tensor<1024x1024xf32>}> : () -> tensor<1024x1024xf32>

// -----
module {
  func.func @multiple_quantizable_ops_in_graph(%arg0: tensor<1xi32>) -> tensor<1x3x1x1xf32> {
    %cst = "tf.Const"() {device = "", value = dense<0> : tensor<i32>} : () -> tensor<i32>
    %cst_0 = "tf.Const"() {value = dense<1.1> : tensor<2x3x3x1024xf32>} : () -> tensor<2x3x3x1024xf32>
    %cst_1 = "tf.Const"() {value = dense<1.1> : tensor<3x3x1024x1xf32>} : () -> tensor<3x3x1024x1xf32>
    %cst_2 = "tf.Const"() {value = dense<1.1> : tensor<1024x3x4x3xf32>} : () -> tensor<1024x3x4x3xf32>
    %0 = "tf.GatherV2"(%cst_2, %arg0, %cst) {batch_dims = 0 : i64, device = ""} : (tensor<1024x3x4x3xf32>, tensor<1xi32>, tensor<i32>) -> tensor<1x3x4x3xf32>
    %1 = "tf.Conv2D"(%0, %cst_0) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1024xf32>) -> tensor<1x3x2x1024xf32>
    %2 = "tf.Conv2D"(%1, %cst_1) {data_format = "NHWC", device = "", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true} : (tensor<1x3x2x1024xf32>, tensor<3x3x1024x1xf32>) -> tensor<1x3x1x1xf32>
    %3 = "tf.Identity"(%2) {device = ""} : (tensor<1x3x1x1xf32>) -> tensor<1x3x1x1xf32>
    return %3 : tensor<1x3x1x1xf32>
  }

// CHECK-LABEL: func @multiple_quantizable_ops_in_graph
// CHECK-DAG: %[[W_1:.*]] = "tf.Const"() <{value = dense<127> : tensor<2x3x3x1024xi8>}> : () -> tensor<2x3x3x1024xi8>
// CHECK-DAG: %[[W_2:.*]] = "tf.Const"() <{value = dense<127> : tensor<3x3x1024x1xi8>}> : () -> tensor<3x3x1024x1xi8>
// CHECK-DAG: %[[W_3:.*]] = "tf.Const"() <{value = dense<127> : tensor<1024x3x4x3xi8>}> : () -> tensor<1024x3x4x3xi8>
// CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() <{value = dense<0> : tensor<i32>}> {device = ""} : () -> tensor<i32>
// CHECK: %[[IDENTITY_1:.*]] = "tf.Identity"(%[[W_1]]) : (tensor<2x3x3x1024xi8>) -> tensor<2x3x3x1024xi8>
// CHECK: %[[DEQUANTIZED_1:.*]] = "tf.PartitionedCall"(%[[IDENTITY_1]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform__}> : (tensor<2x3x3x1024xi8>) -> tensor<2x3x3x1024xf32>
// CHECK: %[[IDENTITY_2:.*]] = "tf.Identity"(%[[W_2]]) : (tensor<3x3x1024x1xi8>) -> tensor<3x3x1024x1xi8>
// CHECK: %[[DEQUANTIZED_2:.*]] = "tf.PartitionedCall"(%[[IDENTITY_2]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform_}> : (tensor<3x3x1024x1xi8>) -> tensor<3x3x1024x1xf32>
// CHECK: %[[IDENTITY_3:.*]] = "tf.Identity"(%[[W_3]]) : (tensor<1024x3x4x3xi8>) -> tensor<1024x3x4x3xi8>
// CHECK: %[[DEQUANTIZED_3:.*]] = "tf.PartitionedCall"(%[[IDENTITY_3]]) <{config = "", config_proto = "", executor_type = "", f = @composite_dequantize_uniform}> : (tensor<1024x3x4x3xi8>) -> tensor<1024x3x4x3xf32>
// CHECK: %[[GATHER:.*]] = "tf.GatherV2"(%[[DEQUANTIZED_3]], %arg0, %[[AXIS]]) <{batch_dims = 0 : i64}> {device = ""} : (tensor<1024x3x4x3xf32>, tensor<1xi32>, tensor<i32>) -> tensor<1x3x4x3xf32>
// CHECK: %[[CONV_1:.*]] = "tf.Conv2D"(%[[GATHER]], %[[DEQUANTIZED_1]]) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}> {device = ""} : (tensor<1x3x4x3xf32>, tensor<2x3x3x1024xf32>) -> tensor<1x3x2x1024xf32>
// CHECK: %[[CONV_2:.*]] = "tf.Conv2D"(%[[CONV_1]], %[[DEQUANTIZED_2]]) <{data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 2, 1], use_cudnn_on_gpu = true}> {device = ""} : (tensor<1x3x2x1024xf32>, tensor<3x3x1024x1xf32>) -> tensor<1x3x1x1xf32>

// CHECK-LABEL: func private @composite_dequantize_uniform__
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.00866141729> : tensor<f32>}> : () -> tensor<f32>

// CHECK-LABEL: func private @composite_dequantize_uniform_
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.00866141729> : tensor<f32>}> : () -> tensor<f32>

// CHECK-LABEL: func private @composite_dequantize_uniform
// CHECK-DAG: %[[SCALE:.*]] = "tf.Const"() <{value = dense<0.00866141729> : tensor<f32>}> : () -> tensor<f32>

}
