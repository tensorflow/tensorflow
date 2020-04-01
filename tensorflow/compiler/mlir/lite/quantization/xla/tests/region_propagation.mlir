// RUN: tf-opt -xla-hlo-propagate-quant %s | FileCheck %s --dump-input-on-failure

// -----

// CHECK-LABEL: @mul_add_source_no_params
func @mul_add_source_no_params(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %region = "quant.region"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4xf32>):	// no predecessors
    %mul = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    %add = xla_hlo.add %mul, %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    "quant.return"(%add) : (tensor<4xf32>) -> ()
  }) {input_specs = [f32, f32, f32], logical_kernel = "generic.mul_add", output_specs = [f32]} :
  (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %region : tensor<4xf32>

// CHECK: input_specs = [f32, f32, f32]
// CHECK-SAME: output_specs = [f32]
}

// -----

// CHECK-LABEL: @mul_add_annotated_no_narrow_range
func @mul_add_annotated_no_narrow_range(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %region = "quant.region"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4xf32>):	// no predecessors
    %mul = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    %add = xla_hlo.add %mul, %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    "quant.return"(%add) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0:-128>, !quant.uniform<i8:f32, 1.0:-128>, f32],
    logical_kernel = "generic.mul_add", output_specs = [!quant.uniform<i8:f32, 1.0:-128>]} :
  (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %region : tensor<4xf32>

// CHECK: input_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>, !quant.uniform<i8:f32, 1.000000e+00:-128>, f32]
// CHECK-SAME: output_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>]
}

// -----

// CHECK-LABEL: @mul_add_annotated
func @mul_add_annotated(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>, %arg2: tensor<4xf32>) -> (tensor<4xf32>) {
  %region = "quant.region"(%arg0, %arg1, %arg2) ( {
    ^bb0(%arg3: tensor<4xf32>, %arg4: tensor<4xf32>, %arg5: tensor<4xf32>):	// no predecessors
    %mul = xla_hlo.multiply %arg3, %arg4 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    %add = xla_hlo.add %mul, %arg5 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : tensor<4xf32>
    "quant.return"(%add) : (tensor<4xf32>) -> ()
  }) {input_specs = [!quant.uniform<i8:f32, 1.0:-128>, !quant.uniform<i8<-127:127>:f32, 1.0:-128>, f32],
    logical_kernel = "generic.mul_add", output_specs = [!quant.uniform<i8:f32, 1.0:-128>]} :
  (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %region : tensor<4xf32>

// CHECK: input_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>, !quant.uniform<i8<-127:127>:f32, 1.000000e+00:-128>, !quant.uniform<i32:f32, 1.000000e+00>]
// CHECK-SAME: output_specs = [!quant.uniform<i8:f32, 1.000000e+00:-128>]
}
