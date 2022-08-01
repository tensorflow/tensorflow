// RUN: xla-opt "-xla-legalize-tf-with-tf2xla=device-type=XLA_CPU_JIT legalize-test-only-ops" %s -verify-diagnostics | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// CHECK-LABEL: binary_op
func.func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: mhlo.atan2 %arg0, %arg1 : tensor<2xf32>
  %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: unknown_op
func.func @unknown_op(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: tf.CustomTestOp
  %0 = "tf.CustomTestOp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: not_allowlisted_op
func.func @not_allowlisted_op(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  // CHECK: tf.TensorListReserve
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?x?xf32>>>
  // CHECK: tf.TensorListGetItem
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  func.return %1 : tensor<?x?x?xf32>
}

// CHECK-LABEL: unranked_operand
func.func @unranked_operand(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: tf.Atan2
  // expected-remark@+1 {{lowering requires static shaped tensor operands}}
  %0 = "tf.Atan2"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: dynamic_operand
func.func @dynamic_operand(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: tf.Atan2
  // expected-remark@+1 {{lowering requires static shaped tensor operands}}
  %0 = "tf.Atan2"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>

  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: tuple_type
func.func @tuple_type(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // Verifies that the pass can handle operands of non-tensor type like tuple
  // from non TensorFlow ops.
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: unsupported_dtype
func.func @unsupported_dtype(%arg0: tensor<2x!tf_type.variant>) -> tensor<2x!tf_type.variant> {
  // CHECK: tf.AddN
  // expected-remark@+1 {{unsupported type: tensor<2x!tf_type.variant>}}
  %0 = "tf.AddN"(%arg0, %arg0) : (tensor<2x!tf_type.variant>, tensor<2x!tf_type.variant>) -> tensor<2x!tf_type.variant>

  func.return %0 : tensor<2x!tf_type.variant>
}

// CHECK-LABEL: multiple_dialect_ops
func.func @multiple_dialect_ops(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: mhlo.negate
  %0 = "mhlo.negate"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: mhlo.atan2
  %1 = "tf.Atan2"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  func.return %1 : tensor<2xf32>
}

// CHECK-LABEL: binary_op_broadcast
func.func @binary_op_broadcast(%arg0: tensor<4x1xf32>, %arg1: tensor<4x1x4xf32>) -> tensor<4x4x4xf32> {
  // CHECK: %[[BROADCAST0:.*]] = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<4x1xf32>) -> tensor<4x4x1xf32>
  // CHECK: %[[RESHAPE0:.*]] = "mhlo.reshape"(%[[BROADCAST0]]) : (tensor<4x4x1xf32>) -> tensor<4x4xf32>
  // CHECK: %[[UPDATED_ARG0:.*]] = "mhlo.broadcast_in_dim"(%[[RESHAPE0]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

  // CHECK: %[[RESHAPE1:.*]] = "mhlo.reshape"(%arg1) : (tensor<4x1x4xf32>) -> tensor<4x4xf32>
  // CHECK: %[[UPDATED_ARG1:.*]] = "mhlo.broadcast_in_dim"(%[[RESHAPE1]]) {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>} : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

  // CHECK: %[[RESULT:.*]] = mhlo.atan2 %[[UPDATED_ARG0]], %[[UPDATED_ARG1]] : tensor<4x4x4xf32>
  // CHECK: return %[[RESULT]] : tensor<4x4x4xf32>

  %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<4x1xf32>, tensor<4x1x4xf32>) -> tensor<4x4x4xf32>
  func.return %0: tensor<4x4x4xf32>
}

// CHECK-LABEL: func @ternary_op
func.func @ternary_op(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: "mhlo.select"(%arg0, %arg1, %arg2)
  %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// CHECK-LABEL: func @convert
func.func @convert(%arg0: tensor<2xi32>) -> tensor<2xf32> {
  // CHECK: mhlo.convert(%arg0) : (tensor<2xi32>) -> tensor<2xf32>
  %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @constant
func.func @constant(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK: %[[RESULT:.*]] = mhlo.divide %[[ONE]], %arg0 : tensor<2xf32>
  // CHECK: return %[[RESULT]]

  %0 = "tf.Inv"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @greater
func.func @greater(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
  // CHECK-NEXT:  "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type SIGNED>, comparison_direction = #mhlo<comparison_direction GT>}
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  func.return %0: tensor<2xi1>
}

// CHECK-LABEL: func @const_inputs
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2xf64>, %[[ARG1:.*]]: tensor<f64>,
func.func @const_inputs(%arg0: tensor<2x2xf64>, %arg1: tensor<f64>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>) -> tensor<6x5xf64> {

  // CHECK: "mhlo.pad"(%[[ARG0]], %[[ARG1]])
  // CHECK-SAME-DAG: edge_padding_high = dense<[1, 2]> : tensor<2xi64>
  // CHECK-SAME-DAG: edge_padding_low = dense<[2, 1]> : tensor<2xi64>
  // CHECK-SAME-DAG: interior_padding = dense<[1, 0]> : tensor<2xi64>

  %0 = mhlo.constant dense<[2, 1]> : tensor<2xi32>
  %1 = mhlo.constant dense<[1, 2]> : tensor<2xi32>
  %2 = mhlo.constant dense<[1, 0]> : tensor<2xi32>
  %3 = "tf.XlaPad"(%arg0, %arg1, %0, %1, %2) : (tensor<2x2xf64>, tensor<f64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<6x5xf64>
  func.return %3 : tensor<6x5xf64>
}

func.func @non_const_inputs(%arg0: tensor<2x2xf64>, %arg1: tensor<f64>, %arg2: tensor<2xi32>, %arg3: tensor<2xi32>, %arg4: tensor<2xi32>) -> tensor<6x5xf64> {
  // expected-remark@+1 {{lowering requires operand #2 to be a constant}}
  %0 = "tf.XlaPad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<2x2xf64>, tensor<f64>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<6x5xf64>
  func.return %0 : tensor<6x5xf64>
}

// CHECK-LABEL: dynamic_result_type
func.func @dynamic_result_type(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  // CHECK: %[[RESULT:.*]] = mhlo.atan2 %arg0, %arg0 : tensor<2xf32>
  // CHECK: tensor.cast %[[RESULT]] : tensor<2xf32> to tensor<*xf32>
  %0 = "tf.Atan2"(%arg0, %arg0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<*xf32>

  // return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

func.func @truncated_normal() -> tensor<2x2xf32> {
  // CHECK-NOT: tf.TruncatedNormal
  %0 = mhlo.constant dense<[2, 2]> : tensor<2xi32>
  %1 = "tf.TruncatedNormal"(%0) {T = i32, device = "", dtype = f32, seed = 0 : i64, seed2 = 1950157571 : i64} : (tensor<2xi32>) -> tensor<2x2xf32>
  func.return %1 : tensor<2x2xf32>
}

// CHECK-LABEL: dynamic_update_slice
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x4xi32>, %[[ARG1:.*]]: tensor<2x2xi32>, %[[ARG2:.*]]: tensor<2xi32>
func.func @dynamic_update_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<2x2xi32>, %arg2: tensor<2xi32>) -> tensor<3x4xi32> {

  // CHECK: %[[SLICE0:.*]] = "mhlo.slice"(%[[ARG2]])
  // CHECK-DAG-SAME: start_indices = dense<0> : tensor<1xi64>
  // CHECK-DAG-SAME: limit_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: %[[DIM0:.*]] = "mhlo.reshape"(%[[SLICE0]]) : (tensor<1xi32>) -> tensor<i32>

  // CHECK: %[[SLICE1:.*]] = "mhlo.slice"(%[[ARG2]])
  // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
  // CHECK-DAG-SAME: limit_indices = dense<2> : tensor<1xi64>
  // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: (tensor<2xi32>) -> tensor<1xi32>
  // CHECK: %[[DIM1:.*]] = "mhlo.reshape"(%[[SLICE1]]) : (tensor<1xi32>) -> tensor<i32>

  // CHECK: "mhlo.dynamic_update_slice"(%[[ARG0]], %[[ARG1]], %[[DIM0]], %[[DIM1]])

  %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<3x4xi32>, tensor<2x2xi32>, tensor<2xi32>) -> tensor<3x4xi32>
  func.return %0: tensor<3x4xi32>
}

// CHECK-LABEL: @sparse_to_dense
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xi32>, %[[ARG1:.*]]: tensor<3xf32>, %[[ARG2:.*]]: tensor<f32>)
func.func @sparse_to_dense(%arg0: tensor<3x2xi32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>) -> tensor<3x3xf32> {

// CHECK:      %[[DEFAULT:.*]] = "mhlo.broadcast_in_dim"(%[[ARG2]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<3x3xf32>

// CHECK:      %[[RESULT:.*]] = "mhlo.scatter"(%[[DEFAULT]], %[[ARG0]], %[[ARG1]]) ({
// CHECK:      ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
// CHECK:        "mhlo.return"(%[[ARG4]]) : (tensor<f32>) -> ()
// CHECK:      })
// CHECK-SAME: indices_are_sorted = false
// CHECK-SAME: scatter_dimension_numbers
// CHECK-SAME:   inserted_window_dims = [0, 1]
// CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
// CHECK-SAME:   index_vector_dim = 1
// CHECK-SAME: unique_indices = false
// CHECK-SAME: (tensor<3x3xf32>, tensor<3x2xi32>, tensor<3xf32>) -> tensor<3x3xf32>

// return %[[RESULT]] : tensor<3x3xf32>

  %cst = mhlo.constant dense<3> : tensor<2xi32>
  %0 = "tf.SparseToDense"(%arg0, %cst, %arg1, %arg2) {validate_indices = true}: (tensor<3x2xi32>, tensor<2xi32>, tensor<3xf32>, tensor<f32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: reverse_sequence
func.func @reverse_sequence(%arg0: tensor<4x2x3x1x1xi32>, %arg1: tensor<3xi32>) -> tensor<4x2x3x1x1xi32> {
  // CHECK-NOT: tf.ReverseSequence
  %0 = "tf.ReverseSequence"(%arg0, %arg1) {batch_dim = 2 : i64, seq_dim = 0 : i64}: (tensor<4x2x3x1x1xi32>, tensor<3xi32>) -> tensor<4x2x3x1x1xi32>
  func.return %0 : tensor<4x2x3x1x1xi32>
}

// CHECK-LABEL: mirror_pad
func.func @mirror_pad(%arg0: tensor<2x3xcomplex<f64>>) -> tensor<4x7xcomplex<f64>> {
  %0 = mhlo.constant dense<[[1, 1], [2, 2]]> : tensor<2x2xi32>
  // CHECK-NOT: tf.MirrorPad
  %1 = "tf.MirrorPad"(%arg0, %0) {mode = "SYMMETRIC"} : (tensor<2x3xcomplex<f64>>, tensor<2x2xi32>) -> tensor<4x7xcomplex<f64>>
  func.return %1 : tensor<4x7xcomplex<f64>>
}

// CHECK-LABEL: bucketize
func.func @bucketize(%arg0: tensor<2x5xf32>) -> tensor<2x5xi32> {
  // CHECK-NOT: tf.Bucketize
  %0 = "tf.Bucketize"(%arg0) {boundaries = [0.000000e+00 : f32, 3.000000e+00 : f32, 8.000000e+00 : f32, 1.100000e+01 : f32]} : (tensor<2x5xf32>) -> tensor<2x5xi32>
  func.return %0 : tensor<2x5xi32>
}

// CHECK-LABEL: arg_min
func.func @arg_min(%arg0: tensor<6xf64>) -> tensor<i32> {
  // CHECK-NOT: ArgMin
  %0 = mhlo.constant dense<0> : tensor<i32>
  %1 = "tf.ArgMin"(%arg0, %0) : (tensor<6xf64>, tensor<i32>) -> tensor<i32>
  func.return %1 : tensor<i32>
}

// CHECK-LABEL: non_max_suppression_v4
func.func @non_max_suppression_v4(%arg0: tensor<3x4xf32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<2xi32> {
  %max_size = mhlo.constant dense<2> : tensor<i32>
  // CHECK-NOT: tf.NonMaxSuppressionV4
  %0:2 = "tf.NonMaxSuppressionV4"(%arg0, %arg1, %max_size, %arg2, %arg3) {pad_to_max_output_size = true}: (tensor<3x4xf32>, tensor<3xf32>, tensor<i32>, tensor<f32>, tensor<f32>) -> (tensor<2xi32>, tensor<i32>)
  func.return %0#0 : tensor<2xi32>
}

// CHECK-LABEL: bessel_i0e
func.func @bessel_i0e(%arg0: tensor<3xf16>, %arg1: tensor<3xf32>, %arg2: tensor<3xf64>) -> (tensor<3xf16>, tensor<3xf32>, tensor<3xf64>) {
  // CHECK-NOT: tf.BesselI0e
  %0 = "tf.BesselI0e"(%arg0) : (tensor<3xf16>) -> (tensor<3xf16>)
  %1 = "tf.BesselI0e"(%arg1) : (tensor<3xf32>) -> (tensor<3xf32>)
  %2 = "tf.BesselI0e"(%arg2) : (tensor<3xf64>) -> (tensor<3xf64>)
  func.return %0, %1, %2 : tensor<3xf16>, tensor<3xf32>, tensor<3xf64>
}

// CHECK-LABEL: bessel_i1e
func.func @bessel_i1e(%arg0: tensor<3xf16>, %arg1: tensor<3xf32>, %arg2: tensor<3xf64>) -> (tensor<3xf16>, tensor<3xf32>, tensor<3xf64>) {
  // CHECK-NOT: tf.BesselI1e
  %0 = "tf.BesselI1e"(%arg0) : (tensor<3xf16>) -> (tensor<3xf16>)
  %1 = "tf.BesselI1e"(%arg1) : (tensor<3xf32>) -> (tensor<3xf32>)
  %2 = "tf.BesselI1e"(%arg2) : (tensor<3xf64>) -> (tensor<3xf64>)
  func.return %0, %1, %2 : tensor<3xf16>, tensor<3xf32>, tensor<3xf64>
}

// CHECK-LABEL: diag
func.func @diag(%arg0: tensor<2xf32>) -> tensor<2x2xf32> {
  // CHECK-NOT: tf.Diag
  %0 = "tf.Diag"(%arg0) : (tensor<2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: random_uniform_int
func.func @random_uniform_int(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<1000xi32> {
  %0 = "tf.Const"() {value = dense<1000> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NOT: tf.RandomUniformInt
  %1 = "tf.RandomUniformInt"(%0, %arg0, %arg1) {seed = 0 : i64, seed2 = 0 : i64} : (tensor<1xi32>, tensor<i32>, tensor<i32>) -> tensor<1000xi32>
  func.return %1 : tensor<1000xi32>
}

// CHECK-LABEL: multinomial
func.func @multinomial(%arg0: tensor<2x4xf32>, %seed: tensor<i32>, %seed2: tensor<i32>) -> tensor<2x10xi32> {
  // CHECK-NOT: tf.Multinomial
  %samples = "tf.Const"() { value = dense<10> : tensor<i32> } : () -> tensor<i32>
  %1 = "tf.Multinomial"(%arg0, %samples) {seed = 0, seed2 = 0}: (tensor<2x4xf32>, tensor<i32>) -> tensor<2x10xi32>
  func.return %1 : tensor<2x10xi32>
}

// CHECK-LABEL: @set_dynamic_dimension_size
func.func @set_dynamic_dimension_size(%input: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf32> {
  %dimension = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
  // CHECK: mhlo.set_dimension_size
  // CHECK-SAME: {dimension = 0 : i64} : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
  %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4xf32>, tensor<i32>, tensor<i32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: unique
func.func @unique(%arg0: tensor<5xf32>) -> (tensor<?xf32>, tensor<?xi32>) {
  // CHECK-NOT: tf.Unique
  %0, %1 = "tf.Unique"(%arg0) : (tensor<5xf32>) -> (tensor<?xf32>, tensor<?xi32>)
  func.return %0, %1 : tensor<?xf32> , tensor<?xi32>
}

// CHECK-LABEL: @erfinv
func.func @erfinv(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: tf.Erfinv
  %0 = "tf.Erfinv"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @ndtri
func.func @ndtri(%input: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NOT: tf.Ndtri
  %0 = "tf.Ndtri"(%input) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @fake_param
func.func @fake_param() -> tensor<4xf32> {
  // CHECK-NOT: tf.FakeParam
  %0 = "tf.FakeParam"() {shape = #tf_type.shape<4>} : () -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: @parameterized_truncated_normal
func.func @parameterized_truncated_normal(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> tensor<10000000xf32> {
  %0 = "tf.Const"() {value = dense<10000000> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-NOT: tf.ParameterizedTruncatedNormal
  %1 = "tf.ParameterizedTruncatedNormal"(%0, %arg0, %arg1, %arg2, %arg3) {seed = 0 : i64, seed2 = 0 : i64} : (tensor<1xi32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<10000000xf32>
  func.return %1 : tensor<10000000xf32>
}

// CHECK-LABEL: @xla_svd
func.func @xla_svd(%arg0: tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>) {
  // CHECK-NOT: XlaSvd
  %s, %u, %v = "tf.XlaSvd"(%arg0) {max_iter = 1, epsilon = 1.0E-09 : f32, precision_config = ""} : (tensor<1x1xf32>) -> (tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>)
  func.return %s, %u, %v : tensor<1xf32>, tensor<1x1xf32>, tensor<1x1xf32>
}

func.func @identity(%arg0: f32) -> f32 {
 func.return %arg0 : f32
}

// This test verifies that legalization for ops with symbol reference attribute
// is not attempted even if they are in allow-list. XLA op kernels for these
// ops compile the function to HLO on-demand which won't work in our case as it
// may contain unsupported ops in the fallback nor we provide XlaCompiler to
// the kernel. Using a allowed op Atan2 to protect against future addition of a
// new op with a symbol ref.

// CHECK-LABEL: @atan2_with_symbol_ref
func.func @atan2_with_symbol_ref(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: tf.Atan2
  // expected-remark@+1 {{ops with symbol references are not supported}}
  %0 = "tf.Atan2"(%arg0, %arg0) {_body = @identity} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>

  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: const
func.func @const() -> tensor<2xf32> {
  // CHECK: mhlo.const
  %cst = "tf.Const"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
  func.return %cst : tensor<2xf32>
}

// TODO(hinsu): Add a test with a valid TF op for which tf2xla kernel is
// available but doesn't support this instance.
}
