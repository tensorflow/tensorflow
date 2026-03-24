// RUN: tf-opt "-xla-legalize-tf=device-type=XLA_CPU_JIT prefer-tf2xla=true use-tf2xla-fallback=true" %s -verify-diagnostics -mlir-disable-threading  | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {
  // CHECK-LABEL: binary_op
  func.func @binary_op(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: mhlo.atan2
    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: multiple_return_values
  func.func @multiple_return_values(%arg0: tensor<3xi64>) -> tensor<i64> {
     %0:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3xi64>) -> (tensor<i64>, tensor<i64>, tensor<i64>)
    // CHECK: return %1 : tensor<i64>
    func.return %0#0 : tensor<i64>
  }

  // CHECK-LABEL: constant_parameter
  func.func @constant_parameter(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "tf.Const"() {value = dense<1.42> : tensor<2xf32>} : () -> tensor<2xf32>
    // CHECK: mhlo.atan2 %arg0, %2
    %1 = "tf.Atan2"(%arg0, %0) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: uses_translated_return_type
  func.func @uses_translated_return_type(%arg0: tensor<3xf32>) -> tensor<?xf32> {
    // CHECK: tensor.cast %{{[0-9]+}} : tensor<?xf32, #mhlo.type_extensions<bounds = [3]>> to tensor<?xf32>
    %y, %idx = "tf.Unique"(%arg0) {device = ""} : (tensor<3xf32>) -> (tensor<?xf32>, tensor<3xi32>)
    return %y : tensor<?xf32>
  }

  // CHECK-LABEL: @abs
  func.func @abs(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK-NOT: tf.Abs
    %0 = "tf.Abs"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @testBroadcastGradientArgs
  func.func @testBroadcastGradientArgs(%s0: tensor<4xi32>, %s1: tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>) {
    // CHECK:     tf.BroadcastGradientArgs
    %r0, %r1 = "tf.BroadcastGradientArgs"(%s0, %s1) : (tensor<4xi32>, tensor<4xi32>) -> (tensor<1xi32>, tensor<0xi32>)
    func.return %r0, %r1 : tensor<1xi32>, tensor<0xi32>
  }

  // CHECK-LABEL: @acos
  func.func @acos(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK-NOT:  tf.Acos
    %0 = "tf.Acos"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }


  // CHECK-LABEL: unknown_op
  func.func @unknown_op(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: tf.CustomTestOp
    %0 = "tf.CustomTestOp"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: add_v2
  func.func @add_v2(%arg0: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK: mhlo.add %arg0, %arg0 : tensor<2xi32>
    %0 = "tf.AddV2"(%arg0, %arg0) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    func.return %0: tensor<2xi32>
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
    // expected-remark@+1 {{lowering requires bounded tensor operands}}
    %0 = "tf.Atan2"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %0 : tensor<*xf32>
  }

  // CHECK-LABEL: dynamic_operand
  func.func @dynamic_operand(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    // CHECK: tf.Atan2
    // expected-remark@+1 {{lowering requires bounded tensor operands}}
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
    // expected-remark@+1 {{skipping legalization due to unsupported type 'tensor<2x!tf_type.variant>'}}
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
    // CHECK: %[[BROADCAST0:.*]] = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}> : (tensor<4x1xf32>) -> tensor<4x4x1xf32>
    // CHECK: %[[RESHAPE0:.*]] = mhlo.reshape %[[BROADCAST0]] : (tensor<4x4x1xf32>) -> tensor<4x4xf32>
    // CHECK: %[[UPDATED_ARG0:.*]] = "mhlo.broadcast_in_dim"(%[[RESHAPE0]]) <{broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}> : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

    // CHECK: %[[RESHAPE1:.*]] = mhlo.reshape %arg1 : (tensor<4x1x4xf32>) -> tensor<4x4xf32>
    // CHECK: %[[UPDATED_ARG1:.*]] = "mhlo.broadcast_in_dim"(%[[RESHAPE1]]) <{broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}> : (tensor<4x4xf32>) -> tensor<4x4x4xf32>

    // CHECK: %[[RESULT:.*]] = mhlo.atan2 %[[UPDATED_ARG0]], %[[UPDATED_ARG1]] : tensor<4x4x4xf32>
    // CHECK: return %[[RESULT]] : tensor<4x4x4xf32>

    %0 = "tf.Atan2"(%arg0, %arg1) : (tensor<4x1xf32>, tensor<4x1x4xf32>) -> tensor<4x4x4xf32>
    func.return %0: tensor<4x4x4xf32>
  }

  // CHECK-LABEL: func @ternary_op
  func.func @ternary_op(%arg0: tensor<2xi1>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> {
    // CHECK: mhlo.select %arg0, %arg1, %arg2
    %0 = "tf.SelectV2"(%arg0, %arg1, %arg2) : (tensor<2xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
    func.return %0: tensor<2xi32>
  }

  // CHECK-LABEL: func @convert
  func.func @convert(%arg0: tensor<2xi32>) -> tensor<2xf32> {
    // CHECK: mhlo.convert %arg0 : (tensor<2xi32>) -> tensor<2xf32>
    %0 = "tf.Cast"(%arg0) {Truncate = false} : (tensor<2xi32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @constant
  func.func @constant(%arg0: tensor<f32>) -> tensor<f32> {
    // CHECK: %[[ONE:.*]] = mhlo.constant dense<1.000000e+00> : tensor<f32>
    // CHECK: %[[RESULT:.*]] = mhlo.divide %[[ONE]], %arg0 : tensor<f32>
    // CHECK: return %[[RESULT]]

    %0 = "tf.Inv"(%arg0) : (tensor<f32>) -> tensor<f32>
    func.return %0 : tensor<f32>
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
    // expected-remark@+1 {{compilation to HLO failed: INVALID_ARGUMENT: Input 2 to node `tf.XlaPad` with op XlaPad must be a compile-time constant.}}
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
    // CHECK: %[[DIM0:.*]] = mhlo.reshape %[[SLICE0]] : (tensor<1xi32>) -> tensor<i32>

    // CHECK: %[[SLICE1:.*]] = "mhlo.slice"(%[[ARG2]])
    // CHECK-DAG-SAME: start_indices = dense<1> : tensor<1xi64>
    // CHECK-DAG-SAME: limit_indices = dense<2> : tensor<1xi64>
    // CHECK-DAG-SAME: strides = dense<1> : tensor<1xi64>
    // CHECK-SAME: (tensor<2xi32>) -> tensor<1xi32>
    // CHECK: %[[DIM1:.*]] = mhlo.reshape %[[SLICE1]] : (tensor<1xi32>) -> tensor<i32>

    // CHECK: mhlo.dynamic_update_slice %[[ARG0]], %[[ARG1]], %[[DIM0]], %[[DIM1]]

    %0 = "tf.XlaDynamicUpdateSlice"(%arg0, %arg1, %arg2) : (tensor<3x4xi32>, tensor<2x2xi32>, tensor<2xi32>) -> tensor<3x4xi32>
    func.return %0: tensor<3x4xi32>
  }

  // CHECK-LABEL: @sparse_to_dense
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<3x2xi32>, %[[ARG1:.*]]: tensor<3xf32>, %[[ARG2:.*]]: tensor<f32>)
  func.func @sparse_to_dense(%arg0: tensor<3x2xi32>, %arg1: tensor<3xf32>, %arg2: tensor<f32>) -> tensor<3x3xf32> {

  // CHECK:      %[[DEFAULT:.*]] = "mhlo.broadcast_in_dim"(%[[ARG2]]) <{broadcast_dimensions = dense<> : tensor<0xi64>}> : (tensor<f32>) -> tensor<3x3xf32>

  // CHECK:      %[[RESULT:.*]] = "mhlo.scatter"(%[[DEFAULT]], %[[ARG0]], %[[ARG1]])
  // CHECK-SAME: indices_are_sorted = false
  // CHECK-SAME: scatter_dimension_numbers
  // CHECK-SAME:   inserted_window_dims = [0, 1]
  // CHECK-SAME:   scatter_dims_to_operand_dims = [0, 1]
  // CHECK-SAME:   index_vector_dim = 1
  // CHECK-SAME: unique_indices = false
  // CHECK-NEXT: ^bb0(%[[ARG3:.*]]: tensor<f32>, %[[ARG4:.*]]: tensor<f32>):
  // CHECK:        mhlo.return %[[ARG4]] : tensor<f32>
  // CHECK:      })
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
    // CHECK-SAME: <{dimension = 0 : i64}> : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
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

  // Check XlaSpmdFullToShardShape's conversion from split sharding to manual
  // sharding.
  // The split sharding is:
  //   type: OTHER
  //   tile_assignment_dimensions: 2
  //   tile_assignment_dimensions: 1
  //   tile_assignment_devices: 0
  //   tile_assignment_devices: 1
  // Serialized string:
  //   "\08\03\1A\02\02\01\22\02\00\01"
  // The manual sharding is:
  //   type: MANUAL
  // Serialized string:
  //   "\08\04"

  // CHECK-LABEL: @xla_spmd_full_to_shard_shape
  func.func @xla_spmd_full_to_shard_shape(%arg0: tensor<2x2xi64>) -> (tensor<1x2xi64>) {
    // CHECK: %[[SHARDING:.*]] = mhlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<2x2xi64>) -> tensor<2x2xi64>
    // CHECK: %[[MANUAL:.*]] = mhlo.custom_call @SPMDFullToShardShape(%[[SHARDING]]) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2x2xi64>) -> tensor<1x2xi64>
    // CHECK: return %[[MANUAL]]
    %0 = "tf.XlaSpmdFullToShardShape"(%arg0) {dim = -1 : i64, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []} : (tensor<2x2xi64>) -> tensor<1x2xi64>
    func.return %0 : tensor<1x2xi64>
  }

  // Check XlaSpmdShardToFullShape's conversion from manual sharding to split
  // sharding.
  // The manual sharding is:
  //   type: MANUAL
  // Serialized string:
  //   "\08\04"
  // The split sharding is:
  //   type: OTHER
  //   tile_assignment_dimensions: 2
  //   tile_assignment_dimensions: 1
  //   tile_assignment_devices: 0
  //   tile_assignment_devices: 1
  // Serialized string:
  //   "\08\03\1A\02\02\01\22\02\00\01"

  // CHECK-LABEL: @xla_spmd_shard_to_full_shape
  func.func @xla_spmd_shard_to_full_shape(%arg0: tensor<1x2xi64>) -> (tensor<2x2xi64>) {
    // CHECK: %[[SHARDING:.*]] = mhlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x2xi64>) -> tensor<1x2xi64>
    // CHECK: %[[FULL:.*]] = mhlo.custom_call @SPMDShardToFullShape(%[[SHARDING]]) {backend_config = "", mhlo.sharding = "{devices=[2,1]0,1}"} : (tensor<1x2xi64>) -> tensor<2x2xi64>
    // CHECK: return %[[FULL]]
    %0 = "tf.XlaSpmdShardToFullShape"(%arg0) {dim = -1 : i64, full_shape = #tf_type.shape<2x2>, manual_sharding = "\08\03\1A\02\02\01\22\02\00\01", unspecified_dims = []} : (tensor<1x2xi64>) -> tensor<2x2xi64>
    func.return %0 : tensor<2x2xi64>
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

  func.func private @branch0(tensor<2xf32>) -> tensor<2xf32>
  func.func private @branch1(tensor<2xf32>) -> tensor<2xf32>

  func.func @case_with_symbol_ref(%arg0: tensor<i32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    // CHECK: tf.Case
    // expected-remark@+1 {{ops with symbol references are not supported}}
    %0 = "tf.Case"(%arg0, %arg1) {branches = [@branch0, @branch1], is_stateless = false} : (tensor<i32>, tensor<2xf32>) -> tensor<2xf32>
    func.return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: const
  func.func @const() -> tensor<2xf32> {
    // CHECK: mhlo.const
    %cst = "tf.Const"() {value = dense<2.0> : tensor<2xf32>} : () -> tensor<2xf32>
    func.return %cst : tensor<2xf32>
  }

  // CHECK-LABEL: @bounds_propagation
  func.func @bounds_propagation(%input: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf32> {
    %dimension = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    // CHECK: %[[BOUNDED:.*]] = "mhlo.set_dimension_size"
    // CHECK-SAME: <{dimension = 0 : i64}> : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
    %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4xf32>, tensor<i32>, tensor<i32>) -> tensor<?xf32>

    %axis = "tf.Const"() { value = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    // CHECK: %[[REVERSED:.*]] = "mhlo.reverse"(%[[BOUNDED]])
    // CHECK-SAME: {dimensions = dense<0> : tensor<1xi64>}
    // CHECK-SAME: (tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
    %1 = "tf.ReverseV2"(%0, %axis) : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>

    // CHECK: %[[RESULT:.*]] = tensor.cast %[[REVERSED]] : tensor<?xf32, #mhlo.type_extensions<bounds = [4]>> to tensor<?xf32>
    // CHECK: return %[[RESULT]] : tensor<?xf32>
    func.return %1 : tensor<?xf32>
  }

  // CHECK-LABEL: @bounds_propagation_skip_symbol_ref_ops
  func.func @bounds_propagation_skip_symbol_ref_ops(%input: tensor<4xf32>, %size: tensor<i32>) -> tensor<?xf32> {
    %dimension = "tf.Const"() { value = dense<0> : tensor<i32> } : () -> tensor<i32>
    // CHECK: %[[BOUNDED:.*]] = "mhlo.set_dimension_size"
    // CHECK-SAME: <{dimension = 0 : i64}> : (tensor<4xf32>, tensor<i32>) -> tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>
    %0 = "tf.XlaSetDynamicDimensionSize"(%input, %dimension, %size) : (tensor<4xf32>, tensor<i32>, tensor<i32>) -> tensor<?xf32>

    // CHECK: %[[ORIGINAL:.*]] = tensor.cast %[[BOUNDED]] : tensor<?xf32, #mhlo.type_extensions<bounds = [4]>> to tensor<?xf32>

    %axis = "tf.Const"() { value = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    // CHECK: tf.ReverseV2
    // CHECK-SAME: (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>
    // expected-remark@+1 {{lowering requires bounded tensor operands}}
    %1 = "tf.ReverseV2"(%0, %axis) {_body = @identity} : (tensor<?xf32>, tensor<1xi32>) -> tensor<?xf32>

    func.return %1 : tensor<?xf32>
  }

  // CHECK-LABEL: func @set_bound
  func.func @set_bound(%arg0: tensor<i32>) -> tensor<i32> {
    %bound = "tf.Const"() {value = dense<16> : tensor<i32>} : () -> tensor<i32>

    // CHECK: %[[RESULT:.*]] = mhlo.custom_call @SetBound(%arg0) {backend_config = "", mhlo.literal = dense<16> : tensor<i32>}
    %bounded = "tf.XlaSetBound"(%arg0, %bound) : (tensor<i32>, tensor<i32>) -> tensor<i32>

    // CHECK: return %[[RESULT]]
    func.return %bounded : tensor<i32>
  }

  // CHECK-LABEL: func @greater
  func.func @greater(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi1> {
    // CHECK-NEXT:  mhlo.compare GT, %arg0, %arg1
    %0 = "tf.Greater"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
    func.return %0: tensor<2xi1>
  }

  // CHECK-LABEL: batchmatmulv2
  func.func @batchmatmulv2(%arg0: tensor<1x4x2xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x4x4xf32> {
    // CHECK: mhlo.reduce
    // CHECK: mhlo.dot_general
    // CHECK: mhlo.transpose
    %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, grad_x = false, grad_y = false, device = ""} : (tensor<1x4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
    func.return %0 : tensor<3x4x4xf32>
  }

  // CHECK-LABEL: approx_topk
  func.func @approx_topk(%arg0: tensor<!tf_type.resource<tensor<10x500xbf16>>> {tf._user_specified_name = "db"}) -> (tensor<10x10xbf16>) {
    %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<!tf_type.resource<tensor<10x500xbf16>>>) -> tensor<10x500xbf16>
    // CHECK: mhlo.compare  GT
    %values, %indices = "tf.ApproxTopK"(%0) {aggregate_to_topk = true, device = "", is_max_k = true, k = 10 : i64, recall_target = 0.949999988 : f32, reduction_dimension = -1 : i64, reduction_input_size_override = -1 : i64} : (tensor<10x500xbf16>) -> (tensor<10x10xbf16>, tensor<10x10xi32>)
    return %values : tensor<10x10xbf16>
  }

  // CHECK-LABEL: fusedBatchNormV3_noTraining
  func.func @fusedBatchNormV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
    // CHECK: "mhlo.batch_norm_inference"({{.*}}, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> tensor<8x8x8x8xf32>
    %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
    func.return %0#0 : tensor<8x8x8x8xf32>
  }

  // CHECK-LABEL: fusedBatchNormV3_training
  func.func @fusedBatchNormV3_training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
    // CHECK: %[[OUT:.*]], %[[MEAN:.*]], %[[VAR:.*]] = "mhlo.batch_norm_training"({{.*}}, %arg1, %arg2) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
    %0:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, exponential_avg_factor = 1.0 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
    func.return %0#0 : tensor<8x8x8x8xf32>
  }

  // CHECK-LABEL: fusedBatchNormGradV3_noTraining
  func.func @fusedBatchNormGradV3_noTraining(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
    // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
    // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
    // CHECK: %[[scr1:.*]] = mhlo.rsqrt
    // CHECK: %[[bcast_arg3:.+]] = "mhlo.dynamic_broadcast_in_dim"(%arg3, {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
    // CHECK: %[[sub:.*]] = mhlo.subtract %[[act]], %[[bcast_arg3]] : tensor<8x8x8x8xf32>
    // CHECK: %[[mul:.*]] = mhlo.multiply %[[grad]], %[[sub]] : tensor<8x8x8x8xf32>
    // CHECK: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
    // CHECK-NEXT: %[[cmul:.*]] = mhlo.convert %[[mul]] : tensor<8x8x8x8xf32>
    // CHECK-NEXT: %[[init:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NEXT: %[[convert_init:.*]] = mhlo.convert %[[init]] : tensor<f32>
    // CHECK: %[[red1:.*]] = mhlo.reduce(%[[cmul]] init: %[[convert_init]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
    // CHECK: %[[scr2:.*]] = mhlo.convert %[[red1]] : tensor<8xf32>

    // CHECK: %[[mul2:.*]] = mhlo.multiply %arg2, %[[scr1]] : tensor<8xf32>
    // CHECK: %[[bcast_mul2:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[mul2]], {{.*}}) <{broadcast_dimensions = dense<3> : tensor<1xi64>}> : (tensor<8xf32>, tensor<4xindex>) -> tensor<8x8x8x8xf32>
    // CHECK: %[[mul3:.*]] = mhlo.multiply %[[grad]], %[[bcast_mul2]] : tensor<8x8x8x8xf32>

    // CHECK: %[[scale_backprop:.*]] = mhlo.multiply %[[scr1]], %[[scr2]] : tensor<8xf32>

    // CHECK: mhlo.constant dense<[0, 1, 2]> : tensor<3xi64>
    // CHECK: %[[cgrad:.*]] = mhlo.convert %[[grad]] : tensor<8x8x8x8xf32>
    // CHECK: %[[init2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-NEXT: %[[convert_init2:.*]] = mhlo.convert %[[init2]] : tensor<f32>
    // CHECK: %[[red2:.*]] = mhlo.reduce(%[[cgrad]] init: %[[convert_init2]]) applies mhlo.add across dimensions = [0, 1, 2] : (tensor<8x8x8x8xf32>, tensor<f32>) -> tensor<8xf32>
    // CHECK: %[[offset_backprop:.*]] = mhlo.convert %[[red2]] : tensor<8xf32>

    // CHECK: %[[x_backprop:.*]] = mhlo.convert %[[mul3]] : tensor<8x8x8x8xf32>
    // CHECK: return %[[x_backprop]] : tensor<8x8x8x8xf32>

    %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
    func.return %0#0 : tensor<8x8x8x8xf32>
  }

  // CHECK-LABEL: fusedBatchNormGradV3_Training
  func.func @fusedBatchNormGradV3_Training(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8x8x8x8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>, %arg5: tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>) {
    // CHECK-NEXT: %[[grad:.*]] = mhlo.convert %arg0 : tensor<8x8x8x8xf32>
    // CHECK-NEXT: %[[act:.*]] = mhlo.convert %arg1 : tensor<8x8x8x8xf32>
    // CHECK-NEXT: %[[grad_operand:.*]], %[[grad_scale:.*]], %[[grad_offset:.*]] = "mhlo.batch_norm_grad"(%[[act]], %arg2, %arg3, %arg4, %[[grad]]) <{epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}> : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8x8x8x8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>)
    // CHECK-NEXT: %[[x_backprop:.*]] = mhlo.convert %[[grad_operand]] : tensor<8x8x8x8xf32>
    // CHECK: return %[[x_backprop]]
    // CHECK-SAME: tensor<8x8x8x8xf32>

    %0:5 = "tf.FusedBatchNormGradV3"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<0xf32>, tensor<*xf32>)
    func.return %0#0, %0#3, %0#4 : tensor<8x8x8x8xf32>, tensor<0xf32>, tensor<*xf32>
  }

  // CHECK-LABEL: @max_pool_grad_valid
  // CHECK-SAME: %[[INPUT:.*]]: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %[[GRAD:.*]]: tensor<10x12x12x64xf32>
  func.func @max_pool_grad_valid(%orig_input: tensor<10x24x24x64xf32>, %orig_output: tensor<10x12x12x64xf32>, %grad: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
    // CHECK: %[[ZERO:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: %[[RESULT:.*]] = "mhlo.select_and_scatter"(%[[INPUT]], %[[GRAD]], %[[ZERO]]) <{
    // CHECK-SAME:  padding = dense<0> : tensor<4x2xi64>, window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>, window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
    // CHECK-SAME }> ({
    // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
    // CHECK: %[[SELECT_RESULT:.*]] = mhlo.compare GE, %[[VALUE_A]], %[[VALUE_B]] : (tensor<f32>, tensor<f32>) -> tensor<i1>
    // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<i1>
    // CHECK: },  {
    // CHECK: ^bb0(%[[VALUE_A:.*]]: tensor<f32>, %[[VALUE_B:.*]]: tensor<f32>):
    // CHECK: %[[SELECT_RESULT:.*]] = mhlo.add %[[VALUE_A]], %[[VALUE_B]] : tensor<f32>
    // CHECK: mhlo.return %[[SELECT_RESULT]] : tensor<f32>
    // CHECK: }) : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
    // CHECK: return %[[RESULT]] : tensor<10x24x24x64xf32>
    %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
      data_format = "NHWC",
      explicit_paddings = [],
      ksize = [1, 2, 2, 1],
      padding = "VALID",
      strides = [1, 2, 2, 1]
    } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32>
    func.return %result : tensor<10x24x24x64xf32>
  }

  // CHECK-LABEL: @max_pool_grad_same
  func.func @max_pool_grad_same(%orig_input: tensor<2x13x25x7xf32>, %orig_output: tensor<2x4x7x7xf32>, %grad: tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32> {
    // CHECK: padding = dense<{{\[\[}}0, 0], [0, 1], [1, 1], [0, 0]]> : tensor<4x2xi64>
    %result = "tf.MaxPoolGrad"(%orig_input, %orig_output, %grad) {
      data_format = "NHWC",
      explicit_paddings = [],
      ksize = [1, 2, 3, 1],
      padding = "SAME",
      strides = [1, 4, 4, 1]
    } : (tensor<2x13x25x7xf32>, tensor<2x4x7x7xf32>, tensor<2x4x7x7xf32>) -> tensor<2x13x25x7xf32>
    func.return %result : tensor<2x13x25x7xf32>
  }

  //===--------------------------------------------------------------------===//
  // tf.XlaReduceScatter legalization
  //===--------------------------------------------------------------------===//
  // CHECK-LABEL: func @xla_reduce_scatter
  func.func @xla_reduce_scatter(%arg0: tensor<128x128xf32>) -> tensor<64x128xf32> {
      %cst = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
      %cst_0 = "tf.Const"() {value = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>
      // CHECK:          "mhlo.reduce_scatter"(%arg0)
      // CHECK{LITERAL}: replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]>
      // CHECK-SAME:     scatter_dimension = 0
      //
      %1 = "tf.XlaReduceScatter"(%arg0, %cst_0, %cst) {reduce_op = "Add"} : (tensor<128x128xf32>, tensor<4x2xi32>, tensor<i32>) -> tensor<64x128xf32>
      func.return %1 : tensor<64x128xf32>
  }

  // CHECK-LABEL: func @tf_mod
  func.func @tf_mod(%arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %cst = "tf.Const"() {value = dense<7.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK: mhlo.broadcast_in_dim
    // CHECK: mhlo.remainder
    %6 = "tf.Mod"(%arg1, %cst) {_global_shape = [#tf_type.shape<4x8>], device = ""} : (tensor<2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
    return %6 : tensor<2x2xf32>
  }

  // CHECK-LABEL: func @concat_v2
  func.func @concat_v2(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<6x3xf32> {
    // CHECK: "mhlo.concatenate"({{.*}}) <{dimension = 0 : i64}> : (tensor<3x3xf32>, tensor<3x3xf32>) -> tensor<6x3xf32>
    %axis = "tf.Const"() { value = dense<0> : tensor<i64> } : () -> tensor<i64>
    %1 = "tf.ConcatV2"(%arg0, %arg1, %axis) : (tensor<3x3xf32>, tensor<3x3xf32>, tensor<i64>) -> tensor<6x3xf32>
    func.return %1 : tensor<6x3xf32>
  }

  // CHECK-LABEL: func @matrix_diag_part_v3
  // CHECK-SAME: %[[ARG:.*]]: tensor<7x140x128xi32>
  func.func @matrix_diag_part_v3(%arg0: tensor<7x140x128xi32>) -> tensor<7x22x128xi32> {
    %0 = mhlo.constant dense<42> : tensor<i32>  // padding value
    %1 = mhlo.constant dense<[-10, 11]> : tensor<2xi32>  // k
    // CHECK: mhlo.iota
    // CHECK: mhlo.reshape
    // CHECK: mhlo.concatenate
    // CHECK: mhlo.gather
    // CHECK: mhlo.broadcast
    // CHECK: mhlo.select
    %2 = "tf.MatrixDiagPartV3"(%arg0, %1, %0) {
        T = i32, align = "RIGHT_LEFT"
    } : (tensor<7x140x128xi32>, tensor<2xi32>, tensor<i32>) -> tensor<7x22x128xi32>
    func.return %2: tensor<7x22x128xi32>
  }

  // CHECK-LABEL: func @xla_call_module
  func.func @xla_call_module(%arg0: tensor<f32>) -> tensor<*xf32> {
    // Equivalent to the following:
    //
    // module @jit_sin {
    //   func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    //     %0 = mhlo.sine %arg0 : tensor<f32>
    //     return %0 : tensor<f32>
    //   }
    // }
    // CHECK: call @main.1
    %0 = "tf.XlaCallModule"(%arg0) {Sout = [#tf_type.shape<*>], device = "", dim_args_spec = [], function_list = [], disabled_checks = [], has_token_input_output = false, module = "ML\EFR\03MLIRxxx-trunk\00\01\17\05\01\05\01\03\05\03\07\07\t\0B\03K5\07\01\1B\07\0B\13\0B3\0B\0B\0B\0B\0F\0B\13\0B\03\1B\0F\1B\0B\0B\0B\0B\0B\0F\13\0B\0B\0B\0B\03\07\0F\17\07\02\A7\1F\05\0D\03\03\03\07\05\0F\03\0B\0B\1B\0D'\0F)\031\113\05\11\05\13\05\15\05\17\1D\15\17\05\19\17\19\EF\01\05\1B\03\03\1D\0D\05\1F!#%\1D\1D\1D\1F\1D!\1D##\03\03\03+\0D\03-/\1D%\1D'\1D)\1D+)\01\05\11\03\01\03\01\t\04A\05\01\11\01\05\07\03\01\05\03\11\01\t\05\03\05\0B\03\01\01\05\06\13\03\01\03\01\07\04\01\03\03\06\03\01\05\01\00\9A\04-\0F\0B\03!\1B\1D\05\1B\83/\1F\15\1D\15\11\13\15\11\11\0F\0B\11builtin\00vhlo\00module\00func_v1\00sine_v1\00return_v1\00sym_name\00jit_sin\00arg_attrs\00function_type\00res_attrs\00sym_visibility\00jit(sin)/jit(main)/sin\00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\00jax.arg_info\00x\00mhlo.sharding\00{replicated}\00jax.result_info\00\00main\00public\00", platforms = ["CPU"], version = 6 : i64} : (tensor<f32>) -> tensor<*xf32>
    func.return %0 : tensor<*xf32>
  }

  // Verifies that the following functions are added from xla_call_module. Note this must be at the end of the file.
  // CHECK: func.func private @main.1(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"}) -> tensor<f32> {
  // CHECK:   %0 = mhlo.sine %arg0 : tensor<f32>
  // CHECK:   return %0 : tensor<f32>
  // CHECK: }

}
