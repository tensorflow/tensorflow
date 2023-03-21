// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo --mlir-print-op-generic --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: mlir-hlo-opt --hlo-legalize-to-stablehlo=allow-experimental-features --mlir-print-op-generic --split-input-file --verify-diagnostics %s | FileCheck %s

// ============ ATTRIBUTES ============

// ArgResultAlias aka #mhlo.result_alias is unused at the moment.
// ChannelHandle aka #mhlo.channel_handle is covered below.

func.func @attr_comparison_direction_eq(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction EQ>
    comparison_direction = #mhlo<comparison_direction EQ>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_eq"

func.func @attr_comparison_direction_ne(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction NE>
    comparison_direction = #mhlo<comparison_direction NE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_ne"

func.func @attr_comparison_direction_ge(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction GE>
    comparison_direction = #mhlo<comparison_direction GE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_ge"

func.func @attr_comparison_direction_gt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction GT>
    comparison_direction = #mhlo<comparison_direction GT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_gt"

func.func @attr_comparison_direction_le(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction LE>
    comparison_direction = #mhlo<comparison_direction LE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_le"

func.func @attr_comparison_direction_lt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #stablehlo<comparison_direction LT>
    comparison_direction = #mhlo<comparison_direction LT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_direction_lt"

func.func @attr_comparison_type_notype(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    // CHECK: compare_type = #stablehlo<comparison_type NOTYPE>,
    compare_type = #mhlo<comparison_type NOTYPE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_notype"

func.func @attr_comparison_type_float(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    // CHECK: compare_type = #stablehlo<comparison_type FLOAT>,
    compare_type = #mhlo<comparison_type FLOAT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_float"

func.func @attr_comparison_type_totalorder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    // CHECK: compare_type = #stablehlo<comparison_type TOTALORDER>,
    compare_type = #mhlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_totalorder"

func.func @attr_comparison_type_signed(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    // CHECK: compare_type = #stablehlo<comparison_type SIGNED>,
    compare_type = #mhlo<comparison_type SIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_signed"

func.func @attr_comparison_type_unsigned(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    // CHECK: compare_type = #stablehlo<comparison_type UNSIGNED>,
    compare_type = #mhlo<comparison_type UNSIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "attr_comparison_type_unsigned"

// ConvDimensionNumbers aka #mhlo.conv is covered below.

func.func @attr_custom_call_api_version_unspecified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 0 : i32
    api_version = 0 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_unspecified"

func.func @attr_custom_call_api_version_original(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 1 : i32
    api_version = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_original"

func.func @attr_custom_call_api_version_status_returning(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 2 : i32
    api_version = 2 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_status_returning"

func.func @attr_custom_call_api_version_status_returning_unified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 3 : i32
    api_version = 3 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_custom_call_api_version_status_returning_unified"

// CustomCallSchedule aka #mhlo<custom_call_schedule> is unsupported at the moment (see negative test below).
// DequantizeMode aka #mhlo<dequantize_mode> is unused at the moment.
// DomainKind aka #mhlo<kind> is unsupported at the moment (see negative test below).
// DotDimensionNumbers aka #mhlo.dot is covered below.

func.func @attr_fft_type_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) {
    // CHECK: fft_type = #stablehlo<fft_type FFT>
    fft_type = #mhlo<fft_type FFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_fft"

func.func @attr_fft_type_ifft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) {
    // CHECK: fft_type = #stablehlo<fft_type IFFT>
    fft_type = #mhlo<fft_type IFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_ifft"

func.func @attr_fft_type_rfft(%arg0: tensor<16xf32>) -> tensor<9xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) {
    // CHECK: fft_type = #stablehlo<fft_type RFFT>
    fft_type = #mhlo<fft_type RFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<9xcomplex<f32>>
  func.return %0 : tensor<9xcomplex<f32>>
}
// CHECK-LABEL: "attr_fft_type_rfft"

func.func @attr_fft_type_irfft(%arg0: tensor<9xcomplex<f32>>) -> tensor<16xf32> {
  %0 = "mhlo.fft"(%arg0) {
    // CHECK: fft_type = #stablehlo<fft_type IRFFT>
    fft_type = #mhlo<fft_type IRFFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<9xcomplex<f32>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "attr_fft_type_irfft"

// FusionKind aka #mhlo<fusion_kind> is unsupported at the moment (see negative test below).
// GatherDimensionNumbers aka #mhlo.gather is covered below.

func.func @attr_precision_default(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#stablehlo<precision DEFAULT>]
    precision_config = [#mhlo<precision DEFAULT>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_default"

func.func @attr_precision_high(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#stablehlo<precision HIGH>]
    precision_config = [#mhlo<precision HIGH>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_high"

func.func @attr_precision_highest(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#stablehlo<precision HIGHEST>]
    precision_config = [#mhlo<precision HIGHEST>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "attr_precision_highest"

func.func @attr_rng_algorithm_default(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "mhlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
    rng_algorithm = #mhlo.rng_algorithm<DEFAULT>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_default"

func.func @attr_rng_algorithm_three_fry(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "mhlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
    rng_algorithm = #mhlo.rng_algorithm<THREE_FRY>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_three_fry"

func.func @attr_rng_algorithm_philox(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "mhlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #stablehlo<rng_algorithm PHILOX>
    rng_algorithm = #mhlo.rng_algorithm<PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "attr_rng_algorithm_philox"

func.func @attr_rng_distribution_uniform(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  %0 = "mhlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #stablehlo<rng_distribution UNIFORM>
    rng_distribution = #mhlo.rng_distribution<UNIFORM>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_rng_distribution_uniform"

func.func @attr_rng_distribution_normal(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  %0 = "mhlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #stablehlo<rng_distribution NORMAL>
    rng_distribution = #mhlo.rng_distribution<NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "attr_rng_distribution_normal"

// ScatterDimensionNumbers aka #mhlo.scatter is covered below.

func.func @attr_transpose_no_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose NO_TRANSPOSE>,
    transpose_a = #mhlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_no_transpose"

func.func @attr_transpose_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose TRANSPOSE>,
    transpose_a = #mhlo<transpose TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_transpose"

func.func @attr_transpose_adjoint(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose ADJOINT>,
    transpose_a = #mhlo<transpose ADJOINT>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "attr_transpose_adjoint"

// TypeExtensionsAttr aka #mhlo.type_extensions is covered below.

func.func @attr_type_extensions_bounds(
    %arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [16, ?]>>)
    -> tensor<?x?xf32, #mhlo.type_extensions<bounds = [16, ?]>> {
  // CHECK: "func.return"(%arg0) : (tensor<?x?xf32, #stablehlo.bounds<16, ?>>) -> ()
  func.return %arg0 : tensor<?x?xf32, #mhlo.type_extensions<bounds = [16, ?]>>
}
// CHECK-LABEL: "attr_type_extensions_bounds"

// ============ OPS ============

func.func @op_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_abs"

// AddDependencyOp aka mhlo.add_dependency is unsupported at the moment (see negative test below).

func.func @op_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_add"

func.func @op_after_all(%arg0: !mhlo.token) -> !mhlo.token {
  // CHECK: "stablehlo.after_all"(%arg0) : (!stablehlo.token) -> !stablehlo.token
  %0 = "mhlo.after_all"(%arg0) : (!mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}
// CHECK-LABEL: "op_after_all"

func.func @op_all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  //               CHECK: "stablehlo.all_gather"(%arg0) {
  //          CHECK-SAME:   all_gather_dim = 1 : i64,
  //          CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   use_global_device_ids
  //          CHECK-SAME: } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
    use_global_device_ids
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_all_gather"

func.func @op_all_reduce(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "stablehlo.all_reduce"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.add"(%[[ARG1]], %[[ARG2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   use_global_device_ids
  //          CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "mhlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
    use_global_device_ids
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_all_reduce"

func.func @op_all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  //               CHECK: "stablehlo.all_to_all"(%arg0) {
  //          CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
  //          CHECK-SAME:   concat_dimension = 0 : i64,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
  //          CHECK-SAME:   split_count = 4 : i64,
  //          CHECK-SAME:   split_dimension = 1 : i64
  //          CHECK-SAME: } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  %0 = "mhlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

func.func @op_and(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "mhlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_and"

// AsyncDoneOp aka mhlo.async_done is unsupported at the moment (see negative test below).
// AsyncStartOp aka mhlo.async_start is unsupported at the moment (see negative test below).
// AsyncUpdateOp aka mhlo.async_update is unsupported at the moment (see negative test below).

func.func @op_atan2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.atan2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.atan2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_atan2"

func.func @op_batch_norm_grad(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  %0:3 = "mhlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}
// CHECK-LABEL: "op_batch_norm_grad"

func.func @op_batch_norm_inference(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  //      CHECK: "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  %0 = "mhlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}
// CHECK-LABEL: "op_batch_norm_inference"

func.func @op_batch_norm_training(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  %0:3 = "mhlo.batch_norm_training"(%arg0, %arg1, %arg2) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}
// CHECK-LABEL: "op_batch_norm_training"

// BitcastOp aka mhlo.bitcast is unsupported at the moment (see negative test below).

func.func @op_bitcast_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "stablehlo.bitcast_convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  %0 = "mhlo.bitcast_convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_bitcast_convert"

func.func @op_broadcast_in_dim(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "stablehlo.broadcast_in_dim"(%arg0) {
  // CHECK-SAME:   broadcast_dimensions = dense<1> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<16x16xf32>
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<1> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_broadcast_in_dim"

func.func @op_broadcast(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "stablehlo.broadcast"(%arg0) {
  // CHECK-SAME:   broadcast_sizes = dense<16> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<16x16xf32>
  %0 = "mhlo.broadcast"(%arg0) {
    broadcast_sizes = dense<16> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_broadcast"

func.func @op_case(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.case"(%arg0) ({
  // CHECK-NEXT:   "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<f32>
  %0 = "mhlo.case"(%arg0) ({
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_case"

func.func @op_cbrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.cbrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.cbrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cbrt"

func.func @op_ceil(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.ceil"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.ceil"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_ceil"

func.func @op_cholesky(%arg0: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> {
  //      CHECK: "stablehlo.cholesky"(%arg0) {
  // CHECK-SAME:   lower = true
  // CHECK-SAME: } : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  %0 = "mhlo.cholesky"(%arg0) {
    lower = true
  } : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  func.return %0 : tensor<1x16x16xf32>
}
// CHECK-LABEL: "op_cholesky"

func.func @op_clamp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_clamp"

func.func @op_count_leading_zeros(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.count_leading_zeros"(%arg0) : (tensor<i32>) -> tensor<i32>
  %0 = "mhlo.count_leading_zeros"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_count_leading_zeros"

func.func @op_collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  //               CHECK: "stablehlo.collective_permute"(%arg0) {
  //          CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  //          CHECK-SAME: } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}
// CHECK-LABEL: "op_collective_permute"

func.func @op_compare(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  //      CHECK: "stablehlo.compare"(%arg0, %arg1) {
  // CHECK-SAME:   compare_type = #stablehlo<comparison_type TOTALORDER>,
  // CHECK-SAME:   comparison_direction = #stablehlo<comparison_direction EQ>
  // CHECK-SAME: } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "mhlo.compare"(%arg0, %arg1) {
    comparison_direction = #mhlo<comparison_direction EQ>,
    compare_type = #mhlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_compare"

func.func @op_complex(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<complex<f32>> {
  // CHECK: "stablehlo.complex"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  %0 = "mhlo.complex"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}
// CHECK-LABEL: "op_complex"

func.func @op_compute_reshape_shape(%arg0: index, %arg1: tensor<1xindex>) -> tensor<1xindex> {
  // CHECK: "stablehlo.compute_reshape_shape"(%arg0, %arg1) : (index, tensor<1xindex>) -> tensor<1xindex>
  %0 = "mhlo.compute_reshape_shape"(%arg0, %arg1) : (index, tensor<1xindex>) -> tensor<1xindex>
  func.return %0 : tensor<1xindex>
}
// CHECK-LABEL: "op_compute_reshape_shape"

func.func @op_concatenate(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.concatenate"(%arg0, %arg1) {
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  %0 = "mhlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_concatenate"

func.func @op_constant(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.constant"() {
  // CHECK-SAME:   value = dense<0.000000e+00> : tensor<f32>
  // CHECK-SAME: } : () -> tensor<f32>
  %0 = "mhlo.constant"() {
    value = dense<0.0> : tensor<f32>
  } : () -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_constant"

func.func @op_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "stablehlo.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  %0 = "mhlo.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_convert"

func.func @op_convolution(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  //      CHECK: "stablehlo.convolution"(%arg0, %arg1) {
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  // CHECK-SAME:   feature_group_count = 1 : i64,
  // CHECK-SAME:   lhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   padding = dense<1> : tensor<2x2xi64>,
  // CHECK-SAME:   precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
  // CHECK-SAME:   rhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   window_reversal = dense<false> : tensor<2xi1>,
  // CHECK-SAME:   window_strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_reversal = dense<false> : tensor<2xi1>,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}
// CHECK-LABEL: "op_convolution"

// CopyOp aka mhlo.copy is unsupported at the moment (see negative test below).

func.func @op_cosine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.cosine"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.cosine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cosine"

func.func @op_create_token() -> !mhlo.token {
  // CHECK: "stablehlo.create_token"() : () -> !stablehlo.token
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  func.return %0 : !mhlo.token
}
// CHECK-LABEL: "op_create_token"

func.func @op_cross_replica_sum(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "stablehlo.cross-replica-sum"(%arg0) {
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  //          CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_cross_replica_sum"

func.func @op_cstr_reshapable(%arg0: index, %arg1: tensor<1xindex>) -> !shape.witness {
  // CHECK: "stablehlo.cstr_reshapable"(%arg0, %arg1) : (index, tensor<1xindex>) -> !shape.witness
  %0 = "mhlo.cstr_reshapable"(%arg0, %arg1) : (index, tensor<1xindex>) -> !shape.witness
  func.return %0 : !shape.witness
}
// CHECK-LABEL: "op_cstr_reshapable"

func.func @called_computation() { func.return }
func.func @op_custom_call_api_version_original(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.custom_call"(%arg0) {
  // CHECK-SAME:   api_version = 1 : i32,
  // CHECK-SAME:   backend_config = "",
  // CHECK-SAME:   call_target_name = "foo",
  // CHECK-SAME:   called_computations = [@foo],
  // CHECK-SAME:   has_side_effect = false,
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>],
  // CHECK-SAME:   output_operand_aliases = [
  // CHECK-SAME:     #stablehlo.output_operand_alias<
  // CHECK-SAME:       output_tuple_indices = [],
  // CHECK-SAME:       operand_index = 0,
  // CHECK-SAME:       operand_tuple_indices = []>]
  // CHECK-SAME:   result_layouts = [dense<> : tensor<0xindex>]
  // CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    has_side_effect = false,
    backend_config = "",
    api_version = 1 : i32,
    called_computations = [@foo],
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_custom_call_api_version_original"

func.func @op_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_divide"

// DomainOp aka mhlo.domain is unsupported at the moment (see negative test below).

func.func @op_dot_general(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  //      CHECK: "stablehlo.dot_general"(%arg0, %arg1) {
  // CHECK-SAME:   dot_dimension_numbers = #stablehlo.dot<
  // CHECK-SAME:     lhs_batching_dimensions = [0],
  // CHECK-SAME:     rhs_batching_dimensions = [0],
  // CHECK-SAME:     lhs_contracting_dimensions = [2],
  // CHECK-SAME:     rhs_contracting_dimensions = [1]
  // CHECK-SAME:   >,
  // CHECK-SAME:   precision_config = []
  // CHECK-SAME: } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = []
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}
// CHECK-LABEL: "op_dot_general"

func.func @op_dot(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "stablehlo.dot"(%arg0, %arg1) {
  // CHECK-SAME:   precision_config = []
  // CHECK-SAME: } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  %0 = "mhlo.dot"(%arg0, %arg1) {
    precision_config = []
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "op_dot"

func.func @op_dynamic_broadcast_in_dim(%arg0: tensor<?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  //      CHECK: "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
  // CHECK-SAME:   broadcast_dimensions = dense<1> : tensor<1xi64>,
  // CHECK-SAME:   known_expanding_dimensions = dense<> : tensor<0xi64>,
  // CHECK-SAME:   known_nonexpanding_dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = dense<1> : tensor<1xi64>,
    known_expanding_dimensions = dense<[]> : tensor<0xi64>,
    known_nonexpanding_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: "op_dynamic_broadcast_in_dim"

func.func @op_dynamic_conv(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<4xi32>) -> tensor<1x?x?x16xf32> {
  //      CHECK: "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  // CHECK-SAME:   feature_group_count = 1 : i64,
  // CHECK-SAME:   lhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   padding = dense<1> : tensor<2x2xi64>,
  // CHECK-SAME:   precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
  // CHECK-SAME:   rhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   window_reversal = dense<false> : tensor<2xi1>,
  // CHECK-SAME:   window_strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<4xi32>) -> tensor<1x?x?x16xf32>
  %0 = "mhlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_reversal = dense<false> : tensor<2xi1>,
    dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<4xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}
// CHECK-LABEL: "op_dynamic_conv"

func.func @op_dynamic_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xf32> {
  //      CHECK: "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false
  // CHECK-SAME: } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  %0 = "mhlo.dynamic_gather"(%arg0, %arg1, %arg2) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  func.return %0 : tensor<1x5x8xf32>
}
// CHECK-LABEL: "op_dynamic_gather"

func.func @op_dynamic_iota(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  //      CHECK: "stablehlo.dynamic_iota"(%arg0) {
  // CHECK-SAME:   iota_dimension = 0 : i64
  // CHECK-SAME: } : (tensor<1xindex>) -> tensor<?xf32>
  %0 = "mhlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_dynamic_iota"

func.func @op_dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %0 = "mhlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_dynamic_pad"

func.func @op_dynamic_reshape(%arg0: tensor<16xf32>, %arg1: tensor<?xindex>) -> tensor<?x?xf32> {
  // CHECK: "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<16xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  %0 = "mhlo.dynamic_reshape"(%arg0, %arg1) : (tensor<16xf32>, tensor<?xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}
// CHECK-LABEL: "op_dynamic_reshape"

func.func @op_dynamic_slice(%arg0: tensor<16xf32>, %arg1: tensor<i64>) -> tensor<4xf32> {
  //      CHECK: "stablehlo.dynamic_slice"(%arg0, %arg1) {
  // CHECK-SAME:   slice_sizes = dense<4> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>, tensor<i64>) -> tensor<4xf32>
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = dense<4> : tensor<1xi64>
  } : (tensor<16xf32>, tensor<i64>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
// CHECK-LABEL: "op_dynamic_slice"

func.func @op_dynamic_update_slice(%arg0: tensor<16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
  // CHECK: "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  %0 = "mhlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_dynamic_update_slice"

func.func @op_einsum(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "stablehlo.einsum"(%arg0, %arg1) {
  // CHECK-SAME:   einsum_config = "ab,bc->ac"
  // CHECK-SAME: } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  %0 = "mhlo.einsum"(%arg0, %arg1) {
    einsum_config = "ab,bc->ac"
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}
// CHECK-LABEL: "op_einsum"

func.func @op_exponential_minus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.exponential_minus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.exponential_minus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_exponential_minus_one"

func.func @op_exponential(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_exponential"

func.func @op_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  //      CHECK: "stablehlo.fft"(%arg0) {
  // CHECK-SAME:   fft_length = dense<16> : tensor<1xi64>,
  // CHECK-SAME:   fft_type = #stablehlo<fft_type FFT>
  // CHECK-SAME: } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  %0 = "mhlo.fft"(%arg0) {
    fft_type = #mhlo<fft_type FFT>,
    fft_length = dense<16> : tensor<1xi64>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}
// CHECK-LABEL: "op_fft"

func.func @op_floor(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_floor"

// FusionOp aka mhlo.fusion is unsupported at the moment (see negative test below).

func.func @op_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>) -> tensor<1x5x1xf32> {
  //      CHECK: "stablehlo.gather"(%arg0, %arg1) {
  // CHECK-SAME:   dimension_numbers = #stablehlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<1> : tensor<3xi64>
  // CHECK-SAME: } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>) -> tensor<1x5x1xf32>
  %0 = "mhlo.gather"(%arg0, %arg1) {
    dimension_numbers = #mhlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    slice_sizes = dense<1> : tensor<3xi64>,
    indices_are_sorted = false
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>) -> tensor<1x5x1xf32>
  func.return %0 : tensor<1x5x1xf32>
}
// CHECK-LABEL: "op_gather"

func.func @op_get_dimension_size(%arg0: tensor<?xf32>) -> tensor<i32> {
  //      CHECK: "stablehlo.get_dimension_size"(%arg0) {
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: } : (tensor<?xf32>) -> tensor<i32>
  %0 = "mhlo.get_dimension_size"(%arg0) {
    dimension = 0 : i64
  } : (tensor<?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_get_dimension_size"

func.func @op_get_tuple_element(%arg0: tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32> {
  //      CHECK: "stablehlo.get_tuple_element"(%arg0) {
  // CHECK-SAME:   index = 4 : i32
  // CHECK-SAME: } : (tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  %0 = "mhlo.get_tuple_element"(%arg0) {
    index = 4 : i32
  } : (tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_get_tuple_element"

func.func @op_if(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.if"(%arg0) ({
  // CHECK-NEXT:   "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<f32>
  %0 = "mhlo.if"(%arg0) ({
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }, {
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_if"

func.func @op_imag(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  %0 = "mhlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_imag"

func.func @op_infeed(%arg0: !mhlo.token) -> (tensor<f32>, !mhlo.token) {
  //               CHECK: "stablehlo.infeed"(%arg0) {
  //          CHECK-SAME:   infeed_config = "",
  // CHECK-SAME{LITERAL}:   layout = [[]]
  //          CHECK-SAME: } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  %0:2 = "mhlo.infeed"(%arg0) {
    infeed_config = "",
    layout = [[]]
  } : (!mhlo.token) -> (tensor<f32>, !mhlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !mhlo.token
}
// CHECK-LABEL: "op_infeed"

func.func @op_iota() -> tensor<16xf32> {
  //      CHECK: "stablehlo.iota"() {
  // CHECK-SAME:   iota_dimension = 0 : i64
  // CHECK-SAME: } : () -> tensor<16xf32>
  %0 = "mhlo.iota"() {
    iota_dimension = 0 : i64
  } : () -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_iota"

func.func @op_is_finite(%arg0: tensor<f32>) -> tensor<i1> {
  // CHECK: "stablehlo.is_finite"(%arg0) : (tensor<f32>) -> tensor<i1>
  %0 = "mhlo.is_finite"(%arg0) : (tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_is_finite"

func.func @op_log(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_log"

func.func @op_log_plus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.log_plus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_log_plus_one"

func.func @op_logistic(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.logistic"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.logistic"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_logistic"

func.func @op_map(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.map"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.abs"(%[[ARG1]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "mhlo.map"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      %1 = "mhlo.abs"(%arg1) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_map"

func.func @op_maximum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_maximum"

func.func @op_minimum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_minimum"

func.func @op_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_multiply"

func.func @op_negate(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_negate"

func.func @op_not(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: "stablehlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  %0 = "mhlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_not"

func.func @op_optimization_barrier(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.optimization_barrier"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.optimization_barrier"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_optimization_barrier"

func.func @op_or(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "stablehlo.or"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_or"

func.func @op_outfeed(%arg0: tensor<f32>, %arg1: !mhlo.token) -> !mhlo.token {
  //      CHECK: "stablehlo.outfeed"(%arg0, %arg1) {
  // CHECK-SAME:   outfeed_config = ""
  // CHECK-SAME: } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  %0 = "mhlo.outfeed"(%arg0, %arg1) {
    outfeed_config = ""
  } : (tensor<f32>, !mhlo.token) -> (!mhlo.token)
  func.return %0 : !mhlo.token
}
// CHECK-LABEL: "op_outfeed"

func.func @op_pad(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.pad"(%arg0, %arg1) {
  // CHECK-SAME:   edge_padding_high = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   edge_padding_low = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   interior_padding = dense<0> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_high = dense<4> : tensor<1xi64>,
    edge_padding_low = dense<4> : tensor<1xi64>,
    interior_padding = dense<0> : tensor<1xi64>
  } : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_pad"

func.func @op_partition_id() -> tensor<ui32> {
  // CHECK: "stablehlo.partition_id"() : () -> tensor<ui32>
  %0 = "mhlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "op_partition_id"

func.func @op_popcnt(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.popcnt"(%arg0) : (tensor<i32>) -> tensor<i32>
  %0 = "mhlo.popcnt"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_popcnt"

func.func @op_power(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_power"

func.func @op_real_dynamic_slice(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %0 = "mhlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "op_real_dynamic_slice"

func.func @op_real(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  %0 = "mhlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_real"

func.func @op_recv(%arg0: !mhlo.token) -> (tensor<f32>, !mhlo.token) {
  //      CHECK: "stablehlo.recv"(%arg0) {
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME:   is_host_transfer = true
  // CHECK-SAME: } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  %0:2 = "mhlo.recv"(%arg0) {
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
    is_host_transfer = true
  } : (!mhlo.token) -> (tensor<f32>, !mhlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !mhlo.token
}
// CHECK-LABEL: "op_recv"

func.func @op_reduce(%arg0: tensor<16xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_reduce"

func.func @op_reduce_precision(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.reduce_precision"(%arg0) {
  // CHECK-SAME:   exponent_bits = 8 : i32,
  // CHECK-SAME:   mantissa_bits = 10 : i32
  // CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.reduce_precision"(%arg0) {
    exponent_bits = 8 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_reduce_precision"

func.func @op_reduce_scatter(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //               CHECK: "stablehlo.reduce_scatter"(%arg0) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.add"(%[[ARG1]], %[[ARG2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   scatter_dimension = 0 : i64
  //          CHECK-SAME: } : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "mhlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "mhlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
    scatter_dimension = 0 : i64
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_reduce_scatter"

func.func @op_reduce_window(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x5x8x7xf32> {
  //               CHECK: "stablehlo.reduce_window"(%arg0, %arg1) ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: tensor<f32>, %[[ARG3:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.maximum"(%[[ARG2]], %[[ARG3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) {
  //          CHECK-SAME:   base_dilations = dense<1> : tensor<4xi64>,
  // CHECK-SAME{LITERAL}:   padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
  //          CHECK-SAME:   window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  //          CHECK-SAME:   window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  //          CHECK-SAME:   window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>
  //          CHECK-SAME: } : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x5x8x7xf32>
  %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "mhlo.maximum"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    base_dilations = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x5x8x7xf32>
  func.return %0 : tensor<2x5x8x7xf32>
}
// CHECK-LABEL: "op_reduce_window"

func.func @op_remainder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.remainder"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.remainder"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_remainder"

func.func @op_replica_id() -> tensor<ui32> {
  // CHECK: "stablehlo.replica_id"() : () -> tensor<ui32>
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "op_replica_id"

func.func @op_reshape(%arg0: tensor<16xf32>) -> tensor<4x4xf32> {
  // CHECK: "stablehlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<4x4xf32>
  %0 = "mhlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
// CHECK-LABEL: "op_reshape"

func.func @op_return(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "stablehlo.case"(%arg0) ({
  // CHECK-NEXT:   "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<f32>
  %0 = "mhlo.case"(%arg0) ({
    "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_return"

func.func @op_reverse(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.reverse"(%arg0) {
  // CHECK-SAME:   dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "mhlo.reverse"(%arg0) {
    dimensions = dense<0> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_reverse"

func.func @op_rng_bit_generator(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  //      CHECK: "stablehlo.rng_bit_generator"(%arg0) {
  // CHECK-SAME:   rng_algorithm = #stablehlo<rng_algorithm PHILOX>
  // CHECK-SAME: } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %0:2 = "mhlo.rng_bit_generator"(%arg0) {
    rng_algorithm = #mhlo.rng_algorithm<PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}
// CHECK-LABEL: "op_rng_bit_generator"

func.func @op_rng(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<?xindex>) -> tensor<f32> {
  //      CHECK: "stablehlo.rng"(%arg0, %arg1, %arg2) {
  // CHECK-SAME:   rng_distribution = #stablehlo<rng_distribution NORMAL>
  // CHECK-SAME: } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  %0 = "mhlo.rng"(%arg0, %arg1, %arg2) {
    rng_distribution = #mhlo.rng_distribution<NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<?xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_rng"

func.func @op_round_nearest_afz(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.round_nearest_afz"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.round_nearest_afz"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_round_nearest_afz"

func.func @op_round_nearest_even(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.round_nearest_even"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.round_nearest_even"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_round_nearest_even"

func.func @op_rsqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.rsqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.rsqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_rsqrt"

func.func @op_scatter(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
  //      CHECK: "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG3:arg.*]]: tensor<f32>, %[[ARG4:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.add"(%[[ARG3]], %[[ARG4]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:  indices_are_sorted = true,
  // CHECK-SAME:  scatter_dimension_numbers = #stablehlo.scatter<
  // CHECK-SAME:    update_window_dims = [1],
  // CHECK-SAME:    inserted_window_dims = [0, 1],
  // CHECK-SAME:    scatter_dims_to_operand_dims = [0, 1],
  // CHECK-SAME:    index_vector_dim = 1
  // CHECK-SAME:  >,
  // CHECK-SAME:  unique_indices = true
  // CHECK-SAME: } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "mhlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1],
      inserted_window_dims = [0, 1],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1
    >,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}
// CHECK-LABEL: "op_scatter"

func.func @op_select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
  //      CHECK: "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG31:arg.*]]: tensor<f32>, %[[ARG41:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL11:.*]] = "stablehlo.compare"(%[[ARG31]], %[[ARG41]]) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:     "stablehlo.return"(%[[VAL11]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG32:arg.*]]: tensor<f32>, %[[ARG42:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL12:.*]] = "stablehlo.add"(%[[ARG32]], %[[ARG42]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "stablehlo.return"(%[[VAL12]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   padding = dense<0> : tensor<4x2xi64>,
  // CHECK-SAME:   window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  // CHECK-SAME:   window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  // CHECK-SAME: } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  %0 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "mhlo.compare"(%arg3, %arg4) {compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "mhlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    padding = dense<0> : tensor<4x2xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %0 : tensor<10x24x24x64xf32>
}
// CHECK-LABEL: "op_select_and_scatter"

func.func @op_select(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_select"

func.func @op_send(%arg0: tensor<f32>, %arg1: !mhlo.token) -> !mhlo.token {
  //      CHECK: "stablehlo.send"(%arg0, %arg1) {
  // CHECK-SAME:   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME:   is_host_transfer = true
  // CHECK-SAME: } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  %0 = "mhlo.send"(%arg0, %arg1) {
    channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
    is_host_transfer = true
  } : (tensor<f32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}
// CHECK-LABEL: "op_send"

func.func @op_set_dimension_size(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.set_dimension_size"(%arg0, %arg1) {
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: } : (tensor<?xf32>, tensor<i32>) -> tensor<16xf32>
  %0 = "mhlo.set_dimension_size"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<?xf32>, tensor<i32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_set_dimension_size"

func.func @op_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "mhlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_left"

func.func @op_shift_right_arithmetic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "mhlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_right_arithmetic"

func.func @op_shift_right_logical(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "mhlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "op_shift_right_logical"

func.func @op_sign(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.sign"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.sign"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sign"

func.func @op_sine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.sine"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.sine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sine"

func.func @op_slice(%arg0: tensor<16xf32>) -> tensor<4xf32> {
  //      CHECK: "stablehlo.slice"(%arg0) {
  // CHECK-SAME:   limit_indices = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   start_indices = dense<0> : tensor<1xi64>,
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<4xf32>
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<0> : tensor<1xi64>,
    limit_indices = dense<4> : tensor<1xi64>,
    strides = dense<1> : tensor<1xi64>
  } : (tensor<16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}
// CHECK-LABEL: "op_slice"

func.func @op_sort(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "stablehlo.sort"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "stablehlo.compare"(%[[ARG1]], %[[ARG2]]) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:     "stablehlo.return"(%[[VAL1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) {
  // CHECK-SAME:   dimension = 0 : i64,
  // CHECK-SAME:   is_stable = true
  // CHECK-SAME: } : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "mhlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "mhlo.compare"(%arg1, %arg2) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "mhlo.return"(%1) : (tensor<i1>) -> ()
  }) {
    dimension = 0 : i64,
    is_stable = true
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "op_sort"

func.func @op_sqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_sqrt"

func.func @op_subtract(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.subtract"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.subtract"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_subtract"

func.func @op_tan(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "stablehlo.custom_call"(%arg0) {
  //          CHECK-SAME:    call_target_name = "mhlo.tan"
  // CHECK-SAME{LITERAL}:    mhlo.attributes = {}
  // CHECK-SAME{LITERAL}:    mhlo.version = 1 : i64
  //          CHECK-SAME: } : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.tan"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_tan"

func.func @op_tanh(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.tanh"(%arg0) : (tensor<f32>) -> tensor<f32>
  %0 = "mhlo.tanh"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_tanh"

func.func @op_torch_index_select(%arg0: tensor<5x1x5xf32>, %arg1: tensor<2xi32>) ->  tensor<2x1x5xf32> {
  //      CHECK: "stablehlo.torch_index_select"(%arg0, %arg1) {
  // CHECK-SAME:   batch_dims = 0 : i64,
  // CHECK-SAME:   dim = 0 : i64
  // CHECK-SAME: } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  %0 = "mhlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  func.return %0 : tensor<2x1x5xf32>
}
// CHECK-LABEL: "op_torch_index_select"

func.func @op_trace(%arg0: tensor<f32>) {
  //      CHECK: "stablehlo.trace"(%arg0) {
  // CHECK-SAME:   tag = "foo"
  // CHECK-SAME: } : (tensor<f32>) -> ()
  "mhlo.trace"(%arg0) {
    tag = "foo"
  } : (tensor<f32>) -> ()
  func.return
}
// CHECK-LABEL: "op_trace"

func.func @op_transpose(%arg0: tensor<16x8xf32>) ->  tensor<8x16xf32> {
  //      CHECK: "stablehlo.transpose"(%arg0) {
  // CHECK-SAME:   permutation = dense<[1, 0]> : tensor<2xi64>
  // CHECK-SAME: } : (tensor<16x8xf32>) -> tensor<8x16xf32>
  %0 = "mhlo.transpose"(%arg0) {
    permutation = dense<[1, 0]> : tensor<2xi64>
  } : (tensor<16x8xf32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}
// CHECK-LABEL: "op_transpose"

func.func @op_triangular_solve(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  //      CHECK: "stablehlo.triangular_solve"(%arg0, %arg1) {
  // CHECK-SAME:   left_side = true,
  // CHECK-SAME:   lower = true,
  // CHECK-SAME:   transpose_a = #stablehlo<transpose NO_TRANSPOSE>,
  // CHECK-SAME:   unit_diagonal = true
  // CHECK-SAME: } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    transpose_a = #mhlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}
// CHECK-LABEL: "op_triangular_solve"

func.func @op_tuple(%arg0: tensor<f32>) -> tuple<tensor<f32>> {
  // CHECK: "stablehlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>
  %0 = "mhlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>
  func.return %0 : tuple<tensor<f32>>
}
// CHECK-LABEL: "op_tuple"

func.func @op_unary_einsum(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
  //      CHECK: "stablehlo.unary_einsum"(%arg0) {
  // CHECK-SAME:   einsum_config = "ab->a"
  // CHECK-SAME: } : (tensor<8x16xf32>) -> tensor<8xf32>
  %0 = "mhlo.unary_einsum"(%arg0) {
    einsum_config = "ab->a"
  } : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
// CHECK-LABEL: "op_unary_einsum"

func.func @op_uniform_dequantize(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32> {
  // CHECK: "stablehlo.uniform_dequantize"(%arg0) : (tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>) -> tensor<f32>
  %0 = "mhlo.uniform_dequantize"(%arg0) : (tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "op_uniform_dequantize"

func.func @op_uniform_quantize(%arg0: tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>> {
  // CHECK: "stablehlo.uniform_quantize"(%arg0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>
  %0 = "mhlo.uniform_quantize"(%arg0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 34.0:16>>
}
// CHECK-LABEL: "op_uniform_quantize"

func.func @op_while(%arg0: tensor<i1>) -> tensor<i1> {
  //      CHECK: "stablehlo.while"(%arg0) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<i1>):
  // CHECK-NEXT:     "stablehlo.return"(%[[ARG1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<i1>):
  // CHECK-NEXT:     "stablehlo.return"(%[[ARG1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<i1>
  %0 = "mhlo.while"(%arg0) ({
    ^bb0(%arg1: tensor<i1>):
      "mhlo.return"(%arg1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i1>):
      "mhlo.return"(%arg1) : (tensor<i1>) -> ()
  }) : (tensor<i1>) -> tensor<i1>
  func.return %0: tensor<i1>
}
// CHECK-LABEL: "op_while"

// XlaRngGetAndUpdateStateOp aka mhlo.xla.rng_get_and_update_state is unsupported at the moment (see negative test below).

func.func @op_xor(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "stablehlo.xor"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "mhlo.xor"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "op_xor"

// ============ TYPES ============

func.func @type_i1(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "mhlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}
// CHECK-LABEL: "type_i1"

func.func @type_i4(%arg0: tensor<i4>, %arg1: tensor<i4>) -> tensor<i4> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<i4>, tensor<i4>) -> tensor<i4>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i4>, tensor<i4>) -> tensor<i4>
  func.return %0 : tensor<i4>
}
// CHECK-LABEL: "type_i4"

func.func @type_i8(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  func.return %0 : tensor<i8>
}
// CHECK-LABEL: "type_i8"

func.func @type_i16(%arg0: tensor<i16>, %arg1: tensor<i16>) -> tensor<i16> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  func.return %0 : tensor<i16>
}
// CHECK-LABEL: "type_i16"

func.func @type_i32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}
// CHECK-LABEL: "type_i32"

func.func @type_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}
// CHECK-LABEL: "type_i64"

func.func @type_ui4(%arg0: tensor<ui4>, %arg1: tensor<ui4>) -> tensor<ui4> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<ui4>, tensor<ui4>) -> tensor<ui4>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<ui4>, tensor<ui4>) -> tensor<ui4>
  func.return %0 : tensor<ui4>
}
// CHECK-LABEL: "type_ui4"

func.func @type_ui8(%arg0: tensor<ui8>, %arg1: tensor<ui8>) -> tensor<ui8> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  func.return %0 : tensor<ui8>
}
// CHECK-LABEL: "type_ui8"

func.func @type_ui16(%arg0: tensor<ui16>, %arg1: tensor<ui16>) -> tensor<ui16> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
  func.return %0 : tensor<ui16>
}
// CHECK-LABEL: "type_ui16"

func.func @type_ui32(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}
// CHECK-LABEL: "type_ui32"

func.func @type_ui64(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  func.return %0 : tensor<ui64>
}
// CHECK-LABEL: "type_ui64"

func.func @type_f8E4M3FN(%arg0: tensor<f8E4M3FN>, %arg1: tensor<f8E4M3FN>) -> tensor<f8E4M3FN> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}
// CHECK-LABEL: "type_f8E4M3FN"

func.func @type_f8E5M2(%arg0: tensor<f8E5M2>, %arg1: tensor<f8E5M2>) -> tensor<f8E5M2> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}
// CHECK-LABEL: "type_f8E5M2"

func.func @type_bf16(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
  func.return %0 : tensor<bf16>
}
// CHECK-LABEL: "type_bf16"

func.func @type_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  func.return %0 : tensor<f16>
}
// CHECK-LABEL: "type_f16"

func.func @type_f32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "type_f32"

func.func @type_f64(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %0 : tensor<f64>
}
// CHECK-LABEL: "type_f64"

func.func @type_complex_f32(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}
// CHECK-LABEL: "type_complex_f32"

func.func @type_complex_f64(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>) -> tensor<complex<f64>> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
  func.return %0 : tensor<complex<f64>>
}
// CHECK-LABEL: "type_complex_f64"

func.func @type_dynamism_ranked(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "stablehlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "mhlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}
// CHECK-LABEL: "type_dynamism_ranked"

func.func @type_dynamism_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "stablehlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  %0 = "mhlo.abs"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
// CHECK-LABEL: "type_dynamism_unranked"

func.func @type_quantization(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "stablehlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>, tensor<f32>) -> tensor<f32>
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<!quant.uniform<i8:f32, 34.0:16>>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}
// CHECK-LABEL: "type_quantization"

func.func @type_sparsity(%arg0: tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<16xf32> {
  // CHECK: "stablehlo.abs"(%arg0) : (tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<16xf32>
  %0 = "mhlo.abs"(%arg0) : (tensor<16xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ] }>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}
// CHECK-LABEL: "type_sparsity"

// AsyncBundle aka !mhlo.async_bundle is unsupported at the moment (see negative test below).

func.func @type_token_callee(%arg0: !mhlo.token) -> !mhlo.token {
  // CHECK: "func.return"(%arg0) : (!stablehlo.token) -> ()
  return %arg0 : !mhlo.token
}
//       CHECK: function_type = (!stablehlo.token) -> !stablehlo.token
// CHECK-LABEL: "type_token_callee"

func.func @type_token_caller(%arg0: !mhlo.token) -> !mhlo.token {
  // CHECK: "func.call"(%arg0) {callee = @type_token_callee} : (!stablehlo.token) -> !stablehlo.token
  %0 = func.call @type_token_callee(%arg0) : (!mhlo.token) -> !mhlo.token
  return %0 : !mhlo.token
}
//       CHECK: function_type = (!stablehlo.token) -> !stablehlo.token
// CHECK-LABEL: "type_token_caller"

func.func @type_token_region(%arg0: tensor<i1>, %arg1: !mhlo.token) {
  //      CHECK: "stablehlo.while"(%arg1) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !stablehlo.token):
  // CHECK-NEXT:     "stablehlo.return"(%arg0) : (tensor<i1>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !stablehlo.token):
  // CHECK-NEXT:     "stablehlo.return"(%[[ARG2]]) : (!stablehlo.token) -> ()
  // CHECK-NEXT: }) : (!stablehlo.token) -> !stablehlo.token
  %0 = "mhlo.while"(%arg1) ({
    ^bb0(%arg2: !mhlo.token):
      mhlo.return %arg0 : tensor<i1>
    }, {
    ^bb0(%arg2: !mhlo.token):
      mhlo.return %arg2 : !mhlo.token
  }) : (!mhlo.token) -> !mhlo.token
  return
}
// CHECK-LABEL: "type_token_region"

func.func @type_tuple(%arg0: tuple<tensor<f32>>) -> tuple<!mhlo.token> {
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo"
  // CHECK: (tuple<tensor<f32>>) -> tuple<!stablehlo.token>
  } : (tuple<tensor<f32>>) -> tuple<!mhlo.token>
  return %0 : tuple<!mhlo.token>
}
// CHECK-LABEL: "type_tuple"

// ============ NEGATIVE TESTS ============
// Some ops, attributes and types used in MHLO programs are not supported in StableHLO.
// The following features are private, and not convertable to StableHLO even
// with the experimental flag.

// -----

func.func @attr_precision_config_invalid() -> tensor<8x8xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.custom_call' that was explicitly marked illegal}}
  %0 = "mhlo.custom_call"() {
    call_target_name = "foo",
    precision_config = [#mhlo<precision PACKED_NIBBLE>, 1 : i32]
  } : () -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @op_add_dependency(%arg0: tensor<16xf32>, %arg1: !mhlo.token) -> tensor<16xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.add_dependency' that was explicitly marked illegal}}
  %0 = "mhlo.add_dependency"(%arg0, %arg1) : (tensor<16xf32>, !mhlo.token) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

func.func @async_computation(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

func.func @op_async_done(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.async_start' that was explicitly marked illegal}}
  %0 = "mhlo.async_start"(%arg0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  // At the moment, mhlo.async_done requires its defining op to be non-empty.
  // As a result, it's impossible to test it in isolation from other async ops.
  // However, if we test it together with other async ops, we cannot get an
  // async_done-specific legalization error.
  %1 = "mhlo.async_done"(%0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (!mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> tensor<16xf32>
  func.return %1 : tensor<16xf32>
}

// -----

func.func @async_computation(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @op_async_start(%arg0: tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>> {
  %0 = "mhlo.async_start"(%arg0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (tensor<16xf32>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  func.return %0 : !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
}

// -----

func.func @async_computation(%arg0: tensor<16xf32>) -> tensor<16xf32>
  attributes {execution_thread = "main"} {
  return %arg0 : tensor<16xf32>
}

// expected-error@+1 {{failed to legalize operation 'func.func' that was explicitly marked illegal}}
func.func @op_async_update(%arg0: !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>> {
  %0 = "mhlo.async_update"(%arg0) {
    called_computation = @async_computation,
    execution_thread = "main"
  } : (!mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>) -> !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
  func.return %0 : !mhlo.async_bundle<tensor<16xf32>, tensor<16xf32>>
}

// -----

func.func @op_bitcast(%arg0: tensor<i32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.bitcast' that was explicitly marked illegal}}
  %0 = "mhlo.bitcast"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_copy(%arg0: tensor<f32>) -> tensor<f32> {
  // mhlo.copy is immediately folded away at the first opportunity,
  // so it doesn't seem to be possible to capture it in FileCheck tests.
  %0 = "mhlo.copy"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_convolution_unknown_dimension_numbers(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.convolution' that was explicitly marked illegal}}
  %0 = "mhlo.convolution"(%arg0, %arg1) {
    window_strides = dense<1> : tensor<2xi64>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = dense<1> : tensor<2xi64>,
    rhs_dilation = dense<1> : tensor<2xi64>,
    window_reversal = dense<false> : tensor<2xi1>,
    dimension_numbers = #mhlo.conv<[b, 0, 1, ?, f]x[0, 1, ?, i, o]->[?, b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32>
  func.return %0 : tensor<32x1x8x8x16xf32>
}

// -----

func.func @op_custom_call_custom_call_schedule(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.custom_call' that was explicitly marked illegal}}
  %0 = "mhlo.custom_call"(%arg0) {
    call_target_name = "foo",
    custom_call_schedule = #mhlo<custom_call_schedule EARLIEST>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_domain(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.domain' that was explicitly marked illegal}}
  %0 = "mhlo.domain"(%arg0) {
    kind = #mhlo<kind sharding>,
    entry_metadata = "\08\01\1A\01\01\22\01\01",
    exit_metadata = "\08\02"
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_fusion(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.fusion' that was explicitly marked illegal}}
  %0 = "mhlo.fusion"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      "mhlo.return"(%arg1) : (tensor<f32>) -> ()
  }) {
    fusion_kind = #mhlo<fusion_kind kCustom>,
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @op_stochastic_convert(%arg0: tensor<f32>, %arg1: tensor<ui32>) -> tensor<i8> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.stochastic_convert' that was explicitly marked illegal}}
  %0 = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<f32>, tensor<ui32>) -> tensor<i8>
  return %0 : tensor<i8>
}

// -----

func.func @op_xla_rng_get_and_update_state() -> tensor<2xui64> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.xla.rng_get_and_update_state' that was explicitly marked illegal}}
  %0 = "mhlo.xla.rng_get_and_update_state"() {
    delta = 1: i64
  } : () -> tensor<2xui64>
  func.return %0 : tensor<2xui64>
}
