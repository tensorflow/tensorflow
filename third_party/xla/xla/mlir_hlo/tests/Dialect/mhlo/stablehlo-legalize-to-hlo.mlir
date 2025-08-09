// RUN: mlir-hlo-opt --stablehlo-legalize-to-hlo --mlir-print-op-generic --split-input-file --verify-diagnostics %s | FileCheck %s

// ============ ATTRIBUTES ============

// ArgResultAlias aka #stablehlo.result_alias is unused at the moment.
// ChannelHandle aka #stablehlo.channel_handle is covered below.

// CHECK-LABEL: "attr_comparison_direction_eq"
func.func @attr_comparison_direction_eq(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction EQ>
    comparison_direction = #stablehlo<comparison_direction EQ>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_direction_ne"
func.func @attr_comparison_direction_ne(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction NE>
    comparison_direction = #stablehlo<comparison_direction NE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_direction_ge"
func.func @attr_comparison_direction_ge(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction GE>
    comparison_direction = #stablehlo<comparison_direction GE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_direction_gt"
func.func @attr_comparison_direction_gt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction GT>
    comparison_direction = #stablehlo<comparison_direction GT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_direction_le"
func.func @attr_comparison_direction_le(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction LE>
    comparison_direction = #stablehlo<comparison_direction LE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_direction_lt"
func.func @attr_comparison_direction_lt(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    // CHECK: comparison_direction = #mhlo<comparison_direction LT>
    comparison_direction = #stablehlo<comparison_direction LT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_type_notype"
func.func @attr_comparison_type_notype(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #mhlo<comparison_type NOTYPE>,
    compare_type = #stablehlo<comparison_type NOTYPE>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_type_float"
func.func @attr_comparison_type_float(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #mhlo<comparison_type FLOAT>,
    compare_type = #stablehlo<comparison_type FLOAT>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_type_totalorder"
func.func @attr_comparison_type_totalorder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #mhlo<comparison_type TOTALORDER>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_type_signed"
func.func @attr_comparison_type_signed(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #mhlo<comparison_type SIGNED>,
    compare_type = #stablehlo<comparison_type SIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_comparison_type_unsigned"
func.func @attr_comparison_type_unsigned(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    // CHECK: compare_type = #mhlo<comparison_type UNSIGNED>,
    compare_type = #stablehlo<comparison_type UNSIGNED>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "attr_custom_call_api_version_unspecified"
func.func @attr_custom_call_api_version_unspecified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 0 : i32
    api_version = 0 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_custom_call_api_version_original"
func.func @attr_custom_call_api_version_original(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 1 : i32
    api_version = 1 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_custom_call_api_version_status_returning"
func.func @attr_custom_call_api_version_status_returning(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 2 : i32
    api_version = 2 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_custom_call_api_version_status_returning_unified"
func.func @attr_custom_call_api_version_status_returning_unified(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    // CHECK: api_version = 3 : i32
    api_version = 3 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_fft_type_fft"
func.func @attr_fft_type_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #mhlo<fft_type FFT>
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}

// -----

// CHECK-LABEL: "attr_fft_type_ifft"
func.func @attr_fft_type_ifft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #mhlo<fft_type IFFT>
    fft_type = #stablehlo<fft_type IFFT>,
    fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}

// -----

// CHECK-LABEL: "attr_fft_type_rfft"
func.func @attr_fft_type_rfft(%arg0: tensor<16xf32>) -> tensor<9xcomplex<f32>> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #mhlo<fft_type RFFT>
    fft_type = #stablehlo<fft_type RFFT>,
    fft_length = array<i64: 16>
  } : (tensor<16xf32>) -> tensor<9xcomplex<f32>>
  func.return %0 : tensor<9xcomplex<f32>>
}

// -----

// CHECK-LABEL: "attr_fft_type_irfft"
func.func @attr_fft_type_irfft(%arg0: tensor<9xcomplex<f32>>) -> tensor<16xf32> {
  %0 = "stablehlo.fft"(%arg0) {
    // CHECK: fft_type = #mhlo<fft_type IRFFT>
    fft_type = #stablehlo<fft_type IRFFT>,
    fft_length = array<i64: 16>
  } : (tensor<9xcomplex<f32>>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "attr_precision_config_default"
func.func @attr_precision_config_default(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#mhlo<precision DEFAULT>]
    precision_config = [#stablehlo<precision DEFAULT>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: "attr_precision_config_high"
func.func @attr_precision_config_high(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#mhlo<precision HIGH>]
    precision_config = [#stablehlo<precision HIGH>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: "attr_precision_config_highest"
func.func @attr_precision_config_highest(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    // CHECK: precision_config = [#mhlo<precision HIGHEST>]
    precision_config = [#stablehlo<precision HIGHEST>]
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: "test_unary_result_accuracy"
func.func @test_unary_result_accuracy(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>) {
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  // CHECK: result_accuracy = #mhlo.result_accuracy<ulps = 10, mode = #mhlo.result_accuracy_mode<TOLERANCE>>
  %cbrt = "stablehlo.cbrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %cosine = "stablehlo.cosine"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %exponential = "stablehlo.exponential"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %exponential_minus_one = "stablehlo.exponential_minus_one"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %log = "stablehlo.log"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %log_plus_one = "stablehlo.log_plus_one"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %logistic = "stablehlo.logistic"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %rsqrt = "stablehlo.rsqrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %sine = "stablehlo.sine"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %sqrt = "stablehlo.sqrt"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %tan = "stablehlo.tan"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  %tanh = "stablehlo.tanh"(%arg0) {
    result_accuracy = #stablehlo.result_accuracy<ulps = 10, mode = #stablehlo.result_accuracy_mode<TOLERANCE>>
  } : (tensor<f32>) -> tensor<f32>
  func.return %cbrt, %cosine, %exponential, %exponential_minus_one, %log, %log_plus_one, %logistic, %rsqrt, %sine, %sqrt, %tan, %tanh : tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: "attr_rng_algorithm_default"
func.func @attr_rng_algorithm_default(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #mhlo.rng_algorithm<DEFAULT>
    rng_algorithm = #stablehlo<rng_algorithm DEFAULT>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: "attr_rng_algorithm_three_fry"
func.func @attr_rng_algorithm_three_fry(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #mhlo.rng_algorithm<THREE_FRY>
    rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: "attr_rng_algorithm_philox"
func.func @attr_rng_algorithm_philox(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    // CHECK: rng_algorithm = #mhlo.rng_algorithm<PHILOX>
    rng_algorithm = #stablehlo<rng_algorithm PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: "attr_rng_distribution_uniform"
func.func @attr_rng_distribution_uniform(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<0xindex>) -> tensor<f32> {
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #mhlo.rng_distribution<UNIFORM>
    rng_distribution = #stablehlo<rng_distribution UNIFORM>
  } : (tensor<f32>, tensor<f32>, tensor<0xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_rng_distribution_normal"
func.func @attr_rng_distribution_normal(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<0xindex>) -> tensor<f32> {
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    // CHECK: rng_distribution = #mhlo.rng_distribution<NORMAL>
    rng_distribution = #stablehlo<rng_distribution NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<0xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "attr_transpose_no_transpose"
func.func @attr_transpose_no_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose NO_TRANSPOSE>,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "attr_transpose_transpose"
func.func @attr_transpose_transpose(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose TRANSPOSE>,
    transpose_a = #stablehlo<transpose TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "attr_transpose_adjoint"
func.func @attr_transpose_adjoint(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    // transpose_a = #mhlo<transpose ADJOINT>,
    transpose_a = #stablehlo<transpose ADJOINT>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "attr_type_extensions_bounds"
func.func @attr_type_extensions_bounds(
    %arg0: tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>>)
    -> tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>> {
  // CHECK: "func.return"([[ARG0:%arg[0-9]+]]) : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [16, ?]>>) -> ()
  func.return %arg0 : tensor<?x?xf32, #stablehlo.type_extensions<bounds = [16, ?]>>
}

// -----

// ============ OPS ============

// CHECK-LABEL: "op_abs"
func.func @op_abs(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.abs"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_add"
func.func @op_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// Tokens flow between StableHLO and MHLO ops, so need to have special converson
// logic. AddDependencyOp is the only op that doesn't exist in StableHLO but
// uses token types, so it can have either StableHLO or MHLO token types as
// input.

// CHECK-LABEL: "add_dependency"
func.func @add_dependency(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // CHECK:      %[[TOK:.*]] = "mhlo.create_token"() {{.*}} : () -> !mhlo.token
  // CHECK-NEXT: %[[COPY:.*]] = "mhlo.add_dependency"(%arg0, %[[TOK]]) : (tensor<3x4xf32>, !mhlo.token) -> tensor<3x4xf32>
  %0 = stablehlo.create_token {xla_shape = "token[]"} : !stablehlo.token
  %1 = mhlo.add_dependency %arg0, %0 : (tensor<3x4xf32>, !stablehlo.token) -> tensor<3x4xf32>
  return %1 : tensor<3x4xf32>
}

// -----

// CHECK-LABEL: "op_after_all"
func.func @op_after_all(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: "mhlo.after_all"([[ARG0:%arg[0-9]+]]) : (!mhlo.token) -> !mhlo.token
  %0 = "stablehlo.after_all"(%arg0) : (!stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "op_all_gather"
func.func @op_all_gather(%arg0: tensor<16x8xf32>) -> tensor<16x16xf32> {
  //               CHECK: "mhlo.all_gather"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   all_gather_dim = 1 : i64,
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   use_global_device_ids
  //          CHECK-SAME: }> : (tensor<16x8xf32>) -> tensor<16x16xf32>
  %0 = "stablehlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    use_global_device_ids
  } : (tensor<16x8xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "op_all_reduce"
func.func @op_all_reduce(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "mhlo.all_reduce"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   use_global_device_ids
  //          CHECK-SAME: }> ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.add"(%[[ARG1]], %[[ARG2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>,
    use_global_device_ids
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_all_reduce_tuple"
func.func @op_all_reduce_tuple(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> (tensor<8xf32>, tensor<f32>) {
  //      CHECK: "mhlo.all_reduce"(%[[ARG0:.*]], %[[ARG1:.*]]) <{replica_groups = dense<> : tensor<0x0xi64>}> ({
  // CHECK-NEXT:   ^bb0(%[[ARG2:.*]]: tensor<f32>, %[[ARG3:.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[ADD:.*]] = "mhlo.add"(%[[ARG2]], %[[ARG3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "mhlo.return"(%[[ADD]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  %0:2 = stablehlo.custom_call @mhlo.all_reduce(%arg0, %arg1) {called_computations = [@all_reduce0], mhlo.attributes = {replica_groups = dense<> : tensor<0x0xi64>}} : (tensor<8xf32>, tensor<f32>) -> (tensor<8xf32>, tensor<f32>)
  return %0#0, %0#1 : tensor<8xf32>, tensor<f32>
}
func.func @all_reduce0(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
  stablehlo.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_all_to_all"
func.func @op_all_to_all(%arg0: tensor<4x16xf32>) -> tensor<16x4xf32> {
  //               CHECK: "mhlo.all_to_all"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
  //          CHECK-SAME:   concat_dimension = 0 : i64,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
  //          CHECK-SAME:   split_count = 4 : i64,
  //          CHECK-SAME:   split_dimension = 1 : i64
  //          CHECK-SAME: }> : (tensor<4x16xf32>) -> tensor<16x4xf32>
  %0 = "stablehlo.all_to_all"(%arg0) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: "op_and"
func.func @op_and(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "mhlo.and"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "op_atan2"
func.func @op_atan2(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.atan2"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.atan2"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_batch_norm_grad"
func.func @op_batch_norm_grad(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "mhlo.batch_norm_grad"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]], [[ARG3:%arg[0-9]+]], [[ARG4:%arg[0-9]+]]) <{
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: }> : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  %0:3 = "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16x16x16x16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_batch_norm_inference"
func.func @op_batch_norm_inference(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16xf32>) -> tensor<16x16x16x16xf32> {
  //      CHECK: "mhlo.batch_norm_inference"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]], [[ARG3:%arg[0-9]+]], [[ARG4:%arg[0-9]+]]) <{
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: }> : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  %0 = "stablehlo.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>, tensor<16xf32>) -> tensor<16x16x16x16xf32>
  func.return %0 : tensor<16x16x16x16xf32>
}

// -----

// CHECK-LABEL: "op_batch_norm_training"
func.func @op_batch_norm_training(%arg0: tensor<16x16x16x16xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) {
  //      CHECK: "mhlo.batch_norm_training"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:   epsilon = 1.000000e-03 : f32,
  // CHECK-SAME:   feature_index = 0 : i64
  // CHECK-SAME: }> : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  %0:3 = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) {
    epsilon = 0.001 : f32,
    feature_index = 0 : i64
  } : (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>) -> (tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>)
  func.return %0#0, %0#1, %0#2 : tensor<16x16x16x16xf32>, tensor<16xf32>, tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_bitcast_convert"
func.func @op_bitcast_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "mhlo.bitcast_convert"([[ARG0:%arg[0-9]+]]) : (tensor<i32>) -> tensor<f32>
  %0 = "stablehlo.bitcast_convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_broadcast_in_dim"
func.func @op_broadcast_in_dim(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "mhlo.broadcast_in_dim"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   broadcast_dimensions = dense<1> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<16xf32>) -> tensor<16x16xf32>
  %0 = "stablehlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = array<i64: 1>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "op_broadcast"
func.func @op_broadcast(%arg0: tensor<16xf32>) -> tensor<16x16xf32> {
  //      CHECK: "mhlo.broadcast"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   broadcast_sizes = dense<16> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<16xf32>) -> tensor<16x16xf32>
  %0 = "stablehlo.broadcast"(%arg0) {
    broadcast_sizes = array<i64: 16>
  } : (tensor<16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "op_case"
func.func @op_case(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.case"([[ARG0:%arg[0-9]+]]) ({
  // CHECK-NEXT:   "mhlo.return"([[ARG1:%arg[0-9]+]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<f32>
  %0 = "stablehlo.case"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_cbrt"
func.func @op_cbrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.cbrt"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.cbrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_ceil"
func.func @op_ceil(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.ceil"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.ceil"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "quantized_op_ceil"
func.func @quantized_op_ceil(%arg0: tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>> {
  // CHECK: "mhlo.ceil"([[ARG0:%arg[0-9]+]]) : (tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
  %0 = "stablehlo.ceil"(%arg0) : (tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
}

// -----

// CHECK-LABEL: "op_cholesky"
func.func @op_cholesky(%arg0: tensor<1x16x16xf32>) -> tensor<1x16x16xf32> {
  //      CHECK: "mhlo.cholesky"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   lower = true
  // CHECK-SAME: }> : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  %0 = "stablehlo.cholesky"(%arg0) {
    lower = true
  } : (tensor<1x16x16xf32>) -> tensor<1x16x16xf32>
  func.return %0 : tensor<1x16x16xf32>
}

// -----

// CHECK-LABEL: "op_clamp"
func.func @op_clamp(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.clamp"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.clamp"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_count_leading_zeros"
func.func @op_count_leading_zeros(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.count_leading_zeros"([[ARG0:%arg[0-9]+]]) : (tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.count_leading_zeros"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_collective_broadcast"
func.func @op_collective_broadcast(%arg0: tensor<1x2xi64>) -> tensor<1x2xi64> {
  //               CHECK: "mhlo.collective_broadcast"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
  //          CHECK-SAME: }> : (tensor<1x2xi64>) -> tensor<1x2xi64>
  %0 = "stablehlo.collective_broadcast"(%arg0) {
    replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<1x2xi64>) -> tensor<1x2xi64>
  func.return %0 : tensor<1x2xi64>
}

// -----

// CHECK-LABEL: "op_collective_permute"
func.func @op_collective_permute(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  //               CHECK: "mhlo.collective_permute"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  //          CHECK-SAME: }> : (tensor<16x8xf32>) -> tensor<16x8xf32>
  %0 = "stablehlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
  } : (tensor<16x8xf32>) -> tensor<16x8xf32>
  func.return %0 : tensor<16x8xf32>
}

// -----

// CHECK-LABEL: "op_compare"
func.func @op_compare(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<i1> {
  //      CHECK: "mhlo.compare"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   compare_type = #mhlo<comparison_type TOTALORDER>,
  // CHECK-SAME:   comparison_direction = #mhlo<comparison_direction EQ>
  // CHECK-SAME: }> : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %0 = "stablehlo.compare"(%arg0, %arg1) {
    comparison_direction = #stablehlo<comparison_direction EQ>,
    compare_type = #stablehlo<comparison_type TOTALORDER>
  } : (tensor<f32>, tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "op_complex"
func.func @op_complex(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<complex<f32>> {
  // CHECK: "mhlo.complex"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  %0 = "stablehlo.complex"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: "op_composite"
func.func @op_composite(%arg0 : tensor<i64>) -> tensor<i64> {
  // CHECK: "mhlo.composite"([[ARG0:%arg[0-9]+]]) <{composite_attributes = {n = 2 : i64}, decomposition = @add_n.impl, name = "stablehlo.add_n"}> : (tensor<i64>) -> tensor<i64>
  %0 = stablehlo.composite "stablehlo.add_n" %arg0 {
    composite_attributes = { n = 2 : i64 },
    decomposition = @add_n.impl
  } : (tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

func.func @add_n.impl(%arg0: tensor<i64>) -> tensor<i64> {
  %0 = stablehlo.constant dense<2> : tensor<i64>
  %1 = stablehlo.add %arg0, %0 : tensor<i64>
  func.return %1 : tensor<i64>
}

// -----

// CHECK-LABEL: "op_concatenate"
func.func @op_concatenate(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.concatenate"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  %0 = "stablehlo.concatenate"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<8xf32>, tensor<8xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_constant"
func.func @op_constant(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.constant"() <{
  // CHECK-SAME:   value = dense<0.000000e+00> : tensor<f32>
  // CHECK-SAME: }> : () -> tensor<f32>
  %0 = "stablehlo.constant"() {
    value = dense<0.0> : tensor<f32>
  } : () -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_convert"
func.func @op_convert(%arg0: tensor<i32>) -> tensor<f32> {
  // CHECK: "mhlo.convert"([[ARG0:%arg[0-9]+]]) : (tensor<i32>) -> tensor<f32>
  %0 = "stablehlo.convert"(%arg0) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_convolution"
func.func @op_convolution(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  //      CHECK: "mhlo.convolution"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  // CHECK-SAME:   feature_group_count = 1 : i64,
  // CHECK-SAME:   lhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   padding = dense<1> : tensor<2x2xi64>,
  // CHECK-SAME:   precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
  // CHECK-SAME:   rhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   window_reversal = dense<false> : tensor<2xi1>,
  // CHECK-SAME:   window_strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  %0 = "stablehlo.convolution"(%arg0, %arg1) {
    window_strides = array<i64: 1, 1>,
    padding = dense<1> : tensor<2x2xi64>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    window_reversal = array<i1: false, false>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

// CHECK-LABEL: "op_cosine"
func.func @op_cosine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.cosine"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.cosine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_create_token"
func.func @op_create_token() -> !stablehlo.token {
  // CHECK: "mhlo.create_token"() : () -> !mhlo.token
  %0 = "stablehlo.create_token"() : () -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "op_cross_replica_sum"
func.func @op_cross_replica_sum(%arg0: tensor<f32>) -> tensor<f32> {
  //               CHECK: "mhlo.cross-replica-sum"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  //          CHECK-SAME: }> : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.cross-replica-sum"(%arg0) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_custom_call_api_version_original"
func.func @called_computation() { func.return }
func.func @op_custom_call_api_version_original(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.custom_call"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   api_version = 1 : i32,
  // CHECK-SAME:   backend_config = "",
  // CHECK-SAME:   call_target_name = "foo",
  // CHECK-SAME:   called_computations = [@foo],
  // CHECK-SAME:   has_side_effect = false,
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>],
  // CHECK-SAME:   output_operand_aliases = [
  // CHECK-SAME:     #mhlo.output_operand_alias<
  // CHECK-SAME:       output_tuple_indices = [],
  // CHECK-SAME:       operand_index = 0,
  // CHECK-SAME:       operand_tuple_indices = []>]
  // CHECK-SAME:   result_layouts = [dense<> : tensor<0xindex>]
  // CHECK-SAME: }> : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo",
    has_side_effect = false,
    backend_config = "",
    api_version = 1 : i32,
    called_computations = [@foo],
    operand_layouts = [dense<> : tensor<0xindex>],
    output_operand_aliases = [
      #stablehlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>],
    result_layouts = [dense<> : tensor<0xindex>]
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_custom_call_api_version_typed_ffi"
func.func @op_custom_call_api_version_typed_ffi(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.custom_call"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   api_version = 4 : i32,
  // CHECK-SAME:   backend_config = {foo = "bar"},
  // CHECK-SAME:   call_target_name = "foo"
  // CHECK-SAME: }> : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.custom_call",
    mhlo.attributes = {api_version = 4 : i32, backend_config = {foo = "bar"}, call_target_name = "foo"},
    mhlo.version = 1 : i64
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: op_custom_call_mhlo_backend_config
func.func @op_custom_call_mhlo_backend_config(%arg0: tensor<16x256xbf16>) -> tensor<16x4xbf16> {
  // CHECK: "mhlo.custom_call"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME: api_version = 4 : i32,
  // CHECK-SAME: backend_config = {aggregate_to_topk = true},
  // CHECK-SAME: call_target_name = "foo"
  // CHECK-SAME: }> : (tensor<16x256xbf16>) -> tensor<16x4xbf16>
  %4 = stablehlo.custom_call @foo(%arg0) {
    "mhlo.backend_config" = {aggregate_to_topk = true}
  } : (tensor<16x256xbf16>) -> tensor<16x4xbf16>
  return %4 : tensor<16x4xbf16>
}

// CHECK-LABEL: "op_divide"
func.func @op_divide(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.divide"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.divide"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_dot_general"
func.func @op_dot_general(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  //      CHECK: "mhlo.dot_general"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_batching_dimensions = [0],
  // CHECK-SAME:     rhs_batching_dimensions = [0],
  // CHECK-SAME:     lhs_contracting_dimensions = [2],
  // CHECK-SAME:     rhs_contracting_dimensions = [1]
  // CHECK-SAME:   >,
  // CHECK-SAME:   precision_config = []
  // CHECK-SAME: }> : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = []
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}

// -----

// CHECK-LABEL: "op_dot_general_algorithm"
func.func @op_dot_general_algorithm(%arg0: tensor<8x8x16xf32>, %arg1: tensor<8x16x8xf32>) -> tensor<8x8x8xf32> {
  //      CHECK: "mhlo.dot_general"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   algorithm = #mhlo.dot_algorithm<
  // CHECK-SAME:     lhs_precision_type = tf32,
  // CHECK-SAME:     rhs_precision_type = tf32,
  // CHECK-SAME:     accumulation_type = f32,
  // CHECK-SAME:     lhs_component_count = 1,
  // CHECK-SAME:     rhs_component_count = 1,
  // CHECK-SAME:     num_primitive_operations = 1,
  // CHECK-SAME:     allow_imprecise_accumulation = false
  // CHECK-SAME:   >,
  // CHECK-SAME:   dot_dimension_numbers = #mhlo.dot<
  // CHECK-SAME:     lhs_batching_dimensions = [0],
  // CHECK-SAME:     rhs_batching_dimensions = [0],
  // CHECK-SAME:     lhs_contracting_dimensions = [2],
  // CHECK-SAME:     rhs_contracting_dimensions = [1]
  // CHECK-SAME:   >
  // CHECK-SAME: }> : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  %0 = "stablehlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #stablehlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    algorithm = #stablehlo.dot_algorithm<
      lhs_precision_type = tf32,
      rhs_precision_type = tf32,
      accumulation_type = f32,
      lhs_component_count = 1,
      rhs_component_count = 1,
      num_primitive_operations = 1,
      allow_imprecise_accumulation = false
    >
  } : (tensor<8x8x16xf32>, tensor<8x16x8xf32>) -> tensor<8x8x8xf32>
  func.return %0 : tensor<8x8x8xf32>
}

// -----

// CHECK-LABEL: "op_dot"
func.func @op_dot(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "mhlo.dot"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   precision_config = []
  // CHECK-SAME: }> : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  %0 = "stablehlo.dot"(%arg0, %arg1) {
    precision_config = []
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_broadcast_in_dim"
func.func @op_dynamic_broadcast_in_dim(%arg0: tensor<?xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  //      CHECK: "mhlo.dynamic_broadcast_in_dim"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   broadcast_dimensions = dense<1> : tensor<1xi64>,
  // CHECK-SAME:   known_expanding_dimensions = dense<> : tensor<0xi64>,
  // CHECK-SAME:   known_nonexpanding_dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %0 = "stablehlo.dynamic_broadcast_in_dim"(%arg0, %arg1) {
    broadcast_dimensions = array<i64: 1>,
    known_expanding_dimensions = array<i64>,
    known_nonexpanding_dimensions = array<i64: 0>
  } : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_conv"
func.func @op_dynamic_conv(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>, %arg2: tensor<2x2xi32>) -> tensor<1x?x?x16xf32> {
  //      CHECK: "mhlo.dynamic_conv"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:   batch_group_count = 1 : i64,
  // CHECK-SAME:   dimension_numbers = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
  // CHECK-SAME:   feature_group_count = 1 : i64,
  // CHECK-SAME:   lhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
  // CHECK-SAME:   rhs_dilation = dense<1> : tensor<2xi64>,
  // CHECK-SAME:   window_reversal = dense<false> : tensor<2xi1>,
  // CHECK-SAME:   window_strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi32>) -> tensor<1x?x?x16xf32>
  %0 = "stablehlo.dynamic_conv"(%arg0, %arg1, %arg2) {
    window_strides = array<i64: 1, 1>,
    lhs_dilation = array<i64: 1, 1>,
    rhs_dilation = array<i64: 1, 1>,
    window_reversal = array<i1: false, false>,
    dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>,
    feature_group_count = 1 : i64,
    batch_group_count = 1 : i64,
    precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]
  } : (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>, tensor<2x2xi32>) -> tensor<1x?x?x16xf32>
  func.return %0 : tensor<1x?x?x16xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_gather"
func.func @op_dynamic_gather(%arg0 : tensor<2x4x9xf32>, %arg1 : tensor<1x5x2xi32>, %arg2 : tensor<3xi32>) -> tensor<1x5x8xf32> {
  //      CHECK: "mhlo.dynamic_gather"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [2],
  // CHECK-SAME:     collapsed_slice_dims = [0, 1],
  // CHECK-SAME:     start_index_map = [0, 1],
  // CHECK-SAME:     index_vector_dim = 2
  // CHECK-SAME:   >,
  // CHECK-SAME:   indices_are_sorted = false
  // CHECK-SAME: }> : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  %0 = "stablehlo.dynamic_gather"(%arg0, %arg1, %arg2) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [2],
      collapsed_slice_dims = [0, 1],
      start_index_map = [0, 1],
      index_vector_dim = 2
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xf32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xf32>
  func.return %0 : tensor<1x5x8xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_iota"
func.func @op_dynamic_iota(%arg0: tensor<1xindex>) -> tensor<?xf32> {
  //      CHECK: "mhlo.dynamic_iota"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   iota_dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<1xindex>) -> tensor<?xf32>
  %0 = "stablehlo.dynamic_iota"(%arg0) {
    iota_dimension = 0 : i64
  } : (tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_pad"
func.func @op_dynamic_pad(%arg0: tensor<?xf32>, %arg1: tensor<f32>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>, %arg4: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "mhlo.dynamic_pad"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]], [[ARG3:%arg[0-9]+]], [[ARG4:%arg[0-9]+]]) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %0 = "stablehlo.dynamic_pad"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?xf32>, tensor<f32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_reshape"
func.func @op_dynamic_reshape(%arg0: tensor<16xf32>, %arg1: tensor<2xindex>) -> tensor<?x?xf32> {
  // CHECK: "mhlo.dynamic_reshape"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<16xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  %0 = "stablehlo.dynamic_reshape"(%arg0, %arg1) : (tensor<16xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_slice"
func.func @op_dynamic_slice(%arg0: tensor<16xf32>, %arg1: tensor<i64>) -> tensor<4xf32> {
  //      CHECK: "mhlo.dynamic_slice"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   slice_sizes = dense<4> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<16xf32>, tensor<i64>) -> tensor<4xf32>
  %0 = "stablehlo.dynamic_slice"(%arg0, %arg1) {
    slice_sizes = array<i64: 4>
  } : (tensor<16xf32>, tensor<i64>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: "op_dynamic_update_slice"
func.func @op_dynamic_update_slice(%arg0: tensor<16xf32>, %arg1: tensor<4xf32>, %arg2: tensor<i64>) -> tensor<16xf32> {
  // CHECK: "mhlo.dynamic_update_slice"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  %0 = "stablehlo.dynamic_update_slice"(%arg0, %arg1, %arg2) : (tensor<16xf32>, tensor<4xf32>, tensor<i64>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_einsum"
func.func @op_einsum(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  //      CHECK: "mhlo.einsum"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   einsum_config = "ab,bc->ac"
  // CHECK-SAME: }> : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  %0 = "stablehlo.einsum"(%arg0, %arg1) {
    einsum_config = "ab,bc->ac"
  } : (tensor<8x16xf32>, tensor<16x8xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: "op_exponential_minus_one"
func.func @op_exponential_minus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.exponential_minus_one"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.exponential_minus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_exponential"
func.func @op_exponential(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.exponential"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.exponential"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_fft"
func.func @op_fft(%arg0: tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>> {
  //      CHECK: "mhlo.fft"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   fft_length = dense<16> : tensor<1xi64>,
  // CHECK-SAME:   fft_type = #mhlo<fft_type FFT>
  // CHECK-SAME: }> : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  %0 = "stablehlo.fft"(%arg0) {
    fft_type = #stablehlo<fft_type FFT>,
    fft_length = array<i64: 16>
  } : (tensor<16xcomplex<f32>>) -> tensor<16xcomplex<f32>>
  func.return %0 : tensor<16xcomplex<f32>>
}

// -----

// CHECK-LABEL: "op_floor"
func.func @op_floor(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.floor"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.floor"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "quantized_op_floor"
func.func @quantized_op_floor(%arg0: tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>> {
  // CHECK: "mhlo.floor"([[ARG0:%arg[0-9]+]]) : (tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
  %0 = "stablehlo.floor"(%arg0) : (tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
}

// -----

// CHECK-LABEL: "op_gather"
func.func @op_gather(%arg0: tensor<2x3x4x2xi32>, %arg1: tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32> {
  //      CHECK: "mhlo.gather"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension_numbers = #mhlo.gather<
  // CHECK-SAME:     offset_dims = [3, 4],
  // CHECK-SAME:     collapsed_slice_dims = [1],
  // CHECK-SAME:     operand_batching_dims = [0],
  // CHECK-SAME:     start_indices_batching_dims = [1],
  // CHECK-SAME:     start_index_map = [2, 1],
  // CHECK-SAME:     index_vector_dim = 3>,
  // CHECK-SAME:   indices_are_sorted = false,
  // CHECK-SAME:   slice_sizes = dense<[1, 1, 2, 2]> : tensor<4xi64>
  // CHECK-SAME: }> : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
  %0 = "stablehlo.gather"(%arg0, %arg1) {
    dimension_numbers = #stablehlo.gather<
      offset_dims = [3, 4],
      collapsed_slice_dims = [1],
      operand_batching_dims = [0],
      start_indices_batching_dims = [1],
      start_index_map = [2, 1],
      index_vector_dim = 3>,
    slice_sizes = array<i64: 1, 1, 2, 2>,
    indices_are_sorted = false
  } : (tensor<2x3x4x2xi32>, tensor<2x2x3x2xi64>) -> tensor<2x2x3x2x2xi32>
  func.return %0 : tensor<2x2x3x2x2xi32>
}

// -----

// CHECK-LABEL: "op_get_dimension_size"
func.func @op_get_dimension_size(%arg0: tensor<?xf32>) -> tensor<i32> {
  //      CHECK: "mhlo.get_dimension_size"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<?xf32>) -> tensor<i32>
  %0 = "stablehlo.get_dimension_size"(%arg0) {
    dimension = 0 : i64
  } : (tensor<?xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_get_tuple_element"
func.func @op_get_tuple_element(%arg0: tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32> {
  //      CHECK: "mhlo.get_tuple_element"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   index = 4 : i32
  // CHECK-SAME: }> : (tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  %0 = "stablehlo.get_tuple_element"(%arg0) {
    index = 4 : i32
  } : (tuple<tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_if"
func.func @op_if(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.if"([[ARG0:%arg[0-9]+]]) ({
  // CHECK-NEXT:   "mhlo.return"([[ARG1:%arg[0-9]+]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   "mhlo.return"([[ARG2:%arg[0-9]+]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<f32>
  %0 = "stablehlo.if"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }, {
    "stablehlo.return"(%arg2) : (tensor<f32>) -> ()
  }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_imag"
func.func @op_imag(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "mhlo.imag"([[ARG0:%arg[0-9]+]]) : (tensor<complex<f32>>) -> tensor<f32>
  %0 = "stablehlo.imag"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_infeed"
func.func @op_infeed(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //               CHECK: "mhlo.infeed"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   infeed_config = "",
  // CHECK-SAME{LITERAL}:   layout = [[]]
  //          CHECK-SAME: }> : (!mhlo.token) -> (tensor<f32>, !mhlo.token)
  %0:2 = "stablehlo.infeed"(%arg0) {
    infeed_config = "",
    layout = [[]]
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}

// -----

// CHECK-LABEL: "op_iota"
func.func @op_iota() -> tensor<16xf32> {
  //      CHECK: "mhlo.iota"() <{
  // CHECK-SAME:   iota_dimension = 0 : i64
  // CHECK-SAME: }> : () -> tensor<16xf32>
  %0 = "stablehlo.iota"() {
    iota_dimension = 0 : i64
  } : () -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_is_finite"
func.func @op_is_finite(%arg0: tensor<f32>) -> tensor<i1> {
  // CHECK: "mhlo.is_finite"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<i1>
  %0 = "stablehlo.is_finite"(%arg0) : (tensor<f32>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "op_log"
func.func @op_log(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.log"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.log"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_log_plus_one"
func.func @op_log_plus_one(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.log_plus_one"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.log_plus_one"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_logistic"
func.func @op_logistic(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.logistic"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.logistic"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_map"
func.func @op_map(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.map"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: }> ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.abs"(%[[ARG1]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "stablehlo.map"(%arg0) ({
    ^bb0(%arg1: tensor<f32>):
      %1 = "stablehlo.abs"(%arg1) : (tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_maximum"
func.func @op_maximum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.maximum"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_minimum"
func.func @op_minimum(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.minimum"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_multiply"
func.func @op_multiply(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.multiply"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_negate"
func.func @op_negate(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.negate"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.negate"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_not"
func.func @op_not(%arg0: tensor<i1>) -> tensor<i1> {
  // CHECK: "mhlo.not"([[ARG0:%arg[0-9]+]]) : (tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.not"(%arg0) : (tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "op_optimization_barrier"
func.func @op_optimization_barrier(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.optimization_barrier"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.optimization_barrier"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_or"
func.func @op_or(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "mhlo.or"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.or"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "op_outfeed"
func.func @op_outfeed(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "mhlo.outfeed"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   outfeed_config = ""
  // CHECK-SAME: }> : (tensor<f32>, !mhlo.token) -> !mhlo.token
  %0 = "stablehlo.outfeed"(%arg0, %arg1) {
    outfeed_config = ""
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "op_pad"
func.func @op_pad(%arg0: tensor<8xf32>, %arg1: tensor<f32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.pad"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   edge_padding_high = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   edge_padding_low = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   interior_padding = dense<0> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  %0 = "stablehlo.pad"(%arg0, %arg1) {
    edge_padding_high = array<i64: 4>,
    edge_padding_low = array<i64: 4>,
    interior_padding = array<i64: 0>
  } : (tensor<8xf32>, tensor<f32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_partition_id"
func.func @op_partition_id() -> tensor<ui32> {
  // CHECK: "mhlo.partition_id"() : () -> tensor<ui32>
  %0 = "stablehlo.partition_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: "op_popcnt"
func.func @op_popcnt(%arg0: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.popcnt"([[ARG0:%arg[0-9]+]]) : (tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.popcnt"(%arg0) : (tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_power"
func.func @op_power(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.power"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.power"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_real_dynamic_slice"
func.func @op_real_dynamic_slice(%arg0: tensor<?xf32>, %arg1: tensor<1xindex>, %arg2: tensor<1xindex>, %arg3: tensor<1xindex>) -> tensor<?xf32> {
  // CHECK: "mhlo.real_dynamic_slice"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]], [[ARG3:%arg[0-9]+]]) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  %0 = "stablehlo.real_dynamic_slice"(%arg0, %arg1, %arg2, %arg3) : (tensor<?xf32>, tensor<1xindex>, tensor<1xindex>, tensor<1xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: "op_real"
func.func @op_real(%arg0: tensor<complex<f32>>) -> tensor<f32> {
  // CHECK: "mhlo.real"([[ARG0:%arg[0-9]+]]) : (tensor<complex<f32>>) -> tensor<f32>
  %0 = "stablehlo.real"(%arg0) : (tensor<complex<f32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_recv"
func.func @op_recv(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //      CHECK: "mhlo.recv"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 3>,
  // CHECK-SAME:   is_host_transfer = true
  // CHECK-SAME: }> : (!mhlo.token) -> (tensor<f32>, !mhlo.token)
  %0:2 = "stablehlo.recv"(%arg0) {
    source_target_pairs = dense<[]> : tensor<0xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 3>,
    is_host_transfer = true
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}

// CHECK-LABEL: "op_recv_d2d"
func.func @op_recv_d2d(%arg0: !stablehlo.token) -> (tensor<f32>, !stablehlo.token) {
  //      CHECK: "mhlo.recv"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 1>,
  // CHECK-SAME:   is_host_transfer = false
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // CHECK-SAME: }> : (!mhlo.token) -> (tensor<f32>, !mhlo.token)
  %0:2 = "stablehlo.recv"(%arg0) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
    is_host_transfer = false,
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (!stablehlo.token) -> (tensor<f32>, !stablehlo.token)
  func.return %0#0, %0#1 : tensor<f32>, !stablehlo.token
}

// -----

// CHECK-LABEL: "op_reduce"
func.func @op_reduce(%arg0: tensor<16xf32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "stablehlo.reduce"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    dimensions = array<i64: 0>
  } : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_reduce_precision"
func.func @op_reduce_precision(%arg0: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.reduce_precision"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   exponent_bits = 8 : i32,
  // CHECK-SAME:   mantissa_bits = 10 : i32
  // CHECK-SAME: }> : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.reduce_precision"(%arg0) {
    exponent_bits = 8 : i32,
    mantissa_bits = 10 : i32
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_reduce_scatter"
func.func @op_reduce_scatter(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //               CHECK: "mhlo.reduce_scatter"([[ARG0:%arg[0-9]+]]) <{
  //          CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 0>,
  // CHECK-SAME{LITERAL}:   replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
  //          CHECK-SAME:   scatter_dimension = 0 : i64
  //          CHECK-SAME: }> ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.add"(%[[ARG1]], %[[ARG2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "stablehlo.reduce_scatter"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.add"(%arg1, %arg2) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
    scatter_dimension = 0 : i64
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_reduce_window"
func.func @op_reduce_window(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x5x8x7xf32> {
  //               CHECK: "mhlo.reduce_window"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  //          CHECK-SAME:   base_dilations = dense<1> : tensor<4xi64>,
  // CHECK-SAME{LITERAL}:   padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
  //          CHECK-SAME:   window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  //          CHECK-SAME:   window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  //          CHECK-SAME:   window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>
  //          CHECK-SAME: }> ({
  //          CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: tensor<f32>, %[[ARG3:arg.*]]: tensor<f32>):
  //          CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.maximum"(%[[ARG2]], %[[ARG3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  //          CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<f32>) -> ()
  //          CHECK-NEXT: }) : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x5x8x7xf32>
  %0 = "stablehlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %1 = "stablehlo.maximum"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 4, 4, 1>,
    base_dilations = array<i64: 1, 1, 1, 1>,
    window_dilations = array<i64: 1, 2, 2, 1>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>
  } : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x5x8x7xf32>
  func.return %0 : tensor<2x5x8x7xf32>
}

// -----

// CHECK-LABEL: "op_remainder"
func.func @op_remainder(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.remainder"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.remainder"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_replica_id"
func.func @op_replica_id() -> tensor<ui32> {
  // CHECK: "mhlo.replica_id"() : () -> tensor<ui32>
  %0 = "stablehlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: "op_reshape"
func.func @op_reshape(%arg0: tensor<16xf32>) -> tensor<4x4xf32> {
  // CHECK: "mhlo.reshape"([[ARG0:%arg[0-9]+]]) : (tensor<16xf32>) -> tensor<4x4xf32>
  %0 = "stablehlo.reshape"(%arg0) : (tensor<16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: "op_reshape_dynamic"
func.func @op_reshape_dynamic(%arg0: tensor<?x1xi64, #mhlo.type_extensions<bounds = [7, ?]>>) -> tensor<7xi64> {
  // CHECK: "mhlo.reshape"({{.*}}) : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [7, ?]>>) -> tensor<7xi64>
  %0 = "stablehlo.reshape"(%arg0) : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [7, ?]>>) -> tensor<7xi64>
  return %0 : tensor<7xi64>
}

// -----

// CHECK-LABEL: "op_return"
func.func @op_return(%arg0: tensor<i32>, %arg1: tensor<f32>) -> tensor<f32> {
  //      CHECK: "mhlo.case"([[ARG0:%arg[0-9]+]]) ({
  // CHECK-NEXT:   "mhlo.return"([[ARG1:%arg[0-9]+]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<i32>) -> tensor<f32>
  %0 = "stablehlo.case"(%arg0) ({
    "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_reverse"
func.func @op_reverse(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.reverse"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimensions = dense<0> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "stablehlo.reverse"(%arg0) {
    dimensions = array<i64: 0>
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_rng_bit_generator"
func.func @op_rng_bit_generator(%arg0: tensor<f32>) -> (tensor<f32>, tensor<f32>) {
  //      CHECK: "mhlo.rng_bit_generator"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   rng_algorithm = #mhlo.rng_algorithm<PHILOX>
  // CHECK-SAME: }> : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  %0:2 = "stablehlo.rng_bit_generator"(%arg0) {
    rng_algorithm = #stablehlo<rng_algorithm PHILOX>
  } : (tensor<f32>) -> (tensor<f32>, tensor<f32>)
  func.return %0#0, %0#1 : tensor<f32>, tensor<f32>
}

// -----

// CHECK-LABEL: "op_rng"
func.func @op_rng(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<0xindex>) -> tensor<f32> {
  //      CHECK: "mhlo.rng"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:   rng_distribution = #mhlo.rng_distribution<NORMAL>
  // CHECK-SAME: }> : (tensor<f32>, tensor<f32>, tensor<0xindex>) -> tensor<f32>
  %0 = "stablehlo.rng"(%arg0, %arg1, %arg2) {
    rng_distribution = #stablehlo<rng_distribution NORMAL>
  } : (tensor<f32>, tensor<f32>, tensor<0xindex>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_round_nearest_afz"
func.func @op_round_nearest_afz(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.round_nearest_afz"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.round_nearest_afz"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_round_nearest_even"
func.func @op_round_nearest_even(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.round_nearest_even"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.round_nearest_even"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_rsqrt"
func.func @op_rsqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.rsqrt"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.rsqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_scatter"
func.func @op_scatter(%arg0: tensor<2x3x4x2xi64>, %arg1: tensor<2x2x3x2xi64>, %arg2: tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64> {
  //      CHECK: "mhlo.scatter"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:  indices_are_sorted = true,
  // CHECK-SAME:  scatter_dimension_numbers = #mhlo.scatter<
  // CHECK-SAME:    update_window_dims = [3, 4],
  // CHECK-SAME:    inserted_window_dims = [1],
  // CHECK-SAME:    input_batching_dims = [0],
  // CHECK-SAME:    scatter_indices_batching_dims = [1],
  // CHECK-SAME:    scatter_dims_to_operand_dims = [2, 1],
  // CHECK-SAME:    index_vector_dim = 3>,
  // CHECK-SAME:  unique_indices = true
  // CHECK-SAME: }> ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG3:arg.*]]: tensor<i64>, %[[ARG4:arg.*]]: tensor<i64>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.add"(%[[ARG3]], %[[ARG4]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<i64>) -> ()
  // CHECK-NEXT: }) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
  %0 = "stablehlo.scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<i64>, %arg4: tensor<i64>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%1) : (tensor<i64>) -> ()
  }) {
    scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [3, 4],
      inserted_window_dims = [1],
      input_batching_dims = [0],
      scatter_indices_batching_dims = [1],
      scatter_dims_to_operand_dims = [2, 1],
      index_vector_dim = 3>,
    indices_are_sorted = true,
    unique_indices = true
  } : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
  func.return %0 : tensor<2x3x4x2xi64>
}

// -----

// CHECK-LABEL: "op_select_and_scatter"
func.func @op_select_and_scatter(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
  //      CHECK: "mhlo.select_and_scatter"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) <{
  // CHECK-SAME:   padding = dense<0> : tensor<4x2xi64>,
  // CHECK-SAME:   window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
  // CHECK-SAME:   window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  // CHECK-SAME: }> ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG31:arg.*]]: tensor<f32>, %[[ARG41:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL11:.*]] = "mhlo.compare"(%[[ARG31]], %[[ARG41]]) <{compare_type = #mhlo<comparison_type TOTALORDER>, comparison_direction = #mhlo<comparison_direction GE>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:     "mhlo.return"(%[[VAL11]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG32:arg.*]]: tensor<f32>, %[[ARG42:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL12:.*]] = "mhlo.add"(%[[ARG32]], %[[ARG42]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-NEXT:     "mhlo.return"(%[[VAL12]]) : (tensor<f32>) -> ()
  // CHECK-NEXT: }) : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  %0 = "stablehlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.compare"(%arg3, %arg4) {compare_type = #stablehlo<comparison_type TOTALORDER>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }, {
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %1 = "stablehlo.add"(%arg3, %arg4) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%1) : (tensor<f32>) -> ()
  }) {
    window_dimensions = array<i64: 1, 2, 2, 1>,
    window_strides = array<i64: 1, 2, 2, 1>,
    padding = dense<0> : tensor<4x2xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  func.return %0 : tensor<10x24x24x64xf32>
}

// -----

// CHECK-LABEL: "op_select"
func.func @op_select(%arg0: tensor<i1>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.select"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]], [[ARG2:%arg[0-9]+]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_send"
func.func @op_send(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "mhlo.send"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 2>,
  // CHECK-SAME:   is_host_transfer = true
  // CHECK-SAME: }> : (tensor<f32>, !mhlo.token) -> !mhlo.token
  %0 = "stablehlo.send"(%arg0, %arg1) {
    source_target_pairs = dense<[]> : tensor<0xi64>,
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 2>,
    is_host_transfer = true
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "op_send_d2d"
func.func @op_send_d2d(%arg0: tensor<f32>, %arg1: !stablehlo.token) -> !stablehlo.token {
  //      CHECK: "mhlo.send"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   channel_handle = #mhlo.channel_handle<handle = 0, type = 1>,
  // CHECK-SAME:   is_host_transfer = false
  // CHECK-SAME{LITERAL}:   source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  // CHECK-SAME: }> : (tensor<f32>, !mhlo.token) -> !mhlo.token
  %0 = "stablehlo.send"(%arg0, %arg1) {
    channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
    is_host_transfer = false,
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<f32>, !stablehlo.token) -> !stablehlo.token
  func.return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "op_set_dimension_size"
func.func @op_set_dimension_size(%arg0: tensor<?xf32>, %arg1: tensor<i32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.set_dimension_size"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension = 0 : i64
  // CHECK-SAME: }> : (tensor<?xf32>, tensor<i32>) -> tensor<16xf32>
  %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) {
    dimension = 0 : i64
  } : (tensor<?xf32>, tensor<i32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_shift_left"
func.func @op_shift_left(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.shift_left"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.shift_left"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_shift_right_arithmetic"
func.func @op_shift_right_arithmetic(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.shift_right_arithmetic"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.shift_right_arithmetic"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_shift_right_logical"
func.func @op_shift_right_logical(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.shift_right_logical"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.shift_right_logical"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "op_sign"
func.func @op_sign(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.sign"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.sign"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_sine"
func.func @op_sine(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.sine"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.sine"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_slice"
func.func @op_slice(%arg0: tensor<16xf32>) -> tensor<4xf32> {
  //      CHECK: "mhlo.slice"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   limit_indices = dense<4> : tensor<1xi64>,
  // CHECK-SAME:   start_indices = dense<0> : tensor<1xi64>,
  // CHECK-SAME:   strides = dense<1> : tensor<1xi64>
  // CHECK-SAME: }> : (tensor<16xf32>) -> tensor<4xf32>
  %0 = "stablehlo.slice"(%arg0) {
    start_indices = array<i64: 0>,
    limit_indices = array<i64: 4>,
    strides = array<i64: 1>
  } : (tensor<16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: "op_sort"
func.func @op_sort(%arg0: tensor<16xf32>) -> tensor<16xf32> {
  //      CHECK: "mhlo.sort"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   dimension = 0 : i64,
  // CHECK-SAME:   is_stable = true
  // CHECK-SAME: }> ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  // CHECK-NEXT:     %[[VAL1:.*]] = "mhlo.compare"(%[[ARG1]], %[[ARG2]]) <{compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-NEXT:     "mhlo.return"(%[[VAL1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "stablehlo.sort"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = "stablehlo.compare"(%arg1, %arg2) {compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "stablehlo.return"(%1) : (tensor<i1>) -> ()
  }) {
    dimension = 0 : i64,
    is_stable = true
  } : (tensor<16xf32>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

// CHECK-LABEL: "op_sqrt"
func.func @op_sqrt(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.sqrt"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.sqrt"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_subtract"
func.func @op_subtract(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.subtract"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.subtract"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_tan_mhlo_v1"
func.func @op_tan_mhlo_v1(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.tan"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.tan",
    mhlo.attributes = {},
    mhlo.version = 1 : i64
  } : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_tanh"
func.func @op_tanh(%arg0: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.tanh"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.tanh"(%arg0) : (tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_torch_index_select"
func.func @op_torch_index_select(%arg0: tensor<5x1x5xf32>, %arg1: tensor<2xi32>) ->  tensor<2x1x5xf32> {
  //      CHECK: "mhlo.torch_index_select"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   batch_dims = 0 : i64,
  // CHECK-SAME:   dim = 0 : i64
  // CHECK-SAME: }> : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  %0 = "stablehlo.torch_index_select"(%arg0, %arg1) {
    dim = 0 : i64,
    batch_dims = 0 : i64
  } : (tensor<5x1x5xf32>, tensor<2xi32>) -> tensor<2x1x5xf32>
  func.return %0 : tensor<2x1x5xf32>
}

// -----

// CHECK-LABEL: "op_transpose"
func.func @op_transpose(%arg0: tensor<16x8xf32>) ->  tensor<8x16xf32> {
  //      CHECK: "mhlo.transpose"([[ARG0:%arg[0-9]+]]) <{
  // CHECK-SAME:   permutation = dense<[1, 0]> : tensor<2xi64>
  // CHECK-SAME: }> : (tensor<16x8xf32>) -> tensor<8x16xf32>
  %0 = "stablehlo.transpose"(%arg0) {
    permutation = array<i64: 1, 0>
  } : (tensor<16x8xf32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
}

// -----

// CHECK-LABEL: "op_triangular_solve"
func.func @op_triangular_solve(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>) ->  tensor<16x16xf32> {
  //      CHECK: "mhlo.triangular_solve"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) <{
  // CHECK-SAME:   left_side = true,
  // CHECK-SAME:   lower = true,
  // CHECK-SAME:   transpose_a = #mhlo<transpose NO_TRANSPOSE>,
  // CHECK-SAME:   unit_diagonal = true
  // CHECK-SAME: }> : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  %0 = "stablehlo.triangular_solve"(%arg0, %arg1) {
    left_side = true,
    lower = true,
    unit_diagonal = true,
    transpose_a = #stablehlo<transpose NO_TRANSPOSE>
  } : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: "op_tuple"
func.func @op_tuple(%arg0: tensor<f32>) -> tuple<tensor<f32>> {
  // CHECK: "mhlo.tuple"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tuple<tensor<f32>>
  %0 = "stablehlo.tuple"(%arg0) : (tensor<f32>) -> tuple<tensor<f32>>
  func.return %0 : tuple<tensor<f32>>
}

// -----

// CHECK-LABEL: "op_uniform_dequantize"
func.func @op_uniform_dequantize(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32> {
  // CHECK: "mhlo.uniform_dequantize"([[ARG0:%arg[0-9]+]]) : (tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>) -> tensor<f32>
  %0 = "stablehlo.uniform_dequantize"(%arg0) : (tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_uniform_quantize"
func.func @op_uniform_quantize(%arg0: tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>> {
  // CHECK: "mhlo.uniform_quantize"([[ARG0:%arg[0-9]+]]) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>
  %0 = "stablehlo.uniform_quantize"(%arg0) : (tensor<f32>) -> tensor<!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL: "op_while"
func.func @op_while(%arg0: tensor<i1>) -> tensor<i1> {
  //      CHECK: "mhlo.while"([[ARG0:%arg[0-9]+]]) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<i1>):
  // CHECK-NEXT:     "mhlo.return"(%[[ARG1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<i1>):
  // CHECK-NEXT:     "mhlo.return"(%[[ARG1]]) : (tensor<i1>) -> ()
  // CHECK-NEXT: }) : (tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.while"(%arg0) ({
    ^bb0(%arg1: tensor<i1>):
      "stablehlo.return"(%arg1) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg1: tensor<i1>):
      "stablehlo.return"(%arg1) : (tensor<i1>) -> ()
  }) : (tensor<i1>) -> tensor<i1>
  func.return %0: tensor<i1>
}

// -----

// CHECK-LABEL: "op_xor"
func.func @op_xor(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "mhlo.xor"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.xor"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// ============ BOUNDED DYNAMISM ============

// CHECK-LABEL: bounded_dynamism_reshape
func.func @bounded_dynamism_reshape(%arg0: tensor<?x1xi64, #stablehlo.bounds<7, ?>>) -> tensor<?xi64, #stablehlo.bounds<7>> {
  // CHECK: mhlo.reshape{{.*}}tensor<?xi64, #mhlo.type_extensions<bounds = [7]>>
  %0 = stablehlo.reshape %arg0 : (tensor<?x1xi64, #stablehlo.bounds<7, ?>>) -> tensor<?xi64, #stablehlo.bounds<7>>
  return %0 : tensor<?xi64, #stablehlo.bounds<7>>
}

// -----

// CHECK-LABEL: bounded_dynamism_broadcast_in_dim
func.func @bounded_dynamism_broadcast_in_dim(%arg0: tensor<1x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<2x1x?xf32, #stablehlo.bounds<?, ?, 5>> {
  // CHECK: mhlo.broadcast_in_dim{{.*}}tensor<2x1x?xf32, #mhlo.type_extensions<bounds = [?, ?, 5]>>
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x?xf32, #stablehlo.bounds<?, 5>>) -> tensor<2x1x?xf32, #stablehlo.bounds<?, ?, 5>>
  return %0 : tensor<2x1x?xf32, #stablehlo.bounds<?, ?, 5>>
}

// CHECK-LABEL: bounded_dynamism_with_unknown_op
func.func @bounded_dynamism_with_unknown_op(%arg0: tensor<1x4xi32>, %arg1: tensor<i32>) -> tensor<1x4xi32> {
  %0 = "stablehlo.set_dimension_size"(%arg0, %arg1) <{dimension = 1 : i64}> : (tensor<1x4xi32>, tensor<i32>) -> tensor<1x?xi32, #stablehlo.bounds<?, 4>>
  // CHECK: "tensor.cast"({{.*}}) : (tensor<1x?xi32, #mhlo.type_extensions<bounds = [?, 4]>>) -> tensor<1x4xi32>
  %cast = tensor.cast %0 : tensor<1x?xi32, #stablehlo.bounds<?, 4>> to tensor<1x4xi32>
  return %cast : tensor<1x4xi32>
}

// ============ TYPES ============

// CHECK-LABEL: "type_i1"
func.func @type_i1(%arg0: tensor<i1>, %arg1: tensor<i1>) -> tensor<i1> {
  // CHECK: "mhlo.and"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  %0 = "stablehlo.and"(%arg0, %arg1) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  func.return %0 : tensor<i1>
}

// -----

// CHECK-LABEL: "type_i4"
func.func @type_i4(%arg0: tensor<i4>, %arg1: tensor<i4>) -> tensor<i4> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i4>, tensor<i4>) -> tensor<i4>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i4>, tensor<i4>) -> tensor<i4>
  func.return %0 : tensor<i4>
}

// -----

// CHECK-LABEL: "type_i8"
func.func @type_i8(%arg0: tensor<i8>, %arg1: tensor<i8>) -> tensor<i8> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
  func.return %0 : tensor<i8>
}

// -----

// CHECK-LABEL: "type_i16"
func.func @type_i16(%arg0: tensor<i16>, %arg1: tensor<i16>) -> tensor<i16> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
  func.return %0 : tensor<i16>
}

// -----

// CHECK-LABEL: "type_i32"
func.func @type_i32(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: "type_i64"
func.func @type_i64(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
  func.return %0 : tensor<i64>
}

// -----

// CHECK-LABEL: "type_ui4"
func.func @type_ui4(%arg0: tensor<ui4>, %arg1: tensor<ui4>) -> tensor<ui4> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<ui4>, tensor<ui4>) -> tensor<ui4>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui4>, tensor<ui4>) -> tensor<ui4>
  func.return %0 : tensor<ui4>
}

// -----

// CHECK-LABEL: "type_ui8"
func.func @type_ui8(%arg0: tensor<ui8>, %arg1: tensor<ui8>) -> tensor<ui8> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
  func.return %0 : tensor<ui8>
}

// -----

// CHECK-LABEL: "type_ui16"
func.func @type_ui16(%arg0: tensor<ui16>, %arg1: tensor<ui16>) -> tensor<ui16> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
  func.return %0 : tensor<ui16>
}

// -----

// CHECK-LABEL: "type_ui32"
func.func @type_ui32(%arg0: tensor<ui32>, %arg1: tensor<ui32>) -> tensor<ui32> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: "type_ui64"
func.func @type_ui64(%arg0: tensor<ui64>, %arg1: tensor<ui64>) -> tensor<ui64> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
  func.return %0 : tensor<ui64>
}

// -----

// CHECK-LABEL: "type_f4E2M1FN"
func.func @type_f4E2M1FN(%arg0: tensor<f4E2M1FN>, %arg1: tensor<f4E2M1FN>) -> tensor<f4E2M1FN> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f4E2M1FN>, tensor<f4E2M1FN>) -> tensor<f4E2M1FN>
  func.return %0 : tensor<f4E2M1FN>
}

// -----

// CHECK-LABEL: "type_f6E2M3FN"
func.func @type_f6E2M3FN(%arg0: tensor<f6E2M3FN>, %arg1: tensor<f6E2M3FN>) -> tensor<f6E2M3FN> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f6E2M3FN>, tensor<f6E2M3FN>) -> tensor<f6E2M3FN>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f6E2M3FN>, tensor<f6E2M3FN>) -> tensor<f6E2M3FN>
  func.return %0 : tensor<f6E2M3FN>
}

// -----

// CHECK-LABEL: "type_f6E3M2FN"
func.func @type_f6E3M2FN(%arg0: tensor<f6E3M2FN>, %arg1: tensor<f6E3M2FN>) -> tensor<f6E3M2FN> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f6E3M2FN>, tensor<f6E3M2FN>) -> tensor<f6E3M2FN>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f6E3M2FN>, tensor<f6E3M2FN>) -> tensor<f6E3M2FN>
  func.return %0 : tensor<f6E3M2FN>
}

// -----

// CHECK-LABEL: "type_f8E3M4"
func.func @type_f8E3M4(%arg0: tensor<f8E3M4>, %arg1: tensor<f8E3M4>) -> tensor<f8E3M4> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E3M4>, tensor<f8E3M4>) -> tensor<f8E3M4>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E3M4>, tensor<f8E3M4>) -> tensor<f8E3M4>
  func.return %0 : tensor<f8E3M4>
}

// -----

// CHECK-LABEL: "type_f8E4M3"
func.func @type_f8E4M3(%arg0: tensor<f8E4M3>, %arg1: tensor<f8E4M3>) -> tensor<f8E4M3> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E4M3>, tensor<f8E4M3>) -> tensor<f8E4M3>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3>, tensor<f8E4M3>) -> tensor<f8E4M3>
  func.return %0 : tensor<f8E4M3>
}

// -----

// CHECK-LABEL: "type_f8E4M3FN"
func.func @type_f8E4M3FN(%arg0: tensor<f8E4M3FN>, %arg1: tensor<f8E4M3FN>) -> tensor<f8E4M3FN> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3FN>, tensor<f8E4M3FN>) -> tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}

// -----

// CHECK-LABEL: "type_f8E4M3FNUZ"
func.func @type_f8E4M3FNUZ(%arg0: tensor<f8E4M3FNUZ>, %arg1: tensor<f8E4M3FNUZ>) -> tensor<f8E4M3FNUZ> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E4M3FNUZ>, tensor<f8E4M3FNUZ>) -> tensor<f8E4M3FNUZ>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3FNUZ>, tensor<f8E4M3FNUZ>) -> tensor<f8E4M3FNUZ>
  func.return %0 : tensor<f8E4M3FNUZ>
}

// -----

// CHECK-LABEL: "type_f8E4M3B11FNUZ"
func.func @type_f8E4M3B11FNUZ(%arg0: tensor<f8E4M3B11FNUZ>, %arg1: tensor<f8E4M3B11FNUZ>) -> tensor<f8E4M3B11FNUZ> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E4M3B11FNUZ>, tensor<f8E4M3B11FNUZ>) -> tensor<f8E4M3B11FNUZ>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E4M3B11FNUZ>, tensor<f8E4M3B11FNUZ>) -> tensor<f8E4M3B11FNUZ>
  func.return %0 : tensor<f8E4M3B11FNUZ>
}

// -----

// CHECK-LABEL: "type_f8E5M2"
func.func @type_f8E5M2(%arg0: tensor<f8E5M2>, %arg1: tensor<f8E5M2>) -> tensor<f8E5M2> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E5M2>, tensor<f8E5M2>) -> tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}

// -----

// CHECK-LABEL: "type_f8E5M2FNUZ"
func.func @type_f8E5M2FNUZ(%arg0: tensor<f8E5M2FNUZ>, %arg1: tensor<f8E5M2FNUZ>) -> tensor<f8E5M2FNUZ> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E5M2FNUZ>, tensor<f8E5M2FNUZ>) -> tensor<f8E5M2FNUZ>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E5M2FNUZ>, tensor<f8E5M2FNUZ>) -> tensor<f8E5M2FNUZ>
  func.return %0 : tensor<f8E5M2FNUZ>
}

// -----

// CHECK-LABEL: "type_f8E8M0FNU"
func.func @type_f8E8M0FNU(%arg0: tensor<f8E8M0FNU>, %arg1: tensor<f8E8M0FNU>) -> tensor<f8E8M0FNU> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f8E8M0FNU>, tensor<f8E8M0FNU>) -> tensor<f8E8M0FNU>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f8E8M0FNU>, tensor<f8E8M0FNU>) -> tensor<f8E8M0FNU>
  func.return %0 : tensor<f8E8M0FNU>
}

// -----

// CHECK-LABEL: "type_bf16"
func.func @type_bf16(%arg0: tensor<bf16>, %arg1: tensor<bf16>) -> tensor<bf16> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
  func.return %0 : tensor<bf16>
}

// -----

// CHECK-LABEL: "type_f16"
func.func @type_f16(%arg0: tensor<f16>, %arg1: tensor<f16>) -> tensor<f16> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
  func.return %0 : tensor<f16>
}

// -----

// CHECK-LABEL: "type_f32"
func.func @type_f32(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "type_f64"
func.func @type_f64(%arg0: tensor<f64>, %arg1: tensor<f64>) -> tensor<f64> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
  func.return %0 : tensor<f64>
}

// -----

// CHECK-LABEL: "type_complex_f32"
func.func @type_complex_f32(%arg0: tensor<complex<f32>>, %arg1: tensor<complex<f32>>) -> tensor<complex<f32>> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>>
  func.return %0 : tensor<complex<f32>>
}

// -----

// CHECK-LABEL: "type_complex_f64"
func.func @type_complex_f64(%arg0: tensor<complex<f64>>, %arg1: tensor<complex<f64>>) -> tensor<complex<f64>> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG1:%arg[0-9]+]]) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
  %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<complex<f64>>, tensor<complex<f64>>) -> tensor<complex<f64>>
  func.return %0 : tensor<complex<f64>>
}

// -----

// CHECK-LABEL: "type_dynamism_ranked"
func.func @type_dynamism_ranked(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: "mhlo.abs"([[ARG0:%arg[0-9]+]]) : (tensor<?xf32>) -> tensor<?xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: "type_quantization"
func.func @type_quantization(%arg0: tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<!quant.uniform<i8:f32, 34.0:16>> {
  // CHECK: "mhlo.add"([[ARG0:%arg[0-9]+]], [[ARG0]]) : (tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>, tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>) -> tensor<!quant.uniform<i8:f32, 3.400000e+01:16>>
  %0 = "stablehlo.add"(%arg0, %arg0) : (tensor<!quant.uniform<i8:f32, 34.0:16>>, tensor<!quant.uniform<i8:f32, 34.0:16>>) -> tensor<!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<!quant.uniform<i8:f32, 34.0:16>>
}

// -----

#SV = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>

// CHECK: #[[$SV:.*]] = #sparse_tensor.encoding<{ map = (d0) -> (d0 : compressed) }>
// CHECK-LABEL: "type_sparsity"
func.func @type_sparsity(%arg0: tensor<16xf32, #SV>) -> tensor<16xf32> {
  // CHECK: "mhlo.abs"([[ARG0:%arg[0-9]+]]) : (tensor<16xf32, #[[$SV]]>) -> tensor<16xf32>
  %0 = "stablehlo.abs"(%arg0) : (tensor<16xf32, #SV>) -> tensor<16xf32>
  func.return %0 : tensor<16xf32>
}

// -----

func.func @type_token_callee(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: function_type = (!mhlo.token) -> !mhlo.token, sym_name = "type_token_callee"
  // CHECK: "func.return"([[ARG0:%arg[0-9]+]]) : (!mhlo.token) -> ()
  return %arg0 : !stablehlo.token
}

func.func @type_token_caller(%arg0: !stablehlo.token) -> !stablehlo.token {
  // CHECK: function_type = (!mhlo.token) -> !mhlo.token, sym_name = "type_token_caller"
  // CHECK: "func.call"([[ARG0:%arg[0-9]+]]) <{callee = @type_token_callee}> : (!mhlo.token) -> !mhlo.token
  %0 = func.call @type_token_callee(%arg0) : (!stablehlo.token) -> !stablehlo.token
  return %0 : !stablehlo.token
}

// -----

// CHECK-LABEL: "type_token_region"
func.func @type_token_region(%arg0: tensor<i1>, %arg1: !stablehlo.token) {
  //      CHECK: "mhlo.while"([[ARG1:%arg[0-9]+]]) ({
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !mhlo.token):
  // CHECK-NEXT:     "mhlo.return"([[ARG0:%arg[0-9]+]]) : (tensor<i1>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   ^[[BB:bb.*]](%[[ARG2:arg.*]]: !mhlo.token):
  // CHECK-NEXT:     "mhlo.return"(%[[ARG2]]) : (!mhlo.token) -> ()
  // CHECK-NEXT: }) : (!mhlo.token) -> !mhlo.token
  %0 = "stablehlo.while"(%arg1) ({
    ^bb0(%arg2: !stablehlo.token):
      stablehlo.return %arg0 : tensor<i1>
    }, {
    ^bb0(%arg2: !stablehlo.token):
      stablehlo.return %arg2 : !stablehlo.token
  }) : (!stablehlo.token) -> !stablehlo.token
  return
}

// -----

// CHECK-LABEL: "type_tuple"
func.func @type_tuple(%arg0: tuple<tensor<f32>>) -> tuple<!stablehlo.token> {
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "foo"
  // CHECK: (tuple<tensor<f32>>) -> tuple<!mhlo.token>
  } : (tuple<tensor<f32>>) -> tuple<!stablehlo.token>
  return %0 : tuple<!stablehlo.token>
}

// -----

// ============ NEGATIVE TESTS ============
// Some ops, attributes and types used in StableHLO programs are not supported in MHLO.
// For those cases, we have negative tests below.

func.func @op_custom_call_botched_extensibility_protocol(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.custom_call",
    backend_config = "{api_version = 4 : i32, backend_config = {foo = \22bar\22}, call_target_name = \22foo\22}",
    has_side_effect = false
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @op_custom_call_botched_mhlo_backend_config(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.custom_call",
    mhlo.backend_config = "."
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @op_custom_call_botched_mhlo_backend_config(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.custom_call",
    mhlo.backend_config = 3
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @op_custom_call_botched_mhlo_backend_config_version(%arg0: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{failed to legalize operation 'stablehlo.custom_call' that was explicitly marked illegal}}
  %0 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.custom_call",
    api_version = 2 : i32,
    mhlo.backend_config = 3
  } : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: "op_topk_mhlo_v1"
func.func @op_topk_mhlo_v1(%arg0: tensor<5x10xf32>) -> (tensor<5x8xf32>, tensor<5x8xi32>) {
  // CHECK: "mhlo.topk"([[ARG0:%arg[0-9]+]]) <{k = 8 : i64, largest = true}> : (tensor<5x10xf32>) -> (tensor<5x8xf32>, tensor<5x8xi32>)
  %0:2 = "stablehlo.custom_call"(%arg0) {
    call_target_name = "mhlo.topk",
    mhlo.attributes = {k = 8 : i64, largest = true},
    mhlo.version = 1 : i64
  } : (tensor<5x10xf32>) -> (tensor<5x8xf32>, tensor<5x8xi32>)
  func.return %0#0, %0#1 : tensor<5x8xf32>, tensor<5x8xi32>
}

// -----

func.func @op_unary_einsum_deprecated(%arg0: tensor<8x16xf32>) -> tensor<8xf32> {
  // expected-error@+2 {{failed to legalize operation 'stablehlo.unary_einsum' that was explicitly marked illegal}}
  // expected-error@+1 {{UnaryEinsumOp is deprecated and not supported in MHLO}}
  %0 = "stablehlo.unary_einsum"(%arg0) {
    einsum_config = "ab->a"
  } : (tensor<8x16xf32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}
