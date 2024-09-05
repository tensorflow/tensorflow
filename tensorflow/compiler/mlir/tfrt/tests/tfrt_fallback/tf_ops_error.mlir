// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd --test_init_function=register_op_handlers 2>&1 | FileCheck %s --dump-input=fail

func.func @register_op_handlers() {
  %fallback = "corert.create_kernel_fallback_op_handler"() : () -> !corert.ophandler
  corert.register_op_handler %fallback "tfkernel"
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_parse_example_v2_error'
func.func @test_parse_example_v2_error() -> !tfrt_fallback.tf_tensor {
  %serialized_th = corert.const_string_tensor {shape = [2, 1], value = ["", ""]}
  %names_th = corert.const_string_tensor {shape = [0], value = []}
  %dense_keys_th = corert.const_string_tensor {shape = [7], value = ["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]}

  %ch = tfrt.new.chain
  %ch0 = tfrt_fallback_async.createop(%ch) key(0) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} num_args(0)
  %ch1 = tfrt_fallback_async.createop(%ch0) key(1) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<i64>} num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(2) device("CPU:0") "tf.ParseExampleV2"()
        {Tdense = [i64, i64, i64, i64, i64, i64, i64], dense_shapes = [#corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>], num_sparse = 0 : i64, ragged_split_types = [], ragged_value_types = [], sparse_types = []}
        num_args(12)

  %ch3, %zero = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} : 1
  %ch4, %one = tfrt_fallback_async.executeop.seq(%ch2) key(1) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<i64>} : 1

  %serialized, %names, %dense_keys = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %serialized_th, %names_th, %dense_keys_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  // expected-error @+1 {{Expected serialized to be a scalar or vector, got shape: [2,1]}}
  %dense_values:7 = tfrt_fallback_async.executeop key(2) cost(100) device("CPU:0")
        "tf.ParseExampleV2"(%serialized, %names, %names, %dense_keys, %names, %zero, %one, %zero, %one, %zero, %one, %zero)
        {Tdense = [i64, i64, i64, i64, i64, i64, i64], dense_shapes = [#corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>], num_sparse = 0 : i64, ragged_split_types = [], ragged_value_types = [], sparse_types = []} : 7

  tfrt.return %dense_values#0: !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: --- Running 'test_assign_variable_error'
func.func @test_assign_variable_error() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %th0 = corert.const_dense_tensor dense<1> : tensor<i32>
  %th1 = corert.const_dense_tensor dense<1.0> : tensor<f32>
  %0, %1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %th0, %th1 {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(3) device("CPU:0") "tf.VarHandleOp"() {container = "", dtype = f32, shape = #corert.shape<>, shared_name = "x"} num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(4) device("CPU:0") "tf.AssignVariableOp"() {dtype = f32} num_args(2)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(5) device("CPU:0") "tf.AssignVariableOp"() {dtype = i32} num_args(2)

  %ch4, %2 = tfrt_fallback_async.executeop.seq(%ch3) key(3) cost(100) device("CPU:0") "tf.VarHandleOp"() {container = "", dtype = f32, shape = #corert.shape<>, shared_name = "x"} : 1
  %ch5 = tfrt_fallback_async.executeop.seq(%ch4) key(4) cost(100) device("CPU:0") "tf.AssignVariableOp"(%2, %1) {dtype = f32}
  // expected-error @+1 {{Trying to assign variable with wrong dtype. Expected float got int32}}
  %ch6 = tfrt_fallback_async.executeop.seq(%ch5) key(5) cost(100) device("CPU:0") "tf.AssignVariableOp"(%2, %0) {dtype = i32}
  tfrt.return %ch6 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_ref_args'
func.func @test_ref_args() {
  %ch0 = tfrt.new.chain
  // expected-error @+1 {{Unsupported ref args in VariableV2}}
  %ch1 = tfrt_fallback_async.createop(%ch0) key(6) device("CPU:0") "tf.VariableV2"() {container = "", dtype = f32, shape = #corert.shape<>, shared_name = "x"} num_args(0)
  %ch2, %0 = tfrt_fallback_async.executeop.seq(%ch1) key(6) cost(100) device("CPU:0") "tf.VariableV2"() {container = "", dtype = f32, shape = #corert.shape<>, shared_name = "x"} : 1
  tfrt.return
}
