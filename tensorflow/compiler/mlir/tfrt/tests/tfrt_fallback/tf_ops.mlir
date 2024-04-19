// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor --work_queue_type=mstd --test_init_function=register_op_handlers 2>&1 | FileCheck %s

func.func @register_op_handlers() -> (!tfrt.chain, !tfrt.chain, !tfrt.chain) {
  %fallback = "corert.create_kernel_fallback_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%fallback) : (!corert.ophandler) -> !corert.ophandler
  %ch0 = corert.register_op_handler %fallback "tfkernel"
  %cpu_identity = "tfrt_test.identity"(%ch0, %cpu) : (!tfrt.chain, !corert.ophandler) -> !corert.ophandler
  %ch1 = corert.register_op_handler %cpu_identity "cpu"
  %ch2 = "corert.add_kernel_fallback_implicit_conversions" (%cpu_identity) : (!corert.ophandler) -> !tfrt.chain
  tfrt.return %ch0, %ch1, %ch2 : !tfrt.chain, !tfrt.chain, !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_tf_random_uniform'
func.func @test_tf_random_uniform() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("CPU:0") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("CPU:0") "tf.RandomUniform"() {T = i64, dtype = f32} num_args(1)

  %ch3, %shape = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("CPU:0") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } : 1
  %random_uniform = tfrt_fallback_async.executeop key(1) cost(100) device("CPU:0") "tf.RandomUniform"(%shape) {T = i64, dtype = f32} : 1

  // CHECK: Tensor<type: float shape: [2,2]
  %ch4 = "tfrt_fallback_async.print_tensor"(%random_uniform, %ch3)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch4 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_tf_vars'
func.func @test_tf_vars() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(2) device("CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(3) device("CPU:0") "tf.VarHandleOp" () { dtype = f32, shape = #corert.shape<2x2>, container = "c", shared_name = "v0"} num_args(0)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(4) device("CPU:0") "tf.AssignVariableOp" () { dtype = f32 } num_args(2)
  %ch4 = tfrt_fallback_async.createop(%ch3) key(5) device("CPU:0") "tf.ReadVariableOp" () { dtype = f32 } num_args(1)

  %ch5, %val = tfrt_fallback_async.executeop.seq(%ch4) key(2) cost(100) device("CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } : 1
  %ch6, %varh = tfrt_fallback_async.executeop.seq(%ch4) key(3) cost(100) device("CPU:0") "tf.VarHandleOp" () { dtype = f32, shape = #corert.shape<2x2>, container = "c", shared_name = "v0"} : 1
  %ch7 = tfrt_fallback_async.executeop.seq(%ch6) key(4) cost(100) device("CPU:0") "tf.AssignVariableOp" (%varh, %val) { dtype = f32 } : 0
  %ch8, %val2 = tfrt_fallback_async.executeop.seq(%ch7) key(5) cost(100) device("CPU:0") "tf.ReadVariableOp" (%varh) { dtype = f32 } : 1

  // CHECK: Tensor<type: float shape: [2,2]
  %ch9 = "tfrt_fallback_async.print_tensor"(%val2, %ch8)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch9 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_parse_example_v2'
func.func @test_parse_example_v2() -> !tfrt.chain {
  %ch = tfrt.new.chain

  %serialized_th = corert.const_string_tensor {shape = [1], value = [""]}
  %names_th = corert.const_string_tensor {shape = [0], value = []}
  %dense_keys_th = corert.const_string_tensor {shape = [7], value = ["has_login_page_feature", "num_terms_inside_postform", "num_terms_outside_postform", "num_terms_outside_postform_without_bp", "query_params_contains_url", "title_with_login_phase", "url_contains_login_terms"]}


  %ch0 = tfrt_fallback_async.createop(%ch) key(6) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} num_args(0)
  %ch1 = tfrt_fallback_async.createop(%ch0) key(7) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<i64>} num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(8) device("CPU:0") "tf.ParseExampleV2"()
        {Tdense = [i64, i64, i64, i64, i64, i64, i64], dense_shapes = [#corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>], num_sparse = 0 : i64, ragged_split_types = [], ragged_value_types = [], sparse_types = []}
        num_args(12)

  %ch3, %zero = tfrt_fallback_async.executeop.seq(%ch2) key(6) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} : 1
  %ch4, %one = tfrt_fallback_async.executeop.seq(%ch2) key(7) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<i64>} : 1

  %serialized, %names, %dense_keys = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %serialized_th, %names_th, %dense_keys_th
    {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)
  %dense_values:7 = tfrt_fallback_async.executeop key(8) cost(100) device("CPU:0")
        "tf.ParseExampleV2"(%serialized, %names, %names, %dense_keys, %names, %zero, %one, %zero, %one, %zero, %one, %zero)
        {Tdense = [i64, i64, i64, i64, i64, i64, i64], dense_shapes = [#corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>, #corert.shape<>], num_sparse = 0 : i64, ragged_split_types = [], ragged_value_types = [], sparse_types = []} : 7

  // CHECK: Tensor<type: int64 shape: [1] values: 0>
  // CHECK: Tensor<type: int64 shape: [1] values: 1>
  // CHECK: Tensor<type: int64 shape: [1] values: 0>
  // CHECK: Tensor<type: int64 shape: [1] values: 1>
  // CHECK: Tensor<type: int64 shape: [1] values: 0>
  // CHECK: Tensor<type: int64 shape: [1] values: 1>
  // CHECK: Tensor<type: int64 shape: [1] values: 0>
  %ch5 = "tfrt_fallback_async.print_tensor"(%dense_values#0, %ch4)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch6 = "tfrt_fallback_async.print_tensor"(%dense_values#1, %ch5)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch7 = "tfrt_fallback_async.print_tensor"(%dense_values#2, %ch6)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch8 = "tfrt_fallback_async.print_tensor"(%dense_values#3, %ch7)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch9 = "tfrt_fallback_async.print_tensor"(%dense_values#4, %ch8)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch10 = "tfrt_fallback_async.print_tensor"(%dense_values#5, %ch9)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch11 = "tfrt_fallback_async.print_tensor"(%dense_values#6, %ch10)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch11 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_tile'
func.func @test_tile() -> !tfrt.chain {
  %ch = tfrt.new.chain

  %input_th = corert.const_string_tensor {shape = [1, 2], value = ["", ""]}
  %multiples_th = corert.const_dense_tensor dense<[7,3]> : tensor<2xi32>

  %input, %multiples = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %input_th, %multiples_th {_tfrt_cost = 1 : i64, device = "CPU:0"}
    : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch0 = tfrt_fallback_async.createop(%ch) key(9) device("CPU:0") "tf.Tile"() {T = !corert.string, Tmultiples = i32} num_args(2)
  %ch1, %output = tfrt_fallback_async.executeop.seq(%ch0) key(9) cost(100) device("CPU:0") "tf.Tile"(%input, %multiples) {T = !corert.string, Tmultiples = i32} : 1

  // CHECK: Tensor<type: string shape: [7,6]
  %ch2 = "tfrt_fallback_async.print_tensor"(%output, %ch1)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'tfrt_test_benchmark_lifetime_regression_test'
func.func @tfrt_test_benchmark_lifetime_regression_test() {
  %ch_0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch_0 "cpu"

  %tensor_0 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"() { shape = [1 : i64], values = [1.0 : f32] } : 1
  %tensor_1 = corert.executeop(%cpu) "tf.VarHandleOp" (){ dtype = f32, shape = #corert.shape<1>, container = "c", shared_name = "v_1" } : 1
  %ch_1 = corert.executeop.seq(%cpu, %ch_0) "tf.AssignVariableOp" (%tensor_1, %tensor_0) { dtype = f32 } : 0

  tfrt_test.benchmark "tfrt_test_benchmark_lifetime_regression_test"(
    %ch_0 : !tfrt.chain,
    %ch_1 : !tfrt.chain,
    %tensor_1 : !corert.tensorhandle,
    %cpu : !corert.ophandler
  )
  duration_secs = 1, max_count = 1, num_warmup_runs = 0
  {
    %ch_3_1, %tensor_2 = corert.executeop.seq(%cpu, %ch_1) "tf.ReadVariableOp" (%tensor_1) { dtype = f32 } : 1

    tfrt.return %ch_0 : !tfrt.chain
  }
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_quantized'
func.func @test_quantized() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(2) device("CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<1x1xf32> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(3) device("CPU:0") "tf.QuantizeV2" () { T = !corert.quint8} num_args(3)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(4) device("CPU:0") "tf.QuantizeV2" () { T = !corert.quint16} num_args(3)
  %ch4 = tfrt_fallback_async.createop(%ch3) key(5) device("CPU:0") "tf.QuantizeV2" () { T = !corert.qint8} num_args(3)
  %ch5 = tfrt_fallback_async.createop(%ch4) key(6) device("CPU:0") "tf.QuantizeV2" () { T = !corert.qint16} num_args(3)
  %ch6 = tfrt_fallback_async.createop(%ch5) key(7) device("CPU:0") "tf.QuantizeV2" () { T = !corert.qint32} num_args(3)

  %ch7, %val = tfrt_fallback_async.executeop.seq(%ch6) key(2) cost(100) device("CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<1x1xf32> } : 1
  %ch8, %val1:3 = tfrt_fallback_async.executeop.seq(%ch7) key(3) cost(100) device("CPU:0") "tf.QuantizeV2" (%val, %val, %val) { T = !corert.quint8} : 3
  %ch9, %val2:3 = tfrt_fallback_async.executeop.seq(%ch8) key(4) cost(100) device("CPU:0") "tf.QuantizeV2" (%val, %val, %val) { T = !corert.quint16} : 3
  %ch10, %val3:3 = tfrt_fallback_async.executeop.seq(%ch9) key(5) cost(100) device("CPU:0") "tf.QuantizeV2" (%val, %val, %val) { T = !corert.qint8} : 3
  %ch11, %val4:3 = tfrt_fallback_async.executeop.seq(%ch10) key(6) cost(100) device("CPU:0") "tf.QuantizeV2" (%val, %val, %val) { T = !corert.qint16} : 3
  %ch12, %val5:3 = tfrt_fallback_async.executeop.seq(%ch11) key(7) cost(100) device("CPU:0") "tf.QuantizeV2" (%val, %val, %val) { T = !corert.qint32} : 3

  // CHECK: Tensor<type: quint8 shape: [1,1]
  %ch13 = "tfrt_fallback_async.print_tensor"(%val1#0, %ch12)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: quint16 shape: [1,1]
  %ch14 = "tfrt_fallback_async.print_tensor"(%val2#0, %ch13)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: qint8 shape: [1,1]
  %ch15 = "tfrt_fallback_async.print_tensor"(%val3#0, %ch14)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: qint16 shape: [1,1]
  %ch16 = "tfrt_fallback_async.print_tensor"(%val4#0, %ch15)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: qint32 shape: [1,1]
  %ch17 = "tfrt_fallback_async.print_tensor"(%val5#0, %ch16)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch17 : !tfrt.chain
}
