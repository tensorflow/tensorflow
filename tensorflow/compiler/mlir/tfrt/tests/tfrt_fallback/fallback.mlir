// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor --test_init_function=register_op_handlers_kernel | FileCheck %s

func.func @register_op_handlers_kernel() {
  %fallback = "corert.create_kernel_fallback_op_handler"() : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%fallback) : (!corert.ophandler) -> !corert.ophandler
  %cpu_ordinal = tfrt.constant.i32 1
  %cpu1 = "corert.create_cpu_op_handler_with_ordinal"(%fallback, %cpu_ordinal) : (!corert.ophandler, i32) -> !corert.ophandler
  corert.register_op_handler %cpu "cpu"
  corert.register_op_handler %cpu1 "cpu"
  corert.register_op_handler %fallback "tfkernel"
  tfrt.return
}

// CHECK-LABEL: --- Running 'test_tf_random_uniform'
func.func @test_tf_random_uniform() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("CPU:0") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("CPU:0") "tf.RandomUniform"() {dtype = f32, T = i64} num_args(1)

  // Test tf.RandomUniform.
  %ch, %shape_th = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("CPU:0") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } : 1
  %random_uniform = tfrt_fallback_async.executeop key(1) cost(100) device("CPU:0") "tf.RandomUniform"(%shape_th) {dtype = f32, T = i64} : 1

  // CHECK: Tensor<type: float shape: [2,2] values:
  %ch3 = "tfrt_fallback_async.print_tensor"(%random_uniform, %ch2)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

func.func @test_tf_random_uniform_on_cpu1() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("CPU:1") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("CPU:1") "tf.RandomUniform"() {dtype = f32, T = i64} num_args(1)

  // Test tf.RandomUniform.
  %ch, %shape_th = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("CPU:1") "tf.Const"()
      { dtype = i64, value = dense<[2,2]> : tensor<2xi64> } : 1
  %random_uniform = tfrt_fallback_async.executeop key(1) cost(100) device("CPU:1") "tf.RandomUniform"(%shape_th) {dtype = f32, T = i64} : 1

  %res_th = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %random_uniform {_tfrt_cost = 1 : i64, device = "CPU:1"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)

  // CHECK: float shape: [2,2] values:
  %ch3 = "corert.print_tensorhandle"(%res_th, %ch) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_parse_single_example'
func.func @test_parse_single_example() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  // %serialized is a tensorflow::Example proto whose content is
  //   features {
  //     feature {
  //       key: "key_0"
  //       value {
  //         int64_list {
  //           value: 100
  //         }
  //       }
  //     }
  //     feature {
  //       key: "key_1"
  //       value {
  //         int64_list {
  //           value: 200
  //         }
  //       }
  //     }
  //   }
  //
  // Note that in MLIR, non-printable bytes are printed as escaped hex numbers.
  %serialized_th = corert.const_string_tensor {shape = [], value = ["\0A!\0A\0E\0A\05key_0\12\05\1A\03\0A\01d\0A\0F\0A\05key_1\12\06\1A\04\0A\02\C8\01"]}
  %serialized = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %serialized_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(2) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<1xi64>} num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(3) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<1xi64>} num_args(0)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(4) device("CPU:0") "tf.ParseSingleExample"()
        {Tdense = [i64, i64], dense_keys = ["key_0", "key_1"], num_sparse = 0 : i64, sparse_types = [], sparse_keys = [], dense_shapes = [#corert.shape<?>, #corert.shape<?>]} num_args(3)

  %ch4, %zero = tfrt_fallback_async.executeop.seq(%ch3) key(2) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<1xi64>} : 1
  %ch5, %one = tfrt_fallback_async.executeop.seq(%ch3) key(3) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<1> : tensor<1xi64>} : 1
  %dense_values:2 = tfrt_fallback_async.executeop key(4) cost(100) device("CPU:0") "tf.ParseSingleExample"(%serialized, %zero, %one)
        {Tdense = [i64, i64], dense_keys = ["key_0", "key_1"], num_sparse = 0 : i64, sparse_types = [], sparse_keys = [], dense_shapes = [#corert.shape<?>, #corert.shape<?>]} : 2

  // CHECK: Tensor<type: int64 shape: [1] values: 100>
  %ch6 = "tfrt_fallback_async.print_tensor"(%dense_values#0, %ch5)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: int64 shape: [1] values: 200>
  %ch7 = "tfrt_fallback_async.print_tensor"(%dense_values#1, %ch6)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch7 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_const_tensor_proto'
func.func @test_const_tensor_proto() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_tensor_proto "\08\0C\12\00\22\01@"

  // CHECK: Tensor<type: quint8 shape: [] values: 64>
  %ch1 = "tfrt_fallback_async.print_tensor"(%a, %ch0)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_const_dense_tensor'
func.func @test_const_dense_tensor() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_dense_tensor dense<[true, false]> : tensor<2xi1> {_tfrt_cost = 1 : i64}

  // CHECK: Tensor<type: bool shape: [2] values: 1 0>
  %ch1 = "tfrt_fallback_async.print_tensor"(%a, %ch0)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_const_string_tensor'
func.func @test_const_string_tensor() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_string_tensor {shape = [2], value = ["const", "string"], _tfrt_cost = 1 : i64}

  // CHECK: Tensor<type: string shape: [2] values: const string>
  %ch1 = "tfrt_fallback_async.print_tensor"(%a, %ch0)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_const_string_tensor_same_value'
func.func @test_const_string_tensor_same_value() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a = tfrt_fallback_async.const_string_tensor {shape = [2], value = ["string"], _tfrt_cost = 1 : i64}

  // CHECK: Tensor<type: string shape: [2] values: string string>
  %ch1 = "tfrt_fallback_async.print_tensor"(%a, %ch0)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_predicate'
func.func @test_predicate() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"

  %a = tfrt_fallback_async.const_dense_tensor dense<0.0> : tensor<f32> {_tfrt_cost = 1 : i64}
  %ra = tfrt_fallback_async.predicate %a {_tfrt_cost = 1 : i64}
  // CHECK: false
  %ch1 = "tfrt_test.print_bool"(%ch0, %ra) : (!tfrt.chain, i1) -> !tfrt.chain

  %b = tfrt_fallback_async.const_dense_tensor dense<1> : tensor<i32> {_tfrt_cost = 1 : i64}
  %rb = tfrt_fallback_async.predicate %b {_tfrt_cost = 1 : i64}
  // CHECK: true
  %ch2 = "tfrt_test.print_bool"(%ch1, %rb) : (!tfrt.chain, i1) -> !tfrt.chain

  %c = tfrt_fallback_async.const_string_tensor {shape = [], value = [""], _tfrt_cost = 1 : i64}
  %rc = tfrt_fallback_async.predicate %c {_tfrt_cost = 1 : i64}
  // CHECK: false
  %ch3 = "tfrt_test.print_bool"(%ch2, %rc) : (!tfrt.chain, i1) -> !tfrt.chain

  %d = tfrt_fallback_async.const_string_tensor {shape = [], value = ["string"], _tfrt_cost = 1 : i64}
  %rd = tfrt_fallback_async.predicate %d {_tfrt_cost = 1 : i64}
  // CHECK: true
  %ch4 = "tfrt_test.print_bool"(%ch3, %rd) : (!tfrt.chain, i1) -> !tfrt.chain

  %e = tfrt_fallback_async.const_dense_tensor dense<[]> : tensor<0xi32> {_tfrt_cost = 1 : i64}
  %re = tfrt_fallback_async.predicate %e {_tfrt_cost = 1 : i64}
  // CHECK: false
  %ch5 = "tfrt_test.print_bool"(%ch4, %re) : (!tfrt.chain, i1) -> !tfrt.chain

  %f = tfrt_fallback_async.const_dense_tensor dense<[0]> : tensor<1xi32> {_tfrt_cost = 1 : i64}
  %rf = tfrt_fallback_async.predicate %f {_tfrt_cost = 1 : i64}
  // CHECK: true
  %ch6 = "tfrt_test.print_bool"(%ch5, %rf) : (!tfrt.chain, i1) -> !tfrt.chain

  tfrt.return %ch6 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_fallback_resource'
func.func @test_fallback_resource() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a_th = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %b_th = corert.const_dense_tensor dense<[false, true]> : tensor<2xi1>
  %ra, %rb = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a_th, %b_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.set_resource %ch0, %ra {device = "cpu", index = 0 : i64}
  %ch2 = tfrt_fallback_async.set_resource %ch1, %rb {device = "cpu", index = 1 : i64}
  %ch3, %b, %a = tfrt_fallback_async.get_resource %ch2 {_tfrt_cost = 1 : i64, device = "cpu", indices = [1 : i64, 0 : i64]} : (!tfrt.chain) -> (!tfrt.chain, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  // CHECK: Tensor<type: bool shape: [2] values: 1 0>
  %ch4 = "tfrt_fallback_async.print_tensor"(%a, %ch3)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: bool shape: [2] values: 0 1>
  %ch5 = "tfrt_fallback_async.print_tensor"(%b, %ch4)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch5 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_i1_dtype'
func.func @test_i1_dtype() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %a_th = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %b_th = corert.const_dense_tensor dense<[false, true]> : tensor<2xi1>
  %a, %b = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a_th, %b_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(5) device("CPU:0") "tf.LogicalAnd"() num_args(2)
  %ch2, %c = tfrt_fallback_async.executeop.seq(%ch1) key(5) cost(100) device("CPU:0") "tf.LogicalAnd"(%a, %b) : 1

  // CHECK: Tensor<type: bool shape: [2] values: 0 0>
  %ch3 = "tfrt_fallback_async.print_tensor"(%c, %ch2)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch3 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_step_container'
func.func @test_step_container() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %size_th = corert.const_dense_tensor dense<5> : tensor<i32>
  %index_th = corert.const_dense_tensor dense<1> : tensor<i32>
  %value_th = corert.const_dense_tensor dense<[1.0, 2.0, 3.0]> : tensor<3xf32>

  %size, %index, %value  = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %size_th, %index_th, %value_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(6) device("CPU:0") "tf.TensorArrayV3"() {dtype = f32, element_shape = #corert.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} num_args(1)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(7) device("CPU:0") "tf.TensorArrayWriteV3"() {T = f32} num_args(4)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(8) device("CPU:0") "tf.TensorArrayReadV3"() {dtype = f32} num_args(3)
  %ch4 = tfrt_fallback_async.createop(%ch3) key(9) device("CPU:0") "tf.TensorArrayCloseV3"() num_args(1)

  %ch5, %ta:2 = tfrt_fallback_async.executeop.seq(%ch4) key(6) cost(100) device("CPU:0") "tf.TensorArrayV3"(%size) {dtype = f32, element_shape = #corert.shape<3>, dynamic_size = false, clear_after_read = true, identical_element_shapes = true, tensor_array_name = "ta"} : 2
  %ch6, %write = tfrt_fallback_async.executeop.seq(%ch5) key(7) cost(100) device("CPU:0") "tf.TensorArrayWriteV3"(%ta#0, %index, %value, %ta#1) {T = f32} : 1
  %ch7, %read = tfrt_fallback_async.executeop.seq(%ch6) key(8) cost(100) device("CPU:0") "tf.TensorArrayReadV3"(%ta#0, %index, %write) {dtype = f32} : 1
  %ch8 = tfrt_fallback_async.executeop.seq(%ch7) key(9) cost(100) device("CPU:0") "tf.TensorArrayCloseV3"(%ta#0)

  // CHECK: Tensor<type: float shape: [3] values: 1 2 3>
  %ch9 = "tfrt_fallback_async.print_tensor"(%read, %ch8)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch9 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_select'
func.func @test_select() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %arg_th = corert.const_dense_tensor dense<[1, 2, 3, 4]> : tensor<4xi64>
  %cond_th = corert.const_dense_tensor dense<[true, false, true, false]> : tensor<4xi1>
  %arg, %cond = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %arg_th, %cond_th {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(10) device("CPU:0") "tf.ZerosLike"() {T = i64} num_args(1)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(11) device("CPU:0") "tf.Select"() {T = i64} num_args(3)

  %ch3, %zeros = tfrt_fallback_async.executeop.seq(%ch2) key(10) cost(100) device("CPU:0") "tf.ZerosLike"(%arg) {T = i64} : 1
  %0 = tfrt_fallback_async.executeop key(11) cost(100) device("CPU:0") "tf.Select"(%cond, %arg, %zeros) {T = i64} : 1

  // CHECK: Tensor<type: int64 shape: [4] values: 1 0 3...>
  %ch4 = "tfrt_fallback_async.print_tensor"(%0, %ch3)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

func.func @while_cond_false(%in: !tfrt.chain, %x: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %ch0 = tfrt_fallback_async.createop(%in) key(12) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} num_args(0)
  %ch1, %zero = tfrt_fallback_async.executeop.seq(%ch0) key(12) cost(100) device("CPU:0") "tf.Const"() {dtype = i64, value = dense<0> : tensor<i64>} : 1
  %zero_th = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %zero {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)

  tfrt.return %ch1, %zero_th : !tfrt.chain, !corert.tensorhandle
}

func.func @while_body_print(%in: !tfrt.chain, %x: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %ch0 = tfrt.new.chain

  %ch1 = "corert.print_tensorhandle"(%x, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %in, %x : !tfrt.chain, !corert.tensorhandle
}

// While loop test
// CHECK-LABEL: --- Running 'while_fallback_condition'
func.func @while_fallback_condition() {
  %ch0 = tfrt.new.chain

  %cpu = corert.get_op_handler %ch0 "cpu"

  %one = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1, 1], values = [1 : i32] } : 1

  // CHECK-NOT: DenseHostTensor dtype = i32, shape = [1, 1], values = [1]
  %ch1, %result = corert.while @while_cond_false @while_body_print (%ch0, %one) : (!corert.tensorhandle) -> (!corert.tensorhandle)

  tfrt.return
}

// CHECK-LABEL: --- Running 'test_copy_if_small'
func.func @test_copy_if_small() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %small = tfrt_fallback_async.const_dense_tensor dense<[1, 2, 3, 4]> : tensor<4xi64> {_tfrt_cost = 1 : i64}
  %small_copies:2 = tfrt_fallback_async.copy_if_small %small {_tfrt_cost = 1 : i64} : (!tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  // CHECK: Tensor<type: int64 shape: [4] values: 1 2 3...>
  // CHECK: Tensor<type: int64 shape: [4] values: 1 2 3...>
  %ch1 = "tfrt_fallback_async.print_tensor"(%small_copies#0, %ch0) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch2 = "tfrt_fallback_async.print_tensor"(%small_copies#1, %ch1) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  %large = tfrt_fallback_async.const_dense_tensor dense<100> : tensor<128xi64> {_tfrt_cost = 1 : i64}
  %large_copies:2 = tfrt_fallback_async.copy_if_small %large {_tfrt_cost = 1 : i64} : (!tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  // CHECK: Tensor<type: int64 shape: [128] values: 100 100 100...>
  // CHECK: Tensor<type: int64 shape: [128] values: 100 100 100...>
  %ch3 = "tfrt_fallback_async.print_tensor"(%large_copies#0, %ch2) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch4 = "tfrt_fallback_async.print_tensor"(%large_copies#1, %ch3) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  %string = tfrt_fallback_async.const_string_tensor {shape = [1], value = ["string"], _tfrt_cost = 1 : i64}
  %string_copies:2 = tfrt_fallback_async.copy_if_small %string {_tfrt_cost = 1 : i64} : (!tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  // CHECK: Tensor<type: string shape: [1] values: string>
  // CHECK: Tensor<type: string shape: [1] values: string>
  %ch5 = "tfrt_fallback_async.print_tensor"(%string_copies#0, %ch4) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  %ch6 = "tfrt_fallback_async.print_tensor"(%string_copies#1, %ch5) : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch4 : !tfrt.chain
}

func.func @branch0(%ch0: !tfrt.chain, %arg0: !corert.tensorhandle, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch0 "cpu"
  %res = corert.executeop(%cpu) "tfrt_test.add"(%arg0, %arg1) : 1
  tfrt.return %ch0, %res : !tfrt.chain, !corert.tensorhandle
}

func.func @branch1(%ch0: !tfrt.chain, %arg0: !corert.tensorhandle, %arg1: !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle) {
  %cpu = corert.get_op_handler %ch0 "cpu"
  %th = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [4 : i32] } : 1
  %add0 = corert.executeop(%cpu) "tfrt_test.add"(%arg0, %arg1) : 1
  %res = corert.executeop(%cpu) "tfrt_test.add"(%add0, %th) : 1
  tfrt.return %ch0, %res : !tfrt.chain, !corert.tensorhandle
}

// CHECK-LABEL: --- Running 'test_case_fallback_index'
func.func @test_case_fallback_index() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %cpu = corert.get_op_handler %ch0 "cpu"

  %ch1 = tfrt_fallback_async.createop(%ch0) key(13) device("CPU:0") "tf.Const"() {dtype = i32, value = dense<0> : tensor<i32>} num_args(0)
  %ch2, %fallback_idx0 = tfrt_fallback_async.executeop.seq(%ch1) key(13) cost(100) device("CPU:0") "tf.Const"() {dtype = i32, value = dense<0> : tensor<i32>} : 1
  %fallback_idx_th0 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %fallback_idx0 {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)
  %idx0 = corert.tensorhandle_to_int32 %fallback_idx_th0

  %ch5 = tfrt_fallback_async.createop(%ch0) key(14) device("CPU:0") "tf.Const"() {dtype = i32, value = dense<1> : tensor<i32>} num_args(0)
  %ch6, %fallback_idx1 = tfrt_fallback_async.executeop.seq(%ch5) key(14) cost(100) device("CPU:0") "tf.Const"() {dtype = i32, value = dense<1> : tensor<i32>} : 1
  %fallback_idx_th1 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %fallback_idx1 {_tfrt_cost = 1 : i64, device = "CPU:0"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)
  %idx1 = corert.tensorhandle_to_int32 %fallback_idx_th1

  %arg0 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [2 : i32] } : 1
  %arg1 = corert.executeop(%cpu)
    "tfrt_test.create_dense_tensor"() { shape = [1], values = [4 : i32] } : 1

  %ch3, %res0 = tfrt.case %idx0 [@branch0, @branch1] (%ch0, %arg0, %arg1) : (!tfrt.chain, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)

  // CHECK: DenseHostTensor dtype = i32, shape = [1], values = [6]
  %ch4 = "corert.print_tensorhandle"(%res0, %ch3) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  %ch7, %res1 = tfrt.case %idx1 [@branch0, @branch1] (%ch4, %arg0, %arg1) : (!tfrt.chain, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt.chain, !corert.tensorhandle)

  // CHECK: DenseHostTensor dtype = i32, shape = [1], values = [10]
  %ch8 = "corert.print_tensorhandle"(%res1, %ch7) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch8 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_fallback_to_host'
func.func @test_fallback_to_host() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %ch1 = tfrt_fallback_async.createop(%ch0) key(2) device("/CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(3) device("/CPU:0") "tf.VarHandleOp" () { dtype = f32, shape = #corert.shape<2x2>, container = "c", shared_name = "v0"} num_args(0)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(4) device("/CPU:0") "tf.AssignVariableOp" () { dtype = f32 } num_args(2)
  %ch4 = tfrt_fallback_async.createop(%ch3) key(5) device("/TPU:0") "tf.ReadVariableOp" () { dtype = f32 } num_args(1)

  %ch5, %val = tfrt_fallback_async.executeop.seq(%ch4) key(2) cost(100) device("/CPU:0") "tf.Const" () { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } : 1
  %ch6, %varh = tfrt_fallback_async.executeop.seq(%ch4) key(3) cost(100) device("/CPU:0") "tf.VarHandleOp" () { dtype = f32, shape = #corert.shape<2x2>, container = "c", shared_name = "v0"} : 1
  %ch7 = tfrt_fallback_async.executeop.seq(%ch6) key(4) cost(100) device("/CPU:0") "tf.AssignVariableOp" (%varh, %val) { dtype = f32 } : 0
  %ch8, %val2 = tfrt_fallback_async.executeop.seq(%ch7) key(5) cost(100) device("/TPU:0") "tf.ReadVariableOp" (%varh) { dtype = f32 } : 1

  // CHECK: Tensor<type: float shape: [2,2]
  %ch9 = "tfrt_fallback_async.print_tensor"(%val2, %ch8)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch9 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_custom_allocator'
func.func @test_custom_allocator() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %allocator = "tfrt_fallback_async.get_test_allocator"() : () -> (!tfrt_fallback.tf_allocator)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("/CPU:0") "tf.Cast"() { SrcT = i32, DstT = f32 } num_args(1)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("/CPU:0") "tf.Cast"() { SrcT = f32, DstT = i32 } num_args(1)
  %ch3 = tfrt_fallback_async.createop(%ch2) key(2) device("/CPU:0") "tf.Const"() { dtype = i32, value = dense<123> : tensor<i32> } num_args(0)

  %ch4, %const = tfrt_fallback_async.executeop.seq(%ch3) key(2) cost(100) device("/CPU:0") "tf.Const"() { dtype = i32, value = dense<123> : tensor<i32> } : 1
  // CHECK: Using TestAllocator
  %val0 = tfrt_fallback_async.executeop.allocator(%allocator) key(0) cost(100) device("/CPU:0") "tf.Cast"(%const) { SrcT = i32, DstT = f32 } : 1
  // CHECK: Using TestAllocator
  %ch5, %val1 = tfrt_fallback_async.executeop.seq.allocator(%ch4, %allocator) key(1) cost(100) device("/CPU:0") "tf.Cast"(%val0) { SrcT = f32, DstT = i32 } : 1

  // CHECK: Tensor<type: float shape: [] values: 123>
  %ch6 = "tfrt_fallback_async.print_tensor"(%val0, %ch5)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain
  // CHECK: Tensor<type: int32 shape: [] values: 123>
  %ch7 = "tfrt_fallback_async.print_tensor"(%val1, %ch6)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch7 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_custom_allocator_async_opkernel'
func.func @test_custom_allocator_async_opkernel() -> !tfrt.chain {
  %ch0 = tfrt.new.chain

  %allocator = "tfrt_fallback_async.get_test_allocator"() : () -> (!tfrt_fallback.tf_allocator)

  %ch1 = tfrt_fallback_async.createop(%ch0) key(0) device("/CPU:0") "tf.Const"() { dtype = i32, value = dense<123> : tensor<i32> } num_args(0)
  %ch2 = tfrt_fallback_async.createop(%ch1) key(1) device("/CPU:0") "tf.TestAsyncIdentity"() { T = i32 } num_args(1)

  %ch3, %const = tfrt_fallback_async.executeop.seq(%ch2) key(0) cost(100) device("/CPU:0") "tf.Const"() { dtype = i32, value = dense<123> : tensor<i32> } : 1
  %val = tfrt_fallback_async.executeop.allocator(%allocator) key(1) cost(100) device("/CPU:0") "tf.TestAsyncIdentity"(%const) { SrcT = i32, DstT = f32 } : 1

  // CHECK: Tensor<type: int32 shape: [] values: 123>
  %ch4 = "tfrt_fallback_async.print_tensor"(%val, %ch3)
    : (!tfrt_fallback.tf_tensor, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch4 : !tfrt.chain
}
