// RUN: tfrt_translate -mlir-to-bef %s | tf_bef_executor --test_init_function=register_op_handlers | FileCheck %s --dump-input=fail

func.func @register_op_handlers() {
  %fallback = "corert.create_runtime_fallback_op_handler"() {tf_device_name="/device:CPU:0"} : () -> !corert.ophandler
  %cpu = "corert.create_cpu_op_handler"(%fallback) : (!corert.ophandler) -> !corert.ophandler
  corert.register_op_handler %fallback "tf"
  corert.register_op_handler %cpu "cpu"
  tfrt.return
}

// CHECK: --- Running 'transfer_to_host'
func.func @transfer_to_host() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %rtfb_handler = "corert.create_runtime_fallback_op_handler"() {tf_device_name="/device:CPU:0"} : () -> !corert.ophandler
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th0 = corert.executeop(%rtfb_handler) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<1xf32> } : 1

  %th1 = corert.executeop(%rtfb_handler) "tf.Exp" (%th0) { T = f32 } : 1

  // TFRuntimeFallbackT->DHT
  %th2_tensor_type = corert.get_dst_tensor_type %th1, %cpu_device
  %th2 = corert.transfer %th1, %cpu_device, %th2_tensor_type

  // CHECK: DenseHostTensor dtype = f32, shape = [1], values = [2.718282e+00]
  %ch1 = "corert.print_tensorhandle"(%th2, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK: --- Running 'scalar_to_runtime_fallback'
func.func @scalar_to_runtime_fallback() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th1 = corert.executeop(%cpu) "tfrt_test.create_from_scalar"()
   {shape = [2: i64, 2: i64], value = 1: i32} : 1

  // ScalarHT->TFRuntimeFallbackT
  %th2_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "RuntimeFallback" } : () -> !tfrt.tensor_type
  %th2 = corert.transfer %th1, %cpu_device, %th2_tensor_type

  // TFRuntimeFallbackT->DHT
  %th3_tensor_type = corert.get_dst_tensor_type %th2, %cpu_device
  %th3 = corert.transfer %th2, %cpu_device, %th3_tensor_type

  // CHECK: DenseHostTensor dtype = i32, shape = [2, 2], values = [1, 1, 1, 1]
  %ch1 = "corert.print_tensorhandle"(%th3, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch1 : !tfrt.chain
}

// CHECK: --- Running 'dht_to_runtime_fallback'
func.func @dht_to_runtime_fallback() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th1 = corert.executeop(%cpu) "tfrt_test.create_dense_tensor"()
    { shape = [2: i64, 2: i64], values = [42 : i32] } : 1

  // DHT->TFRuntimeFallbackT
  %th2_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "RuntimeFallback" } : () -> !tfrt.tensor_type
  %th2 = corert.transfer %th1, %cpu_device, %th2_tensor_type

  // TFRuntimeFallbackT->DHT
  %th3_tensor_type = corert.get_dst_tensor_type %th2, %cpu_device
  %th3 = corert.transfer %th2, %cpu_device, %th3_tensor_type

  // CHECK: DenseHostTensor dtype = i32, shape = [2, 2], values = [42, 42, 42, 42]
  %ch1 = "corert.print_tensorhandle"(%th3, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // DHT->TFKernelFallbackT
  %th4_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "KernelFallback" } : () -> !tfrt.tensor_type
  %th4 = corert.transfer %th1, %cpu_device, %th4_tensor_type

  // TFKernelFallbackT->TFRuntimeFallbackT
  %th5_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "RuntimeFallback" } : () -> !tfrt.tensor_type
  %th5 = corert.transfer %th4, %cpu_device, %th5_tensor_type

  // TFRuntimeFallbackT->TFKernelFallbackT
  %th6_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "KernelFallback" } : () -> !tfrt.tensor_type
  %th6 = corert.transfer %th5, %cpu_device, %th6_tensor_type

  // TFKernelFallbackT->DHT
  %th7_tensor_type = corert.get_dst_tensor_type %th6, %cpu_device
  %th7 = corert.transfer %th6, %cpu_device, %th7_tensor_type

  // CHECK: DenseHostTensor dtype = i32, shape = [2, 2], values = [42, 42, 42, 42]
  %ch2 = "corert.print_tensorhandle"(%th7, %ch1)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

// CHECK: --- Running 'sht_to_fallback'
func.func @sht_to_fallback() -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %cpu = corert.get_op_handler %ch0 "cpu"
  %cpu_device = "tfrt.get_device"(%ch0) {device_name="CPU:0"} : (!tfrt.chain) -> !tfrt.device

  %th1 = corert.const_string_tensor {shape = [2], value = ["uiuc", "berkeley"]}

  // SHT->TFRuntimeFallbackT
  %th2_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "RuntimeFallback" } : () -> !tfrt.tensor_type
  %th2 = corert.transfer %th1, %cpu_device, %th2_tensor_type

  // TFRuntimeFallbackT->SHT
  %th3_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "StringHost" } : () -> !tfrt.tensor_type
  %th3 = corert.transfer %th2, %cpu_device, %th3_tensor_type

  // CHECK: StringHostTensor shape = [2], values = ["uiuc", "berkeley"]
  %ch1 = "corert.print_tensorhandle"(%th3, %ch0)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  // SHT->TFKernelFallbackT
  %th4_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "KernelFallback" } : () -> !tfrt.tensor_type
  %th4 = corert.transfer %th1, %cpu_device, %th4_tensor_type

  // TFKernelFallbackT->TFRuntimeFallbackT
  %th5_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "RuntimeFallback" } : () -> !tfrt.tensor_type
  %th5 = corert.transfer %th4, %cpu_device, %th5_tensor_type

  // TFRuntimeFallbackT->TFKernelFallbackT
  %th6_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "KernelFallback" } : () -> !tfrt.tensor_type
  %th6 = corert.transfer %th5, %cpu_device, %th6_tensor_type

  // TFKernelFallbackT->SHT
  %th7_tensor_type = "tfrt_test.get_static_tensor_type"()
    { tensor_type = "StringHost" } : () -> !tfrt.tensor_type
  %th7 = corert.transfer %th6, %cpu_device, %th7_tensor_type

  // CHECK: StringHostTensor shape = [2], values = ["uiuc", "berkeley"]
  %ch2 = "corert.print_tensorhandle"(%th7, %ch1)
    : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain

  tfrt.return %ch2 : !tfrt.chain
}

