// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor | FileCheck %s


// CHECK-LABEL: --- Not running 'register_runtime_fallback_op_handler_chain' because it has arguments.
func.func @register_runtime_fallback_op_handler_chain(%ch0: !tfrt.chain) -> !tfrt.chain {
  %runtime_fallback = "corert.create_runtime_fallback_op_handler"() {tf_device_name="/device:CPU:0"} : () -> !corert.ophandler
  %ch = corert.register_op_handler %runtime_fallback "tf0"
  tfrt.return %ch : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'get_runtime_fallback_op_handler' because it has arguments.
func.func @get_runtime_fallback_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  %tf0 = corert.get_op_handler %ch0 "tf0"

  %th = corert.executeop(%tf0) "tf.Const"()
      { dtype = f32, value = dense<1.0> : tensor<2x2xf32> } : 1

  %ch2 = "corert.print_tensorhandle" (%th, %ch0) : (!corert.tensorhandle, !tfrt.chain) -> !tfrt.chain
  tfrt.return %ch2 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'failed_runtime_fallback_get_op_handler' because it has arguments.
func.func @failed_runtime_fallback_get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  // expected-error @+1 {{runtime error: op_handler not found}}
  %tf0 = corert.get_op_handler %ch0 "tf0"
  tfrt.return %ch0 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_runtime_fallback_op_handler_chain_kernels'
func.func @test_runtime_fallback_op_handler_chain_kernels()  -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @failed_runtime_fallback_get_op_handler(%ch0) : (!tfrt.chain) -> !tfrt.chain
  %ch2 = tfrt.call @register_runtime_fallback_op_handler_chain(%ch1) : (!tfrt.chain) -> !tfrt.chain
  // CHECK: RuntimeFallbackTensor dtype = float, shape = [2, 2], values = [1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00]
  %ch3 = tfrt.call @get_runtime_fallback_op_handler(%ch2) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return %ch3 : !tfrt.chain
}
