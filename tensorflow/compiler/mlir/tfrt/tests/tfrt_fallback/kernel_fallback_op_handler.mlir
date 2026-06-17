// RUN: tfrt_fallback_translate -mlir-to-bef %s | tf_bef_executor | FileCheck %s


// CHECK-LABEL: --- Not running 'register_op_handler' because it has arguments.
func.func @register_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  %op_handler = "corert.create_kernel_fallback_op_handler"() : () -> !corert.ophandler
  %ch = corert.register_op_handler %op_handler "tfkernel0"
  tfrt.return %ch : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'get_op_handler' because it has arguments.
func.func @get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  %tfkernel0 = corert.get_op_handler %ch0 "tfkernel0"
  tfrt.return %ch0 : !tfrt.chain
}

// CHECK-LABEL: --- Not running 'failed_get_op_handler' because it has arguments.
func.func @failed_get_op_handler(%ch0: !tfrt.chain) -> !tfrt.chain {
  // expected-error @+1 {{runtime error: op_handler not found}}
  %tf0 = corert.get_op_handler %ch0 "tfkernel0"
  tfrt.return %ch0 : !tfrt.chain
}

// CHECK-LABEL: --- Running 'test_op_handler_kernels'
func.func @test_op_handler_kernels()  -> !tfrt.chain {
  %ch0 = tfrt.new.chain
  %ch1 = tfrt.call @failed_get_op_handler(%ch0) : (!tfrt.chain) -> !tfrt.chain
  %ch2 = tfrt.call @register_op_handler(%ch1) : (!tfrt.chain) -> !tfrt.chain
  %ch3 = tfrt.call @get_op_handler(%ch2) : (!tfrt.chain) -> !tfrt.chain
  tfrt.return %ch3 : !tfrt.chain
}
