// RUN: tfrt_translate --mlir-to-bef %s | tf_bef_executor | FileCheck %s

// CHECK: --- Running 'scalar_add_tfrt_forwarding_test'
func.func @scalar_add_tfrt_forwarding_test() {
  %one = tfrt.constant.i32 1
  %two = tfrt.constant.i32 2

  %tft_a = "tfd.constant_tensor"(%one) : (i32) -> !tfd.tensor
  %tft_b = "tfd.constant_tensor"(%two) : (i32) -> !tfd.tensor

  %tft_c = "tfd.forward_kernel"(%tft_a, %tft_b) {op_name = "ScalarAdd"}: (!tfd.tensor, !tfd.tensor) -> !tfd.tensor

  // CHECK: tensor=Tensor<type: int32 shape: [] values: 3>
  "tfd.print_tensor"(%tft_c) : (!tfd.tensor) -> ()

  tfrt.return
}

// CHECK: --- Running 'failing_kernel_tfrt_forwarding_test'
func.func @failing_kernel_tfrt_forwarding_test() -> !tfd.tensor {
  %tft = "tfd.forward_kernel"() {op_name = "FailingKernel"}: () -> !tfd.tensor  // expected-error {{runtime error: OP_REQUIRES failed at filename:999 : Internal: TFRT forwarding error!}}

  tfrt.return %tft : !tfd.tensor
}
// CHECK-NEXT: 'failing_kernel_tfrt_forwarding_test' returned <<error: OP_REQUIRES failed at filename:999 : Internal: TFRT forwarding error!>>

// CHECK: --- Running 'missing_kernel_tfrt_forwarding_test'
func.func @missing_kernel_tfrt_forwarding_test() -> !tfd.tensor {
  %tft = "tfd.forward_kernel"() {op_name = "MissingKernel"}: () -> !tfd.tensor  // expected-error {{runtime error: Not found: Could not find kernel MissingKernel in the registry.}}

  tfrt.return %tft : !tfd.tensor
}
// CHECK-NEXT: 'missing_kernel_tfrt_forwarding_test' returned <<error: Not found: Could not find kernel MissingKernel in the registry.>>


// CHECK: --- Running 'bool_attr_tfrt_forwarding_test'
func.func @bool_attr_tfrt_forwarding_test() {

  // Note that op_name is prefixed with underscore because op_name must be the
  // first attributes when all attributes are sorted by name (b/140896071).
  %tft_t = "tfd.forward_kernel"() {_op_name = "KernelWithBoolAttr", attr1_name = "testattr", attr1_value = "bool$true"}: () -> !tfd.tensor
  %tft_f = "tfd.forward_kernel"() {_op_name = "KernelWithBoolAttr", attr1_name = "testattr", attr1_value = "bool$false"}: () -> !tfd.tensor

  // CHECK: tensor=Tensor<type: bool shape: [] values: 1>
  "tfd.print_tensor"(%tft_t) : (!tfd.tensor) -> ()
  // CHECK: tensor=Tensor<type: bool shape: [] values: 0>
  "tfd.print_tensor"(%tft_f) : (!tfd.tensor) -> ()

  tfrt.return
}
