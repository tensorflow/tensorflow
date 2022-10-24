// RUN: tf-tfrt-opt %s -pass-pipeline='func.func(canonicalize)' | FileCheck %s

// CHECK-LABEL: func @test_const_tensor_canonicalization_single_denst_tensor_operand
func.func @test_const_tensor_canonicalization_single_denst_tensor_operand() -> !tfrt_fallback.tf_tensor {
  // CHECK: tfrt_fallback_async.const_dense_tensor
  %a = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %ra = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  tfrt.return %ra : !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_const_tensor_canonicalization_single_string_operand
func.func @test_const_tensor_canonicalization_single_string_operand() -> !tfrt_fallback.tf_tensor {
  // CHECK: tfrt_fallback_async.const_string_tensor
  %a = corert.const_string_tensor {shape = [2], value = ["string", "tensor"]}
  %ra = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)
  tfrt.return %ra : !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_const_tensor_canonicalization_multiple_operands
func.func @test_const_tensor_canonicalization_multiple_operands() -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) {
  // CHECK:      tfrt_fallback_async.const_dense_tensor
  // CHECK-NEXT: tfrt_fallback_async.const_string_tensor
  %a = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %b = corert.const_string_tensor {shape = [2], value = ["string", "tensor"]}
  %ra, %rb = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a, %b {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  tfrt.return %ra, %rb : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// Tests the case where the conversion op is partially canonicalizable.
// CHECK-LABEL: func @test_const_tensor_canonicalization_mixed_operands
// CHECK-SAME: ([[arg0:%.*]]: !corert.tensorhandle, [[arg1:%.*]]: !corert.tensorhandle)
func.func @test_const_tensor_canonicalization_mixed_operands(%arg0: !corert.tensorhandle, %arg1: !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) {
  %a = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %b = corert.const_dense_tensor dense<[false, true]> : tensor<2xi1>
  // CHECK: [[b:%.*]] = tfrt_fallback_async.const_dense_tensor dense<[false, true]> : tensor<2xi1>
  // CHECK-NEXT:      [[a:%.*]] = tfrt_fallback_async.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  // CHECK-NEXT: tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg0]]
  // CHECK-NEXT: tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg1]]
  %ra, %rarg0, %rb, %rarg1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %a, %arg0, %b, %arg1 {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  tfrt.return %ra, %rarg0, %rb, %rarg1 : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// Tests that if the conversion op is partially canonicalizable, the non-canonicalizable operands are always separated into individual conversion ops.
// CHECK-LABEL: func @test_const_tensor_canonicalization_mixed_operands_no_consolidation
// CHECK-SAME: ([[arg0:%.*]]: !corert.tensorhandle, [[arg1:%.*]]: !corert.tensorhandle)
func.func @test_const_tensor_canonicalization_mixed_operands_no_consolidation(%arg0: !corert.tensorhandle, %arg1: !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor) {
  // CHECK-NEXT: tfrt_fallback_async.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  // CHECK-NEXT: tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg0]]
  // CHECK-NEXT: tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor [[arg1]]

  %a = corert.const_dense_tensor dense<[true, false]> : tensor<2xi1>
  %rarg0, %rarg1, %ra = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %arg0, %arg1, %a {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle, !corert.tensorhandle, !corert.tensorhandle) -> (!tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor)

  tfrt.return %rarg0, %rarg1, %ra : !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor, !tfrt_fallback.tf_tensor
}

// CHECK-LABEL: func @test_remove_double_conversion
// CHECK-SAME: ([[arg:%.*]]: !tfrt_fallback.tf_tensor
func.func @test_remove_double_conversion(%arg: !tfrt_fallback.tf_tensor) -> (!tfrt_fallback.tf_tensor) {
  // CHECK-NOT: fallback_tensor_to_corert_tensorhandle
  // CHECK-NOT: corert_tensorhandle_to_fallback_tensor

  %0 = tfrt_fallback_async.fallback_tensor_to_corert_tensorhandle %arg {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!tfrt_fallback.tf_tensor) -> (!corert.tensorhandle)
  %1 = tfrt_fallback_async.corert_tensorhandle_to_fallback_tensor %0 {_tfrt_cost = 1 : i64, device = "/CPU:0"} : (!corert.tensorhandle) -> (!tfrt_fallback.tf_tensor)

  // CHECK: tfrt.return [[arg]]
  tfrt.return %1 : !tfrt_fallback.tf_tensor
}
