// RUN: tfg-translate -mlir-to-graphdef --split-input-file %s | FileCheck %s

// CHECK: signature {
// CHECK-NEXT: name: "func_no_args_no_results"
// CHECK-NOT: input_arg
// CHECK-NOT: output_arg
tfg.func @func_no_args_no_results() -> () {
  return
}

// -----

// CHECK: signature {
// CHECK-NEXT: name: "func_no_args"
// CHECK-NEXT: output_arg
// CHECK-NOT: input_arg
tfg.func @func_no_args() -> (tensor<1xi32> {tfg.name = "ret1"}) {
  %Const, %ctl_2 = Const name("c") {dtype = i32, value = dense<0> : tensor<1xi32>} : () -> (tensor<1xi32>)
  return (%Const) : tensor<1xi32>
}

// -----

// CHECK: signature {
// CHECK-NEXT: name: "func_no_results"
// CHECK-NEXT: input_arg
// CHECK-NOT: output_arg
tfg.func @func_no_results(%arg : tensor<i32> {tfg.name = "arg"}) -> () {
  return
}
