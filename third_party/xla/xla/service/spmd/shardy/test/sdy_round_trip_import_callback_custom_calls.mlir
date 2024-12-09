// RUN: sdy_opt %s -xla-sdy-round-trip-import-callback-custom-calls 2>&1 | FileCheck %s

func.func @callback_no_result(%arg0: tensor<f64>) {
  %c = stablehlo.constant dense<56238273106176> : tensor<i64>
  // CHECK:      stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {
  // CHECK-SAME:   api_version = 2 : i32, backend_config = "56238273106176",
  // CHECK-SAME:   has_side_effect = true,
  // CHECK-SAME:   operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>],
  // CHECK-SAME:   result_layouts = [dense<> : tensor<0xindex>]
  // CHECK-SAME: } : (tensor<i64>, tensor<f64>) -> tensor<i64>
  stablehlo.custom_call @xla_python_cpu_callback(%c, %arg0) {api_version = 2 : i32, backend_config = "56238273106176", has_side_effect = true, operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>], result_layouts = []} : (tensor<i64>, tensor<f64>) -> ()
  return
}
