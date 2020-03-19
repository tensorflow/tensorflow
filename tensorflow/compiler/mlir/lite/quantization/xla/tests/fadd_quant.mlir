# RUN: not tfcompile --graph=%s.pbtxt --config=%s.config.pbtxt --quantize --cpp_class="::test::fadd_quant"  2>&1 | FileCheck %s -dump-input-on-failure

# TODO(fengliuai): update this file with the progress of the implementation
// CHECK: func @main
// CHECK: %cst = constant dense<0.000000e+00> : tensor<f32>
// CHECK: %cst_0 = constant dense<1.270000e+02> : tensor<f32>
// CHECK: %cst_1 = constant dense<8> : tensor<i32>
// CHECK: %cst_2 = constant dense<false> : tensor<i1>
// CHECK: %0 = "xla_hlo.custom_call"(%arg0, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars", has_side_effect = false, name = "custom-call.9"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
// CHECK: %1 = "xla_hlo.custom_call"(%arg1, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars", has_side_effect = false, name = "custom-call.14"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
// CHECK: %2 = xla_hlo.add %0, %1 {name = "add.15"} : tensor<2x4xf32>
// CHECK: %3 = "xla_hlo.custom_call"(%2, %cst, %cst_0, %cst_1, %cst_2) {backend_config = "", call_target_name = "fake_quant_with_min_max_vars", has_side_effect = false, name = "custom-call.20"} : (tensor<2x4xf32>, tensor<f32>, tensor<f32>, tensor<i32>, tensor<i1>) -> tensor<2x4xf32>
// CHECK: %4 = "xla_hlo.tuple"(%3) {name = "tuple.22"} : (tensor<2x4xf32>) -> tuple<tensor<2x4xf32>>
// CHECK: return %4 : tuple<tensor<2x4xf32>>
// CHECK: }
