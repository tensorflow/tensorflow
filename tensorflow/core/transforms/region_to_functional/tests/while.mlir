// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s

// Check that `WhileRegion` is correctly converted back to functional form.

tfg.graph #tf_type.version<producer = 42, min_consumer = 33> {
  // CHECK: %[[INIT:.*]], %[[CTL:.*]] = Init name("[[INIT_NAME:.*]]")
  %Init, %ctl = Init name("init") : () -> (tensor<*xi32>)
  // CHECK: %[[CONST:.*]], %[[CTL_0:.*]] = Const name("[[CONST_NAME:.*]]")
  %Const, %ctl_0 = Const name("const") : () -> (tensor<*xf32>)
  // CHECK: %[[OUTS:.*]]:2, %[[CTL_1:.*]] = While(%[[INIT]], %[[CONST]]) [%[[CTL]]]
  // CHECK-SAME: {body = #tf_type.func<@foo, {}>, cond = #tf_type.func<@[[COND_FUNC:.*]], {}>,
  %While, %ctl_1 = WhileRegion(%Init) [%ctl] {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    %True, %ctl_2 = True name("true") : () -> (tensor<*xi1>)
    condition %True : tensor<*xi1> (%arg0) [%ctl_2] : tensor<*xi32>
  } do {
  ^bb0(%arg0: tensor<*xi32>, %arg1: !tf_type.control):
    %Double, %ctl_2 = Double(%arg0, %Const) name("double") : (tensor<*xi32>, tensor<*xf32>) -> (tensor<*xi32>)
    yield(%Double) [%ctl_0] : tensor<*xi32>
  } {body_region_attrs = #tfg.region_attrs<{sym_name = "foo", tf._a} [{tf._some_attr}] [{tf._other_attr}]>,
     cond_region_attrs = #tfg.region_attrs<{tf._b} [{}] [{}]>,
     parallel_iterations = 10 : i64} : (tensor<*xi32>) -> (tensor<*xi32>)
}

// CHECK: tfg.func @[[COND_FUNC]]
// CHECK-SAME: (%[[INIT_NAME]]_tfg_result_0: tensor<{{.*}}> {tfg.name = "[[INIT_NAME]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK-NEXT:  %[[CONST_NAME]]_tfg_result_0: tensor<{{.*}}> {tfg.name = "[[CONST_NAME]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK-NEXT:  -> (tensor<{{.*}}xi1> {tfg.name = "[[TRUE_NAME:.*]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK-NEXT: attributes {tf._b} {
// CHECK-NEXT:   %[[TRUE:.*]], %[[CTL:.*]] = True name("[[TRUE_NAME]]")
// CHECK-NEXT:   return(%[[TRUE]]) [%[[CTL]] {tfg.name = "[[TRUE_NAME]]_tfg_result_1"}]

// CHECK: tfg.func @foo
// CHECK-SAME: (%[[INIT_NAME]]_tfg_result_0: tensor<{{.*}}> {tf._some_attr, tfg.name = "[[INIT_NAME]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK-NEXT:  %[[CONST_NAME]]_tfg_result_0: tensor<{{.*}}> {tfg.name = "[[CONST_NAME]]_tfg_result_0", tfg.regenerate_output_shapes})
// CHECK-NEXT:  -> (tensor<{{.*}}> {tf._other_attr, tfg.name = "[[DOUBLE_NAME:.*]]_tfg_result_0", tfg.regenerate_output_shapes},
// CHECK-NEXT:      tensor<{{.*}}> {tfg.name = "[[CONST_NAME]]_tfg_result_0_0", tfg.regenerate_output_shapes})
// CHECK-NEXT: attributes {tf._a} {
// CHECK-NEXT:   %[[DOUBLE:.*]], %[[CTL:.*]] = Double(%[[INIT_NAME]]_tfg_result_0, %[[CONST_NAME]]_tfg_result_0) name("[[DOUBLE_NAME]]")
// CHECK-NEXT:   return(%[[DOUBLE]], %[[CONST_NAME]]_tfg_result_0) [%[[CONST_NAME]]_tfg_result_0.ctl {tfg.name = "[[CONST_NAME]]_tfg_result_0_1"}]
// CHECK-NEXT: }
