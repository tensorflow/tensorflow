// RUN: tf-opt %s -tf-xla-call-module-serialization -tf-shape-inference='enable-stablehlo-propagation=true' -tf-xla-call-module-deserialization | FileCheck %s --check-prefixes=COMMON,ENABLED
// RUN: tf-opt %s -tf-xla-call-module-serialization -tf-shape-inference='enable-stablehlo-propagation=false' -tf-xla-call-module-deserialization | FileCheck %s --check-prefixes=COMMON,DISABLED
// RUN: tf-opt %s -tf-xla-call-module-serialization -tf-standard-pipeline='enable-stablehlo-shape-propagation=true' -tf-xla-call-module-deserialization | FileCheck %s --check-prefixes=COMMON,ENABLED
// RUN: tf-opt %s -tf-xla-call-module-serialization -tf-standard-pipeline='enable-stablehlo-shape-propagation=false' -tf-xla-call-module-deserialization | FileCheck %s --check-prefixes=COMMON,DISABLED

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 130 : i32}} {

  // COMMON-LABEL: func @module_attr_updated_with_stablehlo_shape_refinement
  // COMMON-SAME: (%arg0: tensor<4xf32>) -> tensor<4xi32>
  func.func @module_attr_updated_with_stablehlo_shape_refinement(%arg0: tensor<4xf32>) -> tensor<?xi32> {
    // COMMON: "tf.XlaCallModule"
    // COMMON-SAME: _entry_function = @main,
    // COMMON-SAME: (tensor<4xf32>) -> tensor<4xi32>
    %0 = "tf.XlaCallModule"(%arg0) {Sout = [#tf_type.shape<*>], device = "", dim_args_spec = [], _entry_function = @_stablehlo_main, _stablehlo_module_attrs = { jax.uses_shape_polymorphism = true }, _stablehlo_version = "1.1.0", module = "", platforms = ["cpu"], version = 9 : i64} : (tensor<4xf32>) -> tensor<?xi32>
    func.return %0 : tensor<?xi32>
  }

  // COMMON-LABEL: func @module_attr_not_updated_without_stablehlo_output_shape_refinement
  // COMMON-SAME: (%arg0: tensor<1xf32>) -> tensor<1xi32>
  func.func @module_attr_not_updated_without_stablehlo_output_shape_refinement(%arg0: tensor<1xf32>) -> tensor<*xi32> {
    // COMMON: "tf.XlaCallModule"
    // COMMON-SAME: _entry_function = @main_0,
    // COMMON-SAME: (tensor<1xf32>) -> tensor<1xi32>
    %0 = "tf.XlaCallModule"(%arg0) {Sout = [#tf_type.shape<*>], device = "", dim_args_spec = [], _entry_function = @_stablehlo_main_0, _stablehlo_module_attrs = { jax.uses_shape_polymorphism = true }, _stablehlo_version = "1.1.0", module = "", platforms = ["cpu"], version = 9 : i64} : (tensor<1xf32>) -> tensor<*xi32>
    func.return %0 : tensor<*xi32>
  }

  // COMMON-LABEL: func @xla_call_module_shape_refinement_failure_ok
  func.func @xla_call_module_shape_refinement_failure_ok(%arg0: tensor<?x1024xf32>) -> tensor<*xf32> attributes {tf._original_func_name = "main_0"} {
    %cst = "tf.Const"() <{value = dense<1.000000e+00> : tensor<1024x3xf32>}> : () -> tensor<1024x3xf32>
    // COMMON: tf.XlaCallModule
    // COMMON-SAME: _entry_function = @main_1,
    // COMMON-SAME: (tensor<?x1024xf32>, tensor<1024x3xf32>) -> tensor<?x3xf32>
    %0 = "tf.XlaCallModule"(%arg0, %cst) {Sout = [#tf_type.shape<?x3>], dim_args_spec = [], _entry_function = @_stablehlo_main_1, _stablehlo_module_attrs = { jax.uses_shape_polymorphism = true }, _stablehlo_version = "1.1.0", module = "", platforms = ["cpu", "tpu"], version = 9 : i64} : (tensor<?x1024xf32>, tensor<1024x3xf32>) -> tensor<*xf32>
    // COMMON: return %0 : tensor<?x3xf32>
    return %0 : tensor<*xf32>
  }

  // ENABLED-LABEL: func.func private @main({{.*}}: tensor<4xf32>) -> tensor<4xi32>
  // ENABLED: stablehlo.convert {{.*}} : (tensor<4xf32>) -> tensor<4xi32>
  // ENABLED: return {{.*}} : tensor<4xi32>
  // DISABLED-LABEL: func.func private @main({{.*}}: tensor<4xf32>) -> tensor<?xi32>
  // DISABLED: stablehlo.convert {{.*}} : (tensor<4xf32>) -> tensor<?xi32>
  // DISABLED: return {{.*}} : tensor<?xi32>
  func.func private @_stablehlo_main(%arg0 : tensor<4xf32>) -> tensor<?xi32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convert %arg0 : (tensor<4xf32>) -> tensor<?xi32>
    return %0 : tensor<?xi32>
  }

  // COMMON-LABEL: func.func private @main_0({{.*}}: tensor<1xf32>) -> tensor<1xi32>
  // COMMON: stablehlo.convert {{.*}} : (tensor<1xf32>) -> tensor<?xi32>
  // COMMON: return {{.*}} : tensor<1xi32>
  func.func private @_stablehlo_main_0(%arg0 : tensor<1xf32>) -> tensor<1xi32> attributes {_from_xla_call_module} {
    %0 = stablehlo.convert %arg0 : (tensor<1xf32>) -> tensor<?xi32>
    %1 = stablehlo.constant dense<1> : tensor<1xi32>
    return %1 : tensor<1xi32>
  }

  // COMMON-LABEL: func.func private @main_1({{.*}}: tensor<i32>, {{.*}}: tensor<?x1024xf32>, {{.*}}: tensor<1024x3xf32>) -> tensor<?x3xf32>
  // COMMON: return {{.*}} : tensor<?x3xf32>
  func.func private @_stablehlo_main_1(%arg0: tensor<i32>, %arg1: tensor<?x1024xf32>, %arg2: tensor<1024x3xf32>) -> tensor<?x3xf32> attributes {_from_xla_call_module} {
      %0 = stablehlo.constant dense<0> : tensor<i32>
      %1 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x1024xf32>) -> tensor<i32>
      %2 = stablehlo.compare  EQ, %0, %1 : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.custom_call @shape_assertion(%2) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
      %3 = stablehlo.dot_general %arg1, %arg2,
          batching_dims = [] x [], contracting_dims = [1] x [0]
          {mhlo.frontend_attributes = {grad_x = "false", grad_y = "false"}}
        : (tensor<?x1024xf32>, tensor<1024x3xf32>) -> tensor<?x3xf32>
      return %3 : tensor<?x3xf32>
    }
}
