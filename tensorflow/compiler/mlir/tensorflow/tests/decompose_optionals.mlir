// RUN: tf-opt %s --tf-decompose-optionals --split-input-file | FileCheck %s

// CHECK-LABEL: @from_value
func.func @from_value(%arg0: tensor<f32>) {
  // CHECK-NOT: Optional
  %0 = "tf.OptionalFromValue"(%arg0) : (tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  return
}

// -----

// CHECK-LABEL: @get_value
func.func @get_value(%arg0: tensor<!tf_type.variant<tensor<f32>>>) {
  // CHECK-NOT: Optional
  %1 = "tf.OptionalGetValue"(%arg0) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<f32>
  return
}

// -----

// CHECK-LABEL: @none
func.func @none(%arg0: tensor<!tf_type.variant<tensor<f32>>>) {
  // CHECK-NOT: Optional
  %0 = "tf.OptionalNone"() : () -> tensor<!tf_type.variant<tensor<f32>>>
  return
}

// -----

// CHECK-LABEL: @partitioned_calls
func.func @partitioned_calls(%arg0: tensor<f32>, %arg1: tensor<f32>) -> (tensor<f32>, tensor<!tf_type.variant<tensor<f32>>>) {
  // CHECK-NOT: Optional
  %0 = "tf.OptionalFromValue"(%arg0) : (tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %1 = "tf.PartitionedCall"(%0) {
      config = "",
      config_proto = "",
      executor_type = "",
      f = @identity} : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>>
  %2 = "tf.OptionalGetValue"(%1) : (tensor<!tf_type.variant<tensor<f32>>>) -> tensor<f32>
  %3 = "tf.OptionalNone"() : () -> tensor<!tf_type.variant<tensor<f32>>>
  return %2, %3 : tensor<f32>, tensor<!tf_type.variant<tensor<f32>>>
}

func.func private @identity(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  return %arg0 : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// CHECK-LABEL: @leaves_remote_calls_alone
func.func @leaves_remote_calls_alone(%arg0: tensor<!tf_type.string>, %arg1: tensor<!tf_type.string>, %arg2: tensor<i64>) {
  // CHECK: RemoteCall
  // CHECK-SAME: (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>) -> tensor<!tf_type.string>
  %0 = "tf.RemoteCall"(%arg1, %arg0, %arg2) <{f = @__inference__next_func_3760}> {device = ""} : (tensor<!tf_type.string>, tensor<!tf_type.string>, tensor<i64>) -> tensor<!tf_type.string>
  return
}
func.func private @__inference__next_func_3760(%arg0: tensor<!tf_type.string> {tf._user_specified_name = "string_handle"}, %arg1: tensor<i64> {tf._user_specified_name = "MultiDeviceIteratorInit"}) -> tensor<!tf_type.string> attributes {tf._construction_context = "kEagerRuntime", tf._input_shapes = [#tf_type.shape<>, #tf_type.shape<>], tf.signature.is_stateful} {
  %cst = "tf.Const"() <{value = dense<1> : tensor<i32>}> {device = ""} : () -> tensor<i32>
  %0 = "tf.MultiDeviceIteratorFromStringHandle"(%arg0) <{output_shapes = [#tf_type.shape<>], output_types = [!tf_type.string]}> {device = ""} : (tensor<!tf_type.string>) -> tensor<!tf_type.resource>
  %1 = "tf.MultiDeviceIteratorGetNextFromShard"(%0, %cst, %arg1) {device = ""} : (tensor<!tf_type.resource>, tensor<i32>, tensor<i64>) -> tensor<!tf_type.string>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<!tf_type.string>) -> tensor<!tf_type.string>
  return %2 : tensor<!tf_type.string>
}

// -----

// CHECK-LABEL: @if
func.func @if() {
  // CHECK-NOT: Optional
  %0 = builtin.unrealized_conversion_cast to tensor<5xi1>
  %1 = "tf.OptionalNone"() : () -> tensor<!tf_type.variant<tensor<f32>>>
  %2 = "tf.If"(%0, %1) <{else_branch = @false, is_stateless = false, then_branch = @true}>
      : (tensor<5xi1>, tensor<!tf_type.variant<tensor<f32>>>) -> (tensor<!tf_type.variant<tensor<f32>>>)
  return
}
func.func private @false(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  return %arg0 : tensor<!tf_type.variant<tensor<f32>>>
}
func.func private @true(%arg0: tensor<!tf_type.variant<tensor<f32>>>) -> tensor<!tf_type.variant<tensor<f32>>> {
  return %arg0 : tensor<!tf_type.variant<tensor<f32>>>
}

// -----

// CHECK-LABEL: @if_with_variants_in_branches
func.func @if_with_variants_in_branches() {
  // CHECK-NOT: Optional
  %0 = builtin.unrealized_conversion_cast to tensor<5xi1>
  %2 = "tf.If"(%0) <{else_branch = @false, is_stateless = false, then_branch = @true}>
      : (tensor<5xi1>) -> (tensor<!tf_type.variant<tensor<f32>>>)
  return
}
func.func private @false() -> tensor<!tf_type.variant<tensor<f32>>> {
  %0 = builtin.unrealized_conversion_cast to tensor<f32>
  %1 = "tf.OptionalFromValue"(%0) : (tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  return %1 : tensor<!tf_type.variant<tensor<f32>>>
}
func.func private @true() -> tensor<!tf_type.variant<tensor<f32>>> {
  %1 = "tf.OptionalNone"() : () -> tensor<!tf_type.variant<tensor<f32>>>
  return %1 : tensor<!tf_type.variant<tensor<f32>>>
}
