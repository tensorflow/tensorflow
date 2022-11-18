// RUN: tf-tfrt-opt -tf-executor-to-tfrt-pipeline %s | FileCheck %s

// Checks that the ops' function attribute references the original function name
// `funcB` for `funcB_renamed` after the module is lowered to TFRT. Note that,
// `funcB_renamed` are called twice, so `CreateGuaranteeAllFuncsOneUsePass` will
// make a replicaion of `funcB_renamed` with a different name.

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 567 : i32}} {
  // CHECK-LABEL: @funcA
  func.func @funcA() -> (tensor<!tf_type.variant>, tensor<!tf_type.variant>) attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "flatmapdataset__4_RetVal"}} {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf.Const"() {device = "/device:CPU:0", value = dense<5> : tensor<i64>} : () -> tensor<i64>
    %2 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %3 = "tf.RangeDataset"(%0, %1, %2) {device = "/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<!tf_type.variant>
    // CHECK: tfrt_fallback_async.executeop key({{[0-9]+}}) cost({{.*}}) device("/device:CPU:0") "tf.FlatMapDataset"({{.*}}) {Targuments = [], metadata = "", output_shapes = [#corert.shape<>], output_types = [i64]} {f = "funcB"} : 1
    %4 = "tf.FlatMapDataset"(%3) {Targuments = [], device = "/device:CPU:0", f = @funcB_renamed, output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
    %5 = "tf.RangeDataset"(%1, %2, %0) {device = "/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<!tf_type.variant>
    // CHECK: tfrt_fallback_async.executeop key({{[0-9]+}}) cost({{.*}}) device("/device:CPU:0") "tf.FlatMapDataset"({{.*}}) {Targuments = [], metadata = "", output_shapes = [#corert.shape<>], output_types = [i64]} {f = "funcB"} : 1
    %6 = "tf.FlatMapDataset"(%5) {Targuments = [], device = "/device:CPU:0", f = @funcB_renamed, output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
    func.return %4, %6 : tensor<!tf_type.variant>, tensor<!tf_type.variant>
  }
  // CHECK-LABEL: @funcB_renamed
  func.func private @funcB_renamed(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}) -> tensor<!tf_type.variant> attributes {tf._original_func_name = "funcB", tf._tf_data_function = true, tf.signature.is_stateful} {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf.Const"() {device = "/device:CPU:0", value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %2 = "tf.Const"() {device = "/device:CPU:0", value = dense<5> : tensor<i64>} : () -> tensor<i64>
    %3 = "tf.RangeDataset"(%0, %2, %1) {device = "/device:CPU:0", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<!tf_type.variant>
    // CHECK: tfrt_fallback_async.executeop key({{[0-9]+}}) cost({{.*}}) device("/device:CPU:0") "tf.MapDataset"({{.*}}) {Targuments = [], metadata = "", output_shapes = [#corert.shape<>], output_types = [i64], preserve_cardinality = true, use_inter_op_parallelism = true} {f = "funcC"} : 1
    %4 = "tf.MapDataset"(%3) {device = "/device:CPU:0", f = @funcC_renamed, f._tf_data_function = true, output_shapes = [#tf_type.shape<>], output_types = [i64], preserve_cardinality = true, use_inter_op_parallelism = true, metadata = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
    %5 = "tf.Identity"(%4) {device = "/device:CPU:0"} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
    func.return %5 : tensor<!tf_type.variant>
  }
  // CHECK-LABEL: @funcC_renamed
  func.func private @funcC_renamed(%arg0: tensor<i64> {tf._user_specified_name = "args_0"}) -> tensor<i64> attributes {tf._tf_data_function = true, tf._original_func_name = "funcC"} {
    %0 = "tf.Const"() {device = "/device:CPU:0", value = dense<2> : tensor<i64>} : () -> tensor<i64>
    %1 = "tf.Mul"(%arg0, %0) {device = "/device:CPU:0"} : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %2 = "tf.Identity"(%1) {device = "/device:CPU:0"} : (tensor<i64>) -> tensor<i64>
    func.return %2 : tensor<i64>
  }
}
