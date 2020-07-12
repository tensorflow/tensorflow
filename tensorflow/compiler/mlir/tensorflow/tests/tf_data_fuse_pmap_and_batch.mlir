// RUN: tf-opt -tf-standard-pipeline -tf-data-optimization %s -o %t && FileCheck %s < %t

module {
// CHECK-LABEL: fuse_pmap_and_batch
func @fuse_pmap_and_batch() -> tensor<!tf.variant> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "BatchDatasetV2"}} {
  %0 = "tf.Const"() {value = dense<5> : tensor<i64>} : () -> tensor<i64>
  %1 = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
  %2 = "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  %3 = "tf.Const"() {value = dense<12> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[TSLICE:.*]] = "tf.TensorSliceDataset"
  %4 = "tf.TensorSliceDataset"(%2) {device = "", output_shapes = [#tf.shape<>]} : (tensor<3xi32>) -> tensor<*x!tf.variant>
  // CHECK: "tf.MapAndBatchDataset"(%[[TSLICE]],
  // CHECK-SAME: f = @"__inference_Dataset_map_<lambda>_80",
  %5 = "tf.ParallelMapDataset"(%4, %3) {device = "",
           f = @"__inference_Dataset_map_<lambda>_80",
           output_shapes = [#tf.shape<>], output_types = [i32],
           preserve_cardinality = false, sloppy = false,
           use_inter_op_parallelism = true} : (tensor<*x!tf.variant>, tensor<i32>) -> tensor<!tf.variant>
  %6 = "tf.BatchDatasetV2"(%5, %0, %1) {device = "", output_shapes = [#tf.shape<>], output_types = [i32], parallel_copy = false} : (tensor<!tf.variant>, tensor<i64>, tensor<i1>) -> tensor<!tf.variant>
  return %6 : tensor<!tf.variant>
}

func @"__inference_Dataset_map_<lambda>_80"(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Mul"(%arg0, %0) {device = ""} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %2 = "tf.Identity"(%1) {device = ""} : (tensor<*xi32>) -> tensor<*xi32>
  return %2 : tensor<*xi32>
}
}
