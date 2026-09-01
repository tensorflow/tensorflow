// Covers the direct TF XLA op -> StableHLO -> TF path in -tfl-prepare-tf.
// RUN: litert-opt -tfl-prepare-tf %s | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 268 : i32}} {

// Constant index + collapsed_slice_dims -> slice + reshape.
// dnums: offset_dims=[0], collapsed_slice_dims=[0], start_index_map=[0], index_vector_dim=0
// CHECK-LABEL: xla_gather_const_index_collapsed
func.func @xla_gather_const_index_collapsed(%arg0 : tensor<4x5xf32>) -> tensor<5xf32> {
  %0 = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<[1, 5]> : tensor<2xi32>} : () -> tensor<2xi32>
  %2 = "tf.XlaGather"(%arg0, %0, %1) {device = "", dimension_numbers = "\0A\01\00\12\01\00\1A\01\00", indices_are_sorted = false} : (tensor<4x5xf32>, tensor<1xi32>, tensor<2xi32>) -> tensor<5xf32>
  func.return %2 : tensor<5xf32>
// CHECK-NOT: tf.XlaGather
// CHECK-NOT: stablehlo.
// CHECK-NOT: mhlo.
// CHECK: "tf.StridedSlice"(%arg0
// CHECK-SAME: -> tensor<1x5xf32>
// CHECK: "tf.Reshape"
// CHECK-SAME: -> tensor<5xf32>
}

// Non-constant indices -> generic stablehlo.gather -> tf.GatherNd (odml ConvertGatherOp).
// dnums: offset_dims=[1], collapsed_slice_dims=[0], start_index_map=[0], index_vector_dim=1
// CHECK-LABEL: xla_gather_dynamic_index
func.func @xla_gather_dynamic_index(%arg0 : tensor<5x4xf32>, %arg1 : tensor<3x1xi32>) -> tensor<3x4xf32> {
  %0 = "tf.Const"() {value = dense<[1, 4]> : tensor<2xi32>} : () -> tensor<2xi32>
  %1 = "tf.XlaGather"(%arg0, %arg1, %0) {device = "", dimension_numbers = "\0A\01\01\12\01\00\1A\01\00 \01", indices_are_sorted = false} : (tensor<5x4xf32>, tensor<3x1xi32>, tensor<2xi32>) -> tensor<3x4xf32>
  func.return %1 : tensor<3x4xf32>
// CHECK-NOT: tf.XlaGather
// CHECK-NOT: stablehlo.
// CHECK-NOT: mhlo.
// CHECK: "tf.{{GatherNd|GatherV2}}"(%arg0
}


}
